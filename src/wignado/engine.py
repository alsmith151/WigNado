from __future__ import annotations

"""wignado.engine

Core utilities and conversion helpers for working with genomic
bigWig-like data and (optionally) converting to TileDB-backed stores.

This module intentionally keeps heavy external dependencies at top-level
but many functions are pure and can be unit-tested with lightweight stubs.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable
import warnings

try:
    import pybigtools
except Exception:  # pragma: no cover - optional dependency
    pybigtools = None
    warnings.warn("pybigtools is not installed; fallback code paths will be limited")

import numpy as np
import psutil

try:
    import tiledb
except Exception:  # pragma: no cover - optional dependency
    tiledb = None
    warnings.warn("tiledb is not installed; TileDB output disabled")

try:
    import polars as pl
except Exception:  # pragma: no cover - optional dependency
    pl = None
    warnings.warn("polars is not installed; slower numpy paths will be used")
from loguru import logger
from numba import jit, prange
from pydantic import BaseModel, Field, field_validator
from rich.console import Console
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# thread-local storage for per-thread bigWig handles
_THREAD_LOCAL = threading.local()
from rich.progress import (
    BarColumn, Progress, SpinnerColumn, TaskProgressColumn,
    TextColumn, TimeRemainingColumn
)
from rich.table import Table

warnings.filterwarnings("ignore")

console = Console()

logger.remove()
logger.add(
    sink=lambda msg: console.print(f"[dim]{msg}[/dim]", markup=False),
    format="{time:HH:mm:ss} | {level: <8} | {message}",
    level="INFO",
)


@runtime_checkable
class Closeable(Protocol):
    def close(self) -> None: ...


class GenomicRegion(BaseModel):
    model_config = {"validate_assignment": True, "extra": "forbid"}

    chrom: str = Field(..., min_length=1)
    start: int = Field(..., ge=0)
    end: int = Field(..., gt=0)

    @field_validator("end")
    @classmethod
    def validate_interval(cls, v: int, info) -> int:
        if "start" in info.data and v <= info.data["start"]:
            raise ValueError(
                f"End position ({v}) must be greater than start ({info.data['start']})"
            )
        return v

    """A simple validated genomic region.

    Attributes
    - chrom: chromosome/contig name
    - start: 0-based inclusive start
    - end: 0-based exclusive end (must be > start)
    """

    @property
    def length(self) -> int:
        return self.end - self.start

    def __str__(self) -> str:
        return f"{self.chrom}:{self.start:,}-{self.end:,}"


class ConverterConfig(BaseModel):
    model_config = {"validate_assignment": True, "extra": "forbid"}

    chunk_size: int = Field(default=2_000_000, ge=10_000)
    tile_size: int = Field(default=10_000, ge=1_000)
    compression_level: int = Field(default=3, ge=1, le=9)
    buffer_size: int = Field(default=500_000, ge=10_000)
    max_workers: int = Field(
        default_factory=lambda: max(1, (psutil.cpu_count(logical=False) or 2) // 1),
        ge=1,
    )
    memory_limit_gb: float = Field(default=16.0, gt=0.0)
    cache_size_gb: float = Field(default=4.0, gt=0.0)
    use_polars: bool = Field(default=True)
    use_integer_contig_map: bool = Field(default=False)
    write_batch_size: int = Field(default=64, ge=1)

    def __str__(self) -> str:
        return f"Config(chunk={self.chunk_size:,}, tile={self.tile_size:,}, workers={self.max_workers})"

    """Configuration for a bigWig -> TileDB conversion run.

    Keeps tuning knobs (chunk sizes, tile sizes, worker count, memory limits)
    for the conversion pipeline.
    """


class ConversionMetrics(BaseModel):
    model_config = {"validate_assignment": True}

    total_values: int = Field(default=0, ge=0)
    total_bytes: int = Field(default=0, ge=0)
    chunks_processed: int = Field(default=0, ge=0)
    chunks_failed: int = Field(default=0, ge=0)
    total_time_seconds: float = Field(default=0.0, ge=0.0)
    compression_ratio: float = Field(default=0.0, ge=0.0)

    @property
    def success_rate(self) -> float:
        total = self.chunks_processed + self.chunks_failed
        return (self.chunks_processed / total * 100) if total > 0 else 0.0

    @property
    def throughput_mbps(self) -> float:
        if self.total_time_seconds <= 0:
            return 0.0
        return (self.total_bytes / 1024**2) / self.total_time_seconds

    @property
    def values_per_second(self) -> float:
        return (
            self.total_values / self.total_time_seconds
            if self.total_time_seconds > 0
            else 0.0
        )

    """Simple metrics collected during a conversion run.

    Fields are updated by the converter implementation; properties provide
    derived values such as throughput and success rate.
    """


class QueryBenchmark(BaseModel):
    model_config = {"validate_assignment": True}

    total_queries: int = Field(..., ge=0)
    iterations: int = Field(..., ge=1)
    query_times: list[float] = Field(..., min_length=1)
    total_values_retrieved: int = Field(default=0, ge=0)
    cache_hit_rate: float = Field(default=0.0, ge=0.0, le=100.0)

    @property
    def avg_time_seconds(self) -> float:
        return float(np.mean(self.query_times))

    @property
    def std_time_seconds(self) -> float:
        return float(np.std(self.query_times))

    @property
    def queries_per_second(self) -> float:
        return (
            self.total_queries / self.avg_time_seconds
            if self.avg_time_seconds > 0
            else 0.0
        )

    @property
    def ms_per_query(self) -> float:
        return (
            (self.avg_time_seconds / self.total_queries) * 1000
            if self.total_queries > 0
            else 0.0
        )

    @property
    def percentiles(self) -> dict[str, float]:
        per_query_times = np.array(self.query_times) / max(self.total_queries, 1) * 1000
        return {
            "p50": float(np.percentile(per_query_times, 50)),
            "p90": float(np.percentile(per_query_times, 90)),
            "p95": float(np.percentile(per_query_times, 95)),
            "p99": float(np.percentile(per_query_times, 99)),
        }

    """Benchmarking results for repeated queries against a data store.

    Stores raw per-iteration timings and derived statistics.
    """


class ContigInfo(BaseModel):
    model_config = {"frozen": True}

    name: str = Field(..., min_length=1)
    length: int = Field(..., gt=0)
    gc_content: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    def __str__(self) -> str:
        gc_str = f", GC={self.gc_content:.1%}" if self.gc_content is not None else ""
        return f"{self.name} ({self.length:,} bp{gc_str})"

    """Small immutable struct with contig/chromosome metadata.

    Contains name, length and optional GC fraction.
    """


@jit(nopython=True, parallel=True, cache=True)
def normalize_genomic_values(values: np.ndarray) -> np.ndarray:
    """Normalize a 1D array of genomic signal values.

    Replaces NaN, +/-Inf and extreme negative sentinel values with 0.0 and
    casts the result to float32 for storage efficiency.

    Parameters
    - values: numpy array (1D) of numeric values

    Returns
    - cleaned: numpy.ndarray[float32]
    """
    cleaned = np.empty(len(values), dtype=np.float32)
    for i in prange(len(values)):
        val = values[i]
        is_valid = not (np.isnan(val) or np.isinf(val) or val < -1000.0)
        cleaned[i] = val if is_valid else 0.0
    return cleaned


@jit(nopython=True, cache=True)
def compute_chunk_statistics(values: np.ndarray) -> tuple[float, float, float, int]:
    """Compute basic statistics over a numeric chunk.

    Ignores NaN and Inf values when computing mean/min/max and returns the
    count of valid numeric values.
    """
    valid_mask = ~(np.isnan(values) | np.isinf(values))
    valid_values = values[valid_mask]
    if len(valid_values) == 0:
        return 0.0, 0.0, 0.0, 0
    mean_val = np.mean(valid_values)
    max_val = np.max(valid_values)
    min_val = np.min(valid_values)
    valid_count = len(valid_values)
    return mean_val, max_val, min_val, valid_count


class BigwigToTileDBConverter:
    """Ultra-high performance converter using BigTools (Rust backend).

    This class expects the optional heavy dependencies (bigtools, tiledb,
    polars) to be available. It implements a multi-threaded, chunked
    conversion pipeline with a progress display.
    """

    def __init__(
        self,
        bigwig_path: Path | str,
        tiledb_path: Path | str,
        config: ConverterConfig | None = None,
    ):
        self.bigwig_path = Path(bigwig_path)
        self.tiledb_path = Path(tiledb_path)
        self.config = config or ConverterConfig()

        if not self.bigwig_path.exists():
            raise FileNotFoundError(f"BigWig file not found: {self.bigwig_path}")

        # Validate BigWig file
        try:
            with pybigtools.open(str(self.bigwig_path)) as bw:
                _ = bw.chroms()
        except Exception as e:
            raise ValueError(f"Invalid BigWig file: {e}")

        logger.info(f"BigTools converter initialized: {self.bigwig_path.name} → {self.tiledb_path.name}")
        logger.info(str(self.config))

    def convert(self) -> ConversionMetrics:
        """Main conversion pipeline with progress tracking"""
        start_time = time.perf_counter()

        # Step 1: Analyze input file
        with console.status("[bold green]Reading chromosome information..."):
            chromosomes = self._read_chromosome_info_fast()
            total_bp = sum(chrom.length for chrom in chromosomes)

        logger.info(f"Found {len(chromosomes)} chromosomes, {total_bp:,} total bp")

        # Step 2: Create optimized schema
        # optionally create a numeric contig map to avoid string dimensions
        if self.config.use_integer_contig_map:
            self._contig_map = {c.name: i for i, c in enumerate(chromosomes)}
        else:
            self._contig_map = None

        with console.status("[bold green]Creating TileDB schema..."):
            self._create_optimized_schema(chromosomes)

        # Step 3: Convert with progress tracking
        stats = self._convert(chromosomes)

        # Finalize stats
        total_time = time.perf_counter() - start_time
        stats.total_time_seconds = total_time

        # Calculate compression ratio
        if stats.total_bytes > 0:
            array_size = self._get_tiledb_size()
            stats.compression_ratio = stats.total_bytes / array_size if array_size > 0 else 1.0

        self._display_conversion_summary(stats, chromosomes)
        return stats

    def _read_chromosome_info_fast(self) -> list[ContigInfo]:
        """Fast chromosome reading with BigTools"""
        chromosomes: list[ContigInfo] = []
        with pybigtools.open(str(self.bigwig_path)) as bw:
            chroms_dict = bw.chroms()
            for name, length in chroms_dict.items():
                if length < 1000:
                    continue
                chromosomes.append(ContigInfo(name=name, length=int(length)))

        def sort_key(chrom: ContigInfo) -> tuple[int, Any]:
            name = chrom.name.lower()
            if name.startswith('chr'):
                name = name[3:]
            if name.isdigit():
                return (0, int(name))
            elif name in ['x', 'y', 'm', 'mt']:
                return (1, ord(name[0]))
            else:
                return (2, name)

        return sorted(chromosomes, key=sort_key)

    def _create_optimized_schema(self, chromosomes: list[ContigInfo]) -> None:
        """Create TileDB schema with cutting-edge optimizations"""
        max_chrom_len = max(len(chrom.name) for chrom in chromosomes)
        max_position = max(chrom.length for chrom in chromosomes)

        ctx = tiledb.Ctx({
            'sm.tile_cache_size': str(int(self.config.cache_size_gb * 1024**3)),
            'sm.consolidation.buffer_size': str(512*1024**2),
            'sm.query.dense.reader': 'refactored',
            'sm.mem.malloc_trim': 'true',
            'vfs.file.posix_file_permissions': '644',
            'sm.query.dense.qc_coords_mode': 'true',
            'sm.consolidation.mode': 'commits',
            'sm.io_concurrency_level': str(min(self.config.max_workers, 8)),
            'sm.compute_concurrency_level': str(self.config.max_workers),
        })

        if self.config.use_integer_contig_map:
            # use numeric contig IDs (sparse) to avoid string dtype overhead
            chrom_dim = tiledb.Dim(name="chromosome", domain=(0, len(chromosomes) - 1), dtype=np.uint32, tile=1)
        else:
            chrom_dim = tiledb.Dim(name="chromosome", domain=(None, None), dtype="ascii", tile=max_chrom_len + 10)

        optimal_tile_size = min(self.config.tile_size, max(1000, max_position // 10000))

        pos_dim = tiledb.Dim(name="position", domain=(0, max_position + 1000), dtype=np.uint64, tile=optimal_tile_size)

        domain = tiledb.Domain(chrom_dim, pos_dim)

        filters = tiledb.FilterList([
            tiledb.BitShuffleFilter(),
            tiledb.ByteShuffleFilter(),
            tiledb.ZstdFilter(level=self.config.compression_level),
            tiledb.ChecksumMD5Filter(),
        ])

        value_attr = tiledb.Attr(name="value", dtype=np.float32, filters=filters)

        # When using a string 'chromosome' dimension the array must be sparse
        # because dense arrays require all dims to share the same primitive dtype.
        schema = tiledb.ArraySchema(
            domain=domain,
            attrs=[value_attr],
            sparse=True,
            capacity=self.config.buffer_size,
            ctx=ctx,
        )

        tiledb.Array.create(str(self.tiledb_path), schema, ctx=ctx)
        logger.info(f"Created sparse TileDB array with tile size {optimal_tile_size:,}")

    def _convert(self, chromosomes: list[ContigInfo]) -> ConversionMetrics:
        """Convert with beautiful progress display and parallel processing"""
        chunks = self._generate_smart_chunks(chromosomes)
        stats = ConversionMetrics()

        # Partition chunks into per-worker batches to reduce executor overhead
        num_workers = max(1, min(self.config.max_workers, len(chunks)))

        chunk_batches = self._partition_chunks(chunks, num_workers)

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(bar_width=40), TaskProgressColumn(), TimeRemainingColumn(), TextColumn("[bold blue]{task.fields[throughput]}"), console=console, refresh_per_second=4) as progress:
            task = progress.add_task("Converting genomic data", total=len(chunks), throughput="0 MB/s")

            start_time = time.perf_counter()
            total_bytes = 0

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(self._worker_process_chunks, batch, progress, task) for batch in chunk_batches]

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        # result is a dict of aggregated stats from that worker
                        stats.chunks_processed += result.get('chunks_processed', 0)
                        stats.chunks_failed += result.get('chunks_failed', 0)
                        stats.total_values += result.get('total_values', 0)
                        bytes_written = result.get('total_bytes', 0)
                        stats.total_bytes += bytes_written
                        total_bytes += bytes_written
                    except Exception as e:
                        logger.error(f"Worker failed: {e}")

                    elapsed_time = time.perf_counter() - start_time
                    if elapsed_time > 0:
                        throughput_mbps = (total_bytes / 1024**2) / elapsed_time
                        progress.update(task, advance=0, throughput=f"{throughput_mbps:.1f} MB/s")

        return stats

    def _partition_chunks(self, chunks: list[Tuple[str, int, int]], n: int) -> list[list[Tuple[str, int, int]]]:
        """Split chunks into n roughly-equal batches."""
        batches: list[list[Tuple[str, int, int]]] = [[] for _ in range(n)]
        for i, c in enumerate(chunks):
            batches[i % n].append(c)
        return [b for b in batches if b]

    def _worker_process_chunks(self, chunks: list[Tuple[str, int, int]], progress: Progress, task_id: int) -> Dict[str, int]:
        """Worker that processes a list of chunks, writes them in-batch, and returns aggregated stats.

        Each worker opens its own bigWig handle and TileDB array writer so there is minimal cross-thread contention.
        """
        local_stats = {'chunks_processed': 0, 'chunks_failed': 0, 'total_values': 0, 'total_bytes': 0}

        # per-worker bigWig handle
        bw_handle = None
        try:
            if pybigtools is not None:
                bw_handle = pybigtools.open(str(self.bigwig_path))
                _THREAD_LOCAL.bw_handle = bw_handle

            # Open a TileDB array per worker for batched writes
            tdb_array = None
            if tiledb is not None:
                tdb_array = tiledb.open(str(self.tiledb_path), 'w')

            bytes_written = 0

            # buffered writes to reduce per-chunk write overhead
            buf_chroms: list = []
            buf_positions: list = []
            buf_values: list = []
            buf_count = 0

            for idx, chunk in enumerate(chunks):
                chunk_result = self._process_chunk_bigtools(chunk)
                if chunk_result is None:
                    local_stats['chunks_failed'] += 1
                    continue

                buf_chroms.append(chunk_result['chromosomes'])
                buf_positions.append(chunk_result['positions'])
                buf_values.append(chunk_result['values'])
                buf_count += 1

                try:
                    if buf_count >= self.config.write_batch_size and tdb_array is not None:
                        # concatenate and write in a single operation
                        all_chrom = np.concatenate(buf_chroms)
                        all_pos = np.concatenate(buf_positions)
                        all_vals = np.concatenate(buf_values)
                        tdb_array[all_chrom, all_pos] = all_vals
                        cb = all_vals.size * 4
                        local_stats['chunks_processed'] += buf_count
                        local_stats['total_values'] += all_vals.size
                        local_stats['total_bytes'] += cb
                        bytes_written += cb
                        # reset buffers
                        buf_chroms = []
                        buf_positions = []
                        buf_values = []
                        buf_count = 0
                except Exception as e:
                    logger.error(f"Worker batched write failed: {e}")
                    local_stats['chunks_failed'] += buf_count
                    buf_chroms = []
                    buf_positions = []
                    buf_values = []
                    buf_count = 0

                # update progress every 16 chunks to reduce contention
                if (idx & 0x0F) == 0:
                    progress.update(task_id, advance=16 if (idx + 16) <= len(chunks) else 1)

            # flush remaining buffers
            if buf_count > 0 and tdb_array is not None:
                try:
                    all_chrom = np.concatenate(buf_chroms)
                    all_pos = np.concatenate(buf_positions)
                    all_vals = np.concatenate(buf_values)
                    tdb_array[all_chrom, all_pos] = all_vals
                    cb = all_vals.size * 4
                    local_stats['chunks_processed'] += buf_count
                    local_stats['total_values'] += all_vals.size
                    local_stats['total_bytes'] += cb
                    bytes_written += cb
                except Exception as e:
                    logger.error(f"Worker flush write failed: {e}")
                    local_stats['chunks_failed'] += buf_count

            # final progress update for any remainder
            progress.update(task_id, advance=0)

        finally:
            try:
                if bw_handle is not None:
                    bw_handle.close()
                    _THREAD_LOCAL.bw_handle = None
            except Exception:
                pass
            try:
                if tdb_array is not None:
                    tdb_array.close()
            except Exception:
                pass

        return local_stats

    def _generate_smart_chunks(self, chromosomes: list[ContigInfo]) -> list[Tuple[str, int, int]]:
        chunks: list[Tuple[str, int, int]] = []
        for chrom in chromosomes:
            if chrom.length < 50_000_000:
                chunk_size = min(self.config.chunk_size, chrom.length // 4)
            else:
                chunk_size = self.config.chunk_size
            chunk_size = max(chunk_size, 100_000)
            for start in range(0, chrom.length, chunk_size):
                end = min(start + chunk_size, chrom.length)
                chunks.append((chrom.name, start, end))
        logger.info(f"Generated {len(chunks)} processing chunks")
        return chunks

    def _process_chunk_bigtools(self, chunk_info: Tuple[str, int, int]) -> Optional[Dict[str, Any]]:
        chrom, start, end = chunk_info

        # Try to reuse a per-thread bigWig handle to avoid repeated open()/close().
        bw_handle = getattr(_THREAD_LOCAL, 'bw_handle', None)

        try:
            if bw_handle is None:
                # If pybigtools is available create a persistent handle per-thread.
                if pybigtools is not None:
                    bw_handle = pybigtools.open(str(self.bigwig_path))
                    _THREAD_LOCAL.bw_handle = bw_handle
                    values = bw_handle.values(chrom, start, end)
                else:
                    # Fallback: open, read, and close immediately
                    with pybigtools.open(str(self.bigwig_path)) as bw:
                        values = bw.values(chrom, start, end)
            else:
                values = bw_handle.values(chrom, start, end)
        except Exception:
            return None

        if values is None or len(values) == 0:
            return None

        # Normalize in-place and cast to float32 to reduce memory
        cleaned_values = normalize_genomic_values(values.astype(np.float64)).astype(np.float32)

        mean_val, max_val, min_val, valid_count = compute_chunk_statistics(cleaned_values)
        if valid_count == 0:
            return None

        # Fast numpy-first filtering (avoid constructing Polars DataFrames in the hot path)
        nonzero_mask = cleaned_values != 0.0
        if not nonzero_mask.any():
            return None

        positions = np.nonzero(nonzero_mask)[0].astype(np.uint64) + np.uint64(start)
        values_out = cleaned_values[nonzero_mask]

        # Create chromosome array matching positions length. Use unicode dtype sized to name length.
        chrom_arr = np.full(len(positions), chrom, dtype=f'U{len(chrom)}')

        return {
            'chromosomes': chrom_arr,
            'positions': positions,
            'values': values_out,
            'value_count': len(values_out),
            'stats': {'mean': mean_val, 'max': max_val, 'min': min_val},
        }

    def _write_chunk_to_tiledb(self, array: Any, chunk_data: Dict[str, Any]) -> None:
        try:
            array[chunk_data['chromosomes'], chunk_data['positions']] = chunk_data['values']
        except Exception as e:
            logger.error(f"TileDB write failed: {e}")
            raise

    def _get_tiledb_size(self) -> int:
        try:
            total_size = 0
            for path in self.tiledb_path.rglob('*'):
                if path.is_file():
                    total_size += path.stat().st_size
            return total_size
        except Exception:
            return 0

    def _display_conversion_summary(self, stats: ConversionMetrics, chromosomes: list[ContigInfo]) -> None:
        table = Table(title="🧬 Genomic Conversion Results", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="green")
        table.add_column("Unit", style="yellow")
        table.add_row("Total Values", f"{stats.total_values:,}", "values")
        table.add_row("Data Size", f"{stats.total_bytes/1024**3:.2f}", "GB")
        table.add_row("Processing Time", f"{stats.total_time_seconds:.1f}", "seconds")
        table.add_row("Throughput", f"{stats.throughput_mbps:.1f}", "MB/s")
        table.add_row("Values/Second", f"{stats.values_per_second:,.0f}", "values/s")
        table.add_row("Success Rate", f"{stats.success_rate:.1f}", "%")
        if stats.compression_ratio > 0:
            table.add_row("Compression", f"{stats.compression_ratio:.1f}x", "ratio")
        console.print(table)


class TileDBQueryEngine:
    """Simple query API for TileDB-backed genomic signal.

    Provides a small, testable surface: query_region and bench_queries.
    If `tiledb` is not installed the methods raise a clear error.
    """

    def __init__(self, tiledb_path: Path | str):
        self.tiledb_path = Path(tiledb_path)
        if tiledb is None:
            raise RuntimeError("TileDB is not installed; TileDBQueryEngine unavailable")

    def query_region(self, chrom: str | int, start: int, end: int) -> np.ndarray:
        """Return values for a region as a numpy array of floats.

        If contigs are stored as integer IDs, the caller should pass the
        numeric chrom ID. This method always returns positions and values
        aligned to [start, end).
        """
        if not self.tiledb_path.exists():
            raise FileNotFoundError(f"TileDB array not found: {self.tiledb_path}")

        with tiledb.open(str(self.tiledb_path), 'r') as arr:
            # Support both sparse and dense arrays; attempt a simple coordinate read
            try:
                coords = (chrom, np.arange(start, end, dtype=np.uint64))
                vals = arr[coords]
            except Exception:
                # Fallback: read ranged query using queries API
                q = arr.query()
                q.set_ranges('position', [(start, end - 1)])
                q.set_ranges('chromosome', [(chrom, chrom)])
                df = q.df[:]
                if df is None or len(df) == 0:
                    return np.array([], dtype=np.float32)
                return df['value'].to_numpy(dtype=np.float32)

            if vals is None:
                return np.array([], dtype=np.float32)

            # TileDB may return an OrderedDict or mapping of attribute -> values
            if isinstance(vals, dict):
                # Prefer attribute named 'value', otherwise take first attribute
                if 'value' in vals:
                    vals = vals['value']
                else:
                    # OrderedDict -> take first value array
                    first = next(iter(vals.values()), None)
                    vals = first if first is not None else np.array([], dtype=np.float32)

            # If vals is a sequence of (pos, val) tuples, try to extract values
            if isinstance(vals, (list, tuple)) and len(vals) > 0 and isinstance(vals[0], (list, tuple)):
                try:
                    vals = np.array([v for (_, v) in vals], dtype=np.float32)
                except Exception:
                    vals = np.asarray(vals, dtype=np.float32)

            return np.asarray(vals, dtype=np.float32)

    def bench_queries(self, queries: list[GenomicRegion], iterations: int = 3) -> QueryBenchmark:
        """Run repeated queries and return a QueryBenchmark summary."""
        times: list[float] = []
        total_values = 0
        for it in range(iterations):
            t0 = time.perf_counter()
            for q in queries:
                vals = self.query_region(q.chrom, q.start, q.end)
                total_values += len(vals)
            times.append(time.perf_counter() - t0)

        return QueryBenchmark(total_queries=len(queries), iterations=iterations, query_times=times, total_values_retrieved=total_values)
