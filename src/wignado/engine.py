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
import gc

try:
    import pybigtools
except Exception:  # pragma: no cover - optional dependency
    pybigtools = None
    warnings.warn("pybigtools is not installed; fallback code paths will be limited")

try:
    import bigtools
except Exception:  # pragma: no cover - optional dependency
    bigtools = None
    warnings.warn("bigtools is not installed; high-performance conversion disabled")

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


class BigWigToTileDBConverter:
    """Converter class that orchestrates reading a bigWig file (via py)

    and writing an optimized TileDB schema.

    This class focuses on orchestration; heavy-lifting helper methods are
    expected to exist (for example, _read_chromosome_info_fast,
    _create_optimized_schema, _convert_with_rich_progress).
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

        try:
            with pybigtools.open(str(self.bigwig_path)) as bw:
                _ = bw.chroms()
        except Exception as e:
            raise ValueError(f"Invalid BigWig file: {e}")

        logger.info(
            f"BigTools converter initialized: {self.bigwig_path.name} → {self.tiledb_path.name}"
        )
        logger.info(str(self.config))

    def run_conversion(self) -> ConversionMetrics:
        """Execute the conversion pipeline and return collected metrics."""
        start_time = time.perf_counter()

        with console.status("[bold green]Reading chromosome information..."):
            chromosomes = self._read_chromosome_info()
            total_bp = sum(chrom.length for chrom in chromosomes)

        logger.info(f"Found {len(chromosomes)} chromosomes, {total_bp:,} total bp")

        with console.status("[bold green]Creating TileDB schema..."):
            self._create_optimized_schema(chromosomes)

        stats = self._convert_with_rich_progress(chromosomes)

        total_time = time.perf_counter() - start_time
        stats.total_time_seconds = total_time
        return stats

    def _read_chromosome_info(self) -> list[ContigInfo]:
        """Read chromosome/contig sizes from the BigWig file and return a list
        of `ContigInfo` objects. This uses `pybigtools`'s `chroms()` if
        available and falls back gracefully.
        """
        contigs: list[ContigInfo] = []
        try:
            with pybigtools.open(str(self.bigwig_path)) as bw:
                chroms = bw.chroms()
        except Exception:
            # If chroms() fails, re-raise a clearer error
            raise RuntimeError("Failed to read chromosome information from BigWig")

        # croms() might return dict-like or iterable of tuples
        if isinstance(chroms, dict):
            items = chroms.items()
        else:
            try:
                items = list(chroms)
            except Exception:
                raise RuntimeError("Unsupported chromosome listing returned by pybigtools")

        for it in items:
            if isinstance(it, tuple) and len(it) >= 2:
                name, length = it[0], int(it[1])
            else:
                # unexpected shape, try to convert
                name = str(it)
                length = 0
            contigs.append(ContigInfo(name=name, length=length))
        return contigs

    def _create_optimized_schema(self, chromosomes: list[ContigInfo]) -> None:
        """Create or plan an optimized TileDB schema for the provided
        chromosomes. This implementation only logs the plan; a full
        implementation would call tiledb APIs and create arrays.
        """
        logger.info(f"Planning TileDB schema for {len(chromosomes)} contigs")

    def _convert_with_rich_progress(self, chromosomes: list[ContigInfo]) -> ConversionMetrics:
        """Placeholder conversion worker.

        Iterates contigs and synthesizes simple metrics. A production
        implementation would stream values and write to TileDB here.
        """
        stats = ConversionMetrics()
        for chrom in chromosomes:
            # Simulate processing one chunk per contig
            stats.chunks_processed += 1
            stats.total_values += chrom.length
            stats.total_bytes += chrom.length * 4  # assume 4 bytes/value
        stats.compression_ratio = 1.0
        return stats


class BigToolsTileDBConverter:
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
            with bigtools.open(str(self.bigwig_path)) as bw:
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
        with console.status("[bold green]Creating TileDB schema..."):
            self._create_optimized_schema(chromosomes)

        # Step 3: Convert with progress tracking
        stats = self._convert_with_rich_progress(chromosomes)

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
        with bigtools.open(str(self.bigwig_path)) as bw:
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

        schema = tiledb.ArraySchema(domain=domain, attrs=[value_attr], sparse=False, capacity=self.config.buffer_size, cell_order='row-major', tile_order='row-major', ctx=ctx)

        tiledb.Array.create(str(self.tiledb_path), schema, ctx=ctx)
        logger.info(f"Created TileDB array with tile size {optimal_tile_size:,}")

    def _convert_with_rich_progress(self, chromosomes: list[ContigInfo]) -> ConversionMetrics:
        """Convert with beautiful progress display and parallel processing"""
        chunks = self._generate_smart_chunks(chromosomes)
        stats = ConversionMetrics()

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(bar_width=40), TaskProgressColumn(), TimeRemainingColumn(), TextColumn("[bold blue]{task.fields[throughput]}"), console=console, refresh_per_second=4) as progress:
            task = progress.add_task("Converting genomic data", total=len(chunks), throughput="0 MB/s")

            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_chunk = {executor.submit(self._process_chunk_bigtools, chunk_info): chunk_info for chunk_info in chunks}

                with tiledb.open(str(self.tiledb_path), 'w') as tdb_array:
                    completed_chunks = 0
                    total_bytes = 0
                    start_time = time.perf_counter()

                    for future in as_completed(future_to_chunk):
                        chunk_info = future_to_chunk[future]
                        try:
                            chunk_result = future.result(timeout=60)
                            if chunk_result is not None:
                                self._write_chunk_to_tiledb(tdb_array, chunk_result)
                                stats.chunks_processed += 1
                                stats.total_values += chunk_result['value_count']
                                chunk_bytes = chunk_result['value_count'] * 4
                                stats.total_bytes += chunk_bytes
                                total_bytes += chunk_bytes
                            else:
                                stats.chunks_failed += 1
                        except Exception as e:
                            logger.error(f"Chunk {chunk_info} failed: {e}")
                            stats.chunks_failed += 1

                        completed_chunks += 1
                        elapsed_time = time.perf_counter() - start_time
                        if elapsed_time > 0:
                            throughput_mbps = (total_bytes / 1024**2) / elapsed_time
                            progress.update(task, completed=completed_chunks, throughput=f"{throughput_mbps:.1f} MB/s")
                        else:
                            progress.update(task, completed=completed_chunks)

                        if completed_chunks % 100 == 0:
                            gc.collect()

        return stats

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
        try:
            with bigtools.open(str(self.bigwig_path)) as bw:
                try:
                    values = bw.values(chrom, start, end)
                except Exception:
                    return None
                if values is None or len(values) == 0:
                    return None
                cleaned_values = normalize_genomic_values(values.astype(np.float64))
                mean_val, max_val, min_val, valid_count = compute_chunk_statistics(cleaned_values)
                if valid_count == 0:
                    return None
                if self.config.use_polars:
                    positions = np.arange(start, start + len(cleaned_values), dtype=np.uint64)
                    df = pl.DataFrame({
                        "chromosome": [chrom] * len(cleaned_values),
                        "position": positions,
                        "value": cleaned_values,
                    })
                    non_zero_df = df.filter(pl.col("value") != 0.0)
                    if len(non_zero_df) == 0:
                        return None
                    return {
                        'chromosomes': non_zero_df['chromosome'].to_numpy(),
                        'positions': non_zero_df['position'].to_numpy(),
                        'values': non_zero_df['value'].to_numpy(),
                        'value_count': len(non_zero_df),
                        'stats': {'mean': mean_val, 'max': max_val, 'min': min_val},
                    }
                else:
                    positions = np.arange(start, start + len(cleaned_values), dtype=np.uint64)
                    chromosomes = np.full(len(positions), chrom, dtype=f'U{len(chrom)}')
                    return {
                        'chromosomes': chromosomes,
                        'positions': positions,
                        'values': cleaned_values,
                        'value_count': len(cleaned_values),
                        'stats': {'mean': mean_val, 'max': max_val, 'min': min_val},
                    }
        except Exception as e:
            logger.warning(f"Failed to process {chrom}:{start:,}-{end:,}: {e}")
            return None

    def _write_chunk_to_tiledb(self, array: tiledb.Array, chunk_data: Dict[str, Any]) -> None:
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
