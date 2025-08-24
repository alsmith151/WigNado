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

# Optional dependencies with graceful fallbacks
try:
    import pybigtools
except ImportError:
    pybigtools = None
    warnings.warn("pybigtools not installed; fallback paths limited", ImportWarning)

try:
    import tiledb
except ImportError:
    tiledb = None
    warnings.warn("tiledb not installed; TileDB output disabled", ImportWarning)

try:
    import polars as pl
except ImportError:
    pl = None
    warnings.warn("polars not installed; slower numpy paths used", ImportWarning)

import numpy as np
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from loguru import logger
from numba import jit, prange
from pydantic import BaseModel, Field, field_validator
from rich.console import Console
from rich.progress import (
    BarColumn, Progress, SpinnerColumn, TaskProgressColumn,
    TextColumn, TimeRemainingColumn
)
from rich.table import Table

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Global console and thread-local storage
console = Console()
_THREAD_LOCAL = threading.local()

# Configure logger
logger.remove()
logger.add(
    sink=lambda msg: console.print(f"{msg}", markup=False),
    format="{time:HH:mm:ss} | {level: <8} | {message}",
    level="INFO",
)


@runtime_checkable
class Closeable(Protocol):
    """Protocol for objects that can be closed."""
    def close(self) -> None: ...


class GenomicRegion(BaseModel):
    """A validated genomic region with chromosome, start, and end coordinates.
    
    Attributes:
        chrom: chromosome/contig name
        start: 0-based inclusive start position  
        end: 0-based exclusive end position (must be > start)
    """
    model_config = {"validate_assignment": True, "extra": "forbid"}

    chrom: str = Field(..., min_length=1)
    start: int = Field(..., ge=0)
    end: int = Field(..., gt=0)

    @field_validator("end")
    @classmethod
    def validate_interval(cls, v: int, info) -> int:
        if "start" in info.data and v <= info.data["start"]:
            raise ValueError(f"End position ({v}) must be greater than start ({info.data['start']})")
        return v

    @property
    def length(self) -> int:
        """Return the length of this genomic region."""
        return self.end - self.start

    def __str__(self) -> str:
        return f"{self.chrom}:{self.start:,}-{self.end:,}"


class ConverterConfig(BaseModel):
    """Configuration for bigWig to TileDB conversion.
    
    Streamlined configuration with only essential tuning parameters.
    """
    model_config = {"validate_assignment": True, "extra": "forbid", "arbitrary_types_allowed": True}

    # Core processing parameters
    overwrite: bool = Field(default=False, description="Overwrite existing output")
    chunk_size: int = Field(default=2_000_000, ge=50_000, description="Processing chunk size in bp")
    max_workers: int = Field(
        default_factory=lambda: max(1, (psutil.cpu_count(logical=False) or 2)),
        ge=1,
        description="Number of worker threads"
    )
    
    # Storage and compression
    tile_size: int = Field(default=10_000, ge=1_000, description="TileDB tile size")
    compression_level: int = Field(default=3, ge=1, le=9, description="Compression level (1-9)")
    value_dtype: str = Field(default="float32", pattern="^(float32|float16)$", 
                            description="Storage dtype: float32 or float16")
    
    # Memory management (combined into single parameter)
    memory_gb: float = Field(default=8.0, gt=0.0, le=64.0, description="Total memory limit in GB")
    
    # Batch processing
    write_batch_size: int = Field(default=64, ge=8, le=512, description="Write batch size")
    
    # Advanced options
    use_integer_contig_map: bool = Field(default=False, description="Use integer chromosome IDs")

    @property
    def cache_size_gb(self) -> float:
        """Derive cache size from total memory (25% allocation)."""
        return self.memory_gb * 0.25
    
    @property
    def buffer_size(self) -> int:
        """Derive buffer size from memory and worker count."""
        base_buffer = int(self.memory_gb * 1024 * 1024 * 0.1)  # 10% of memory in MB
        return max(100_000, base_buffer // self.max_workers)
    
    @property
    def numpy_dtype(self) -> np.dtype:
        """Convert string dtype to numpy dtype."""
        return np.float16 if self.value_dtype == "float16" else np.float32

    def __str__(self) -> str:
        return f"Config(chunk={self.chunk_size:,}, tile={self.tile_size:,}, workers={self.max_workers}, mem={self.memory_gb}GB)"


class ConversionMetrics(BaseModel):
    """Metrics collected during a bigWig to TileDB conversion.
    
    Tracks values processed, bytes written, timing, and success rates.
    """
    model_config = {"validate_assignment": True}

    total_values: int = Field(default=0, ge=0)
    total_bytes: int = Field(default=0, ge=0)
    chunks_processed: int = Field(default=0, ge=0)
    chunks_failed: int = Field(default=0, ge=0)
    total_time_seconds: float = Field(default=0.0, ge=0.0)
    compression_ratio: float = Field(default=0.0, ge=0.0)

    @property
    def success_rate(self) -> float:
        """Calculate the success rate as a percentage."""
        total = self.chunks_processed + self.chunks_failed
        return (self.chunks_processed / total * 100) if total > 0 else 0.0

    @property
    def throughput_mbps(self) -> float:
        """Calculate throughput in MB/s."""
        if self.total_time_seconds <= 0:
            return 0.0
        return (self.total_bytes / 1024**2) / self.total_time_seconds

    @property
    def values_per_second(self) -> float:
        """Calculate values processed per second."""
        return self.total_values / self.total_time_seconds if self.total_time_seconds > 0 else 0.0


class QueryBenchmark(BaseModel):
    """Benchmarking results for repeated queries against a data store.
    
    Stores raw per-iteration timings and provides derived statistics.
    """
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
        return self.total_queries / self.avg_time_seconds if self.avg_time_seconds > 0 else 0.0

    @property
    def ms_per_query(self) -> float:
        return (self.avg_time_seconds / self.total_queries) * 1000 if self.total_queries > 0 else 0.0

    @property
    def percentiles(self) -> dict[str, float]:
        """Calculate query time percentiles in milliseconds."""
        per_query_times = np.array(self.query_times) / max(self.total_queries, 1) * 1000
        return {
            "p50": float(np.percentile(per_query_times, 50)),
            "p90": float(np.percentile(per_query_times, 90)),
            "p95": float(np.percentile(per_query_times, 95)),
            "p99": float(np.percentile(per_query_times, 99)),
        }


class ContigInfo(BaseModel):
    """Immutable contig/chromosome metadata.
    
    Contains name, length and optional GC content fraction.
    """
    model_config = {"frozen": True}

    name: str = Field(..., min_length=1)
    length: int = Field(..., gt=0)
    gc_content: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    def __str__(self) -> str:
        gc_str = f", GC={self.gc_content:.1%}" if self.gc_content is not None else ""
        return f"{self.name} ({self.length:,} bp{gc_str})"


@jit(nopython=True, parallel=True, cache=True)
def normalize_genomic_values(values: np.ndarray) -> np.ndarray:
    """Normalize genomic signal values by replacing invalid values with 0.0.

    Replaces NaN, +/-Inf and extreme negative sentinel values with 0.0 and
    casts to float32 for storage efficiency.

    Args:
        values: 1D numpy array of numeric values

    Returns:
        Cleaned numpy array of float32 values
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

    Args:
        values: Input array of values

    Returns:
        Tuple of (mean, max, min, valid_count)
    """
    valid_mask = ~(np.isnan(values) | np.isinf(values))
    valid_values = values[valid_mask]
    
    if len(valid_values) == 0:
        return 0.0, 0.0, 0.0, 0
    
    return (
        np.mean(valid_values),
        np.max(valid_values),
        np.min(valid_values),
        len(valid_values)
    )


class BigwigToTileDBConverter:
    """Ultra-high performance converter using BigTools (Rust backend).

    Implements a multi-threaded, chunked conversion pipeline with progress display.
    Requires pybigtools and tiledb to be available.
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
        self._contig_map: Optional[Dict[str, int]] = None
        self._value_np_dtype = self.config.numpy_dtype

        self._validate_dependencies()
        self._validate_input_file()
        self._prepare_output_directory()

        logger.info(f"BigTools converter initialized: {self.bigwig_path.name} → {self.tiledb_path.name}")
        logger.info(str(self.config))

    def _validate_dependencies(self) -> None:
        """Ensure required dependencies are available."""
        if pybigtools is None:
            raise RuntimeError("pybigtools is required but not installed")
        if tiledb is None:
            raise RuntimeError("tiledb is required but not installed")

    def _validate_input_file(self) -> None:
        """Validate that the input bigWig file exists and is readable."""
        if not self.bigwig_path.exists():
            raise FileNotFoundError(f"BigWig file not found: {self.bigwig_path}")

        try:
            with pybigtools.open(str(self.bigwig_path)) as bw:
                _ = bw.chroms()
        except Exception as e:
            raise ValueError(f"Invalid BigWig file: {e}") from e

    def _prepare_output_directory(self) -> None:
        """Prepare the output directory, handling overwrite if needed."""
        if self.config.overwrite and self.tiledb_path.exists():
            import shutil
            shutil.rmtree(self.tiledb_path)
            logger.info(f"Overwriting existing TileDB directory: {self.tiledb_path}")

    def convert(self) -> ConversionMetrics:
        """Execute the main conversion pipeline with progress tracking."""
        start_time = time.perf_counter()

        # Step 1: Analyze input file
        chromosomes = self._read_chromosome_info()
        total_bp = sum(chrom.length for chrom in chromosomes)
        logger.info(f"Found {len(chromosomes)} chromosomes, {total_bp:,} total bp")

        # Step 2: Create optimized schema
        self._setup_contig_mapping(chromosomes)
        self._create_optimized_schema(chromosomes)

        # Step 3: Convert with progress tracking
        stats = self._perform_conversion(chromosomes)

        # Step 4: Finalize metrics
        self._finalize_metrics(stats, start_time, chromosomes)
        return stats

    def _read_chromosome_info(self) -> list[ContigInfo]:
        """Read chromosome information from the bigWig file."""
        with console.status("[bold green]Reading chromosome information..."):
            chromosomes: list[ContigInfo] = []
            
            with pybigtools.open(str(self.bigwig_path)) as bw:
                chroms_dict = bw.chroms()
                for name, length in chroms_dict.items():
                    if length >= 1000:  # Filter out very small contigs
                        chromosomes.append(ContigInfo(name=name, length=int(length)))

            return self._sort_chromosomes(chromosomes)

    def _sort_chromosomes(self, chromosomes: list[ContigInfo]) -> list[ContigInfo]:
        """Sort chromosomes in a sensible order (numeric, then X/Y/M, then others)."""
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

    def _setup_contig_mapping(self, chromosomes: list[ContigInfo]) -> None:
        """Set up contig mapping if using integer contig IDs."""
        if self.config.use_integer_contig_map:
            self._contig_map = {c.name: i for i, c in enumerate(chromosomes)}
        else:
            self._contig_map = None

    def _create_optimized_schema(self, chromosomes: list[ContigInfo]) -> None:
        """Create TileDB schema with optimizations."""
        with console.status("[bold green]Creating TileDB schema..."):
            max_chrom_len = max(len(chrom.name) for chrom in chromosomes)
            max_position = max(chrom.length for chrom in chromosomes)

            ctx = self._create_tiledb_context()
            domain = self._create_domain(chromosomes, max_chrom_len, max_position)
            
            # Create schema with optimized settings
            filters = self._create_compression_filters()
            value_attr = tiledb.Attr(name="value", dtype=self._value_np_dtype, filters=filters)
            
            schema = tiledb.ArraySchema(
                domain=domain,
                attrs=[value_attr],
                sparse=True,  # Always use sparse for flexibility
                capacity=self.config.buffer_size,
                ctx=ctx,
            )

            tiledb.Array.create(str(self.tiledb_path), schema, ctx=ctx)
            logger.info(f"Created sparse TileDB array with tile size {self.config.tile_size:,}")

            # Persist contig mapping metadata (for integer mapping queries later)
            if self._contig_map is not None:
                try:
                    import json
                    with tiledb.open(str(self.tiledb_path), 'w', ctx=ctx) as arr:
                        arr.meta["contig_map_json"] = json.dumps(self._contig_map)
                except Exception as e:
                    logger.warning(f"Failed to store contig map metadata: {e}")

    def _create_tiledb_context(self) -> Any:
        """Create optimized TileDB context."""
        return tiledb.Ctx({
            'sm.tile_cache_size': str(int(self.config.cache_size_gb * 1024**3)),
            'sm.consolidation.buffer_size': str(int(self.config.memory_gb * 128 * 1024**2)),  # 128MB per GB
            'sm.query.dense.reader': 'refactored',
            'sm.mem.malloc_trim': 'true',
            'vfs.file.posix_file_permissions': '644',
            'sm.query.dense.qc_coords_mode': 'true',
            'sm.consolidation.mode': 'commits',
            'sm.io_concurrency_level': str(min(self.config.max_workers, 8)),
            'sm.compute_concurrency_level': str(self.config.max_workers),
        })

    def _create_domain(self, chromosomes: list[ContigInfo], max_chrom_len: int, max_position: int) -> Any:
        """Create TileDB domain with appropriate dimensions."""
        if self.config.use_integer_contig_map:
            chrom_dim = tiledb.Dim(
                name="chromosome", 
                domain=(0, len(chromosomes) - 1), 
                dtype=np.uint32, 
                tile=1
            )
        else:
            chrom_dim = tiledb.Dim(
                name="chromosome", 
                domain=(None, None), 
                dtype="ascii", 
                tile=max_chrom_len + 10
            )

        optimal_tile_size = min(self.config.tile_size, max(1000, max_position // 10000))
        pos_dim = tiledb.Dim(
            name="position", 
            domain=(0, max_position + 1000), 
            dtype=np.uint64, 
            tile=optimal_tile_size
        )

        return tiledb.Domain(chrom_dim, pos_dim)

    def _create_compression_filters(self) -> Any:
        """Create compression filter list."""
        return tiledb.FilterList([
            tiledb.BitShuffleFilter(),
            tiledb.ByteShuffleFilter(),
            tiledb.ZstdFilter(level=self.config.compression_level),
            tiledb.ChecksumMD5Filter(),
        ])

    def _perform_conversion(self, chromosomes: list[ContigInfo]) -> ConversionMetrics:
        """Execute the conversion with parallel processing and progress tracking."""
        chunks = self._generate_chunks(chromosomes)
        stats = ConversionMetrics()
        
        num_workers = max(1, min(self.config.max_workers, len(chunks)))
        chunk_batches = self._partition_chunks(chunks, num_workers)

        with Progress(
            SpinnerColumn(), 
            TextColumn("[progress.description]{task.description}"), 
            BarColumn(bar_width=40), 
            TaskProgressColumn(), 
            TimeRemainingColumn(), 
            TextColumn("[bold blue]{task.fields[throughput]}"), 
            console=console, 
            refresh_per_second=4
        ) as progress:
            
            task = progress.add_task("Converting genomic data", total=len(chunks), throughput="0 MB/s")
            start_time = time.perf_counter()
            total_bytes = 0

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(self._worker_process_chunks, batch, progress, task) 
                    for batch in chunk_batches
                ]

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        self._update_stats(stats, result)
                        total_bytes += result.get('total_bytes', 0)
                        
                        # Update throughput display
                        elapsed_time = time.perf_counter() - start_time
                        if elapsed_time > 0:
                            throughput_mbps = (total_bytes / 1024**2) / elapsed_time
                            progress.update(task, advance=0, throughput=f"{throughput_mbps:.1f} MB/s")
                            
                    except Exception as e:
                        logger.error(f"Worker failed: {e}")

        return stats

    def _generate_chunks(self, chromosomes: list[ContigInfo]) -> list[Tuple[str, int, int]]:
        """Generate processing chunks with adaptive sizing."""
        chunks: list[Tuple[str, int, int]] = []
        
        for chrom in chromosomes:
            # Adaptive chunk sizing based on chromosome length
            if chrom.length < 50_000_000:
                chunk_size = min(self.config.chunk_size, chrom.length // 4)
            else:
                chunk_size = self.config.chunk_size
            
            chunk_size = max(chunk_size, 100_000)  # Minimum chunk size
            
            for start in range(0, chrom.length, chunk_size):
                end = min(start + chunk_size, chrom.length)
                chunks.append((chrom.name, start, end))
        
        logger.info(f"Generated {len(chunks)} processing chunks")
        return chunks

    def _partition_chunks(self, chunks: list[Tuple[str, int, int]], n: int) -> list[list[Tuple[str, int, int]]]:
        """Partition chunks into roughly equal batches for workers."""
        batches: list[list[Tuple[str, int, int]]] = [[] for _ in range(n)]
        for i, chunk in enumerate(chunks):
            batches[i % n].append(chunk)
        return [batch for batch in batches if batch]

    def _worker_process_chunks(
        self, 
        chunks: list[Tuple[str, int, int]], 
        progress: Progress, 
        task_id: int
    ) -> Dict[str, int]:
        """Worker function that processes a batch of chunks."""
        local_stats = {'chunks_processed': 0, 'chunks_failed': 0, 'total_values': 0, 'total_bytes': 0}

        # Per-worker handles to minimize overhead
        bw_handle = None
        tdb_array = None
        
        try:
            bw_handle = pybigtools.open(str(self.bigwig_path))
            _THREAD_LOCAL.bw_handle = bw_handle
            tdb_array = tiledb.open(str(self.tiledb_path), 'w')

            # Buffered writing to reduce per-chunk overhead
            write_buffer = WriteBuffer()

            for idx, chunk in enumerate(chunks):
                chunk_result = self._process_single_chunk(chunk)
                
                if chunk_result is None:
                    local_stats['chunks_failed'] += 1
                    continue

                write_buffer.add(chunk_result)

                # Flush buffer when full or at end
                if write_buffer.should_flush(self.config.write_batch_size) or idx == len(chunks) - 1:
                    try:
                        bytes_written = self._flush_buffer(write_buffer, tdb_array)
                        local_stats['chunks_processed'] += write_buffer.count
                        local_stats['total_values'] += write_buffer.total_values
                        local_stats['total_bytes'] += bytes_written
                        write_buffer.clear()
                    except Exception as e:
                        logger.error(f"Worker batch write failed: {e}")
                        local_stats['chunks_failed'] += write_buffer.count
                        write_buffer.clear()

                # Update progress periodically
                if (idx & 0x0F) == 0:  # Every 16 chunks
                    advance = min(16, len(chunks) - idx)
                    progress.update(task_id, advance=advance)

        finally:
            self._cleanup_worker_resources(bw_handle, tdb_array)

        return local_stats

    def _process_single_chunk(self, chunk_info: Tuple[str, int, int]) -> Optional[Dict[str, Any]]:
        """Process a single genomic chunk."""
        chrom, start, end = chunk_info

        # Reuse per-thread bigWig handle
        bw_handle = getattr(_THREAD_LOCAL, 'bw_handle', None)
        
        try:
            if bw_handle is None:
                bw_handle = pybigtools.open(str(self.bigwig_path))
                _THREAD_LOCAL.bw_handle = bw_handle
            
            values = bw_handle.values(chrom, start, end)
        except Exception:
            return None

        if values is None or len(values) == 0:
            return None

        # Clean and filter values (avoid extra copies – work in float32)
        if values.dtype != np.float32:
            values = values.astype(np.float32, copy=False)
        cleaned_values = normalize_genomic_values(values)

        # Fast path: skip chunks with all zeros quickly
        if np.all(cleaned_values == 0.0):  # cheap rejection
            return None

        # Filter out zero values for sparse storage
        nonzero_mask = cleaned_values != 0.0
        if not nonzero_mask.any():  # secondary guard (in case of all zeros)
            return None

        positions = np.nonzero(nonzero_mask)[0].astype(np.uint64, copy=False) + np.uint64(start)
        values_out = cleaned_values[nonzero_mask]

        # Optional statistics only if needed for future extensions (compute lazily)
        # Kept lightweight by not allocating masks twice; compute on filtered values.
        # (Mean/max/min used only for potential logging or downstream QC.)
        try:
            mean_val = float(values_out.mean())
            max_val = float(values_out.max())
            min_val = float(values_out.min())
        except Exception:
            mean_val = max_val = min_val = 0.0

        # Use integer contig IDs if configured (saves memory + TileDB ASCII overhead)
        if self._contig_map is not None:
            chrom_id = np.uint32(self._contig_map.get(chrom, 0))
            chrom_arr = np.full(len(positions), chrom_id, dtype=np.uint32)
        else:
            chrom_arr = np.full(len(positions), chrom, dtype=f'U{len(chrom)}')

        return {
            'chromosomes': chrom_arr,
            'positions': positions,
            'values': values_out,
            'value_count': values_out.size,
            'stats': {'mean': mean_val, 'max': max_val, 'min': min_val},
        }

    def _update_stats(self, stats: ConversionMetrics, result: Dict[str, int]) -> None:
        """Update conversion statistics with worker results."""
        stats.chunks_processed += result.get('chunks_processed', 0)
        stats.chunks_failed += result.get('chunks_failed', 0)
        stats.total_values += result.get('total_values', 0)
        stats.total_bytes += result.get('total_bytes', 0)

    def _finalize_metrics(self, stats: ConversionMetrics, start_time: float, chromosomes: list[ContigInfo]) -> None:
        """Finalize conversion metrics and display summary."""
        stats.total_time_seconds = time.perf_counter() - start_time
        
        if stats.total_bytes > 0:
            array_size = self._get_tiledb_size()
            stats.compression_ratio = stats.total_bytes / array_size if array_size > 0 else 1.0

        self._display_conversion_summary(stats, chromosomes)

    def _cleanup_worker_resources(self, bw_handle: Any, tdb_array: Any) -> None:
        """Clean up worker resources."""
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

    def _get_tiledb_size(self) -> int:
        """Calculate total TileDB array size on disk."""
        try:
            return sum(
                path.stat().st_size 
                for path in self.tiledb_path.rglob('*') 
                if path.is_file()
            )
        except Exception:
            return 0

    def _display_conversion_summary(self, stats: ConversionMetrics, chromosomes: list[ContigInfo]) -> None:
        """Display a formatted summary of conversion results."""
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

    def _flush_buffer(self, buffer: 'WriteBuffer', tdb_array: Any) -> int:
        """Flush write buffer to TileDB array."""
        if buffer.is_empty():
            return 0
        # Concatenate once; avoid implicit Python loops where possible
        all_chrom = np.concatenate(buffer.chromosomes)
        all_pos = np.concatenate(buffer.positions)
        all_vals = np.concatenate(buffer.values)
        if all_vals.dtype != self._value_np_dtype:
            all_vals = all_vals.astype(self._value_np_dtype, copy=False)

        # Perform a single vectorized assignment
        tdb_array[all_chrom, all_pos] = all_vals
        return int(all_vals.size * all_vals.itemsize)


class WriteBuffer:
    """Buffer for batched writes to reduce TileDB overhead."""
    
    def __init__(self):
        self.clear()
    
    def clear(self) -> None:
        """Clear all buffers."""
        self.chromosomes: list = []
        self.positions: list = []
        self.values: list = []
        self.count = 0
        self.total_values = 0
    
    def add(self, chunk_data: Dict[str, Any]) -> None:
        """Add chunk data to buffer."""
        self.chromosomes.append(chunk_data['chromosomes'])
        self.positions.append(chunk_data['positions'])
        self.values.append(chunk_data['values'])
        self.count += 1
        self.total_values += chunk_data['value_count']
    
    def should_flush(self, batch_size: int) -> bool:
        """Check if buffer should be flushed."""
        return self.count >= batch_size
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self.count == 0



class TileDBQueryEngine:
    """Query engine for TileDB-backed genomic signal data.

    Provides a simple API for querying genomic regions and benchmarking
    query performance against TileDB arrays.
    """

    def __init__(self, tiledb_path: Path | str):
        self.tiledb_path = Path(tiledb_path)
        
        if tiledb is None:
            raise RuntimeError("TileDB is not installed; TileDBQueryEngine unavailable")
        
        if not self.tiledb_path.exists():
            raise FileNotFoundError(f"TileDB array not found: {self.tiledb_path}")

        # Detect integer chromosome mapping via metadata (if present)
        self._contig_map: Optional[dict[str, int]] = None
        self._reverse_contig_map: Optional[list[str]] = None
        try:
            import json
            with tiledb.open(str(self.tiledb_path), 'r') as arr:
                if "contig_map_json" in arr.meta:
                    raw = arr.meta["contig_map_json"]
                    if isinstance(raw, (bytes, bytearray)):
                        raw = raw.decode('utf-8')
                    self._contig_map = json.loads(raw)
                    rev = sorted(self._contig_map.items(), key=lambda kv: kv[1])
                    self._reverse_contig_map = [k for k, _ in rev]
        except Exception:
            self._contig_map = None

    def query_region(self, chrom: str | int, start: int, end: int, *, values_only: bool = False) -> np.ndarray:
        """Query values for a genomic region (no silent fallbacks).

        Parameters:
            chrom: chromosome identifier (string or integer if contig map used)
            start: 0-based inclusive start
            end: 0-based exclusive end (must be > start)
            values_only: if True, skips coordinate materialization using Arrow dims=False/has_coords=False path

        Returns:
            1D float32 numpy array of values for region (may be empty)
        """
        if end <= start:
            raise ValueError("end must be > start")
        if self._contig_map is not None and isinstance(chrom, str):
            chrom = self._map_chromosome(chrom)
        with tiledb.open(str(self.tiledb_path), 'r') as arr:
            if values_only:
                return self._range_query_values_only(arr, chrom, start, end)
            return self._range_query(arr, chrom, start, end)

    # ----------------------------------------------------------------------------------
    # Internal helpers (range addition compatibility)
    # ----------------------------------------------------------------------------------

    def _add_range(self, q: Any, arr: Any, dim_name: str, start: int, end_exclusive: int) -> None:
        """Add a half-open [start, end_exclusive) range to a query.

        Handles both modern query.add_range(dim_name, start, end_inclusive) API and
        older query.subarray().add_range(dim_index, start, end_inclusive) patterns.
        """
        end_inclusive = end_exclusive - 1
        if end_inclusive < start:
            return  # empty interval
        if hasattr(q, 'add_range'):
            q.add_range(dim_name, start, end_inclusive)
            return
        # Fallback: use Subarray API if available
        if hasattr(q, 'subarray'):
            try:
                sub = q.subarray()
                # Find dimension index
                schema = arr.schema if hasattr(arr, 'schema') else None
                if schema is None:
                    raise RuntimeError("Cannot resolve dimension index for fallback range addition")
                for idx, dim in enumerate(schema.domain):  # type: ignore[attr-defined]
                    if dim.name == dim_name:
                        sub.add_range(idx, start, end_inclusive)
                        return
                raise KeyError(f"Dimension '{dim_name}' not found in schema for range addition")
            except Exception as e:
                raise RuntimeError(f"Failed to add range via subarray fallback: {e}") from e
        raise RuntimeError("TileDB Python package missing 'add_range' and Subarray fallback; upgrade TileDB.")

    def _execute_query(self, arr: Any, chrom: str | int, start: int, end: int) -> np.ndarray:  # legacy shim
        return self._range_query(arr, chrom, start, end)

    def _range_query(self, arr: Any, chrom: str | int, start: int, end: int) -> np.ndarray:
        """Execute range-based query using TileDB query API (strict, fast)."""
        if self._contig_map is not None and isinstance(chrom, str):
            chrom = self._map_chromosome(chrom)
        # Try Arrow zero-copy path
        use_arrow = False
        try:
            query = arr.query(return_arrow=True, use_arrow=True, index_col=False, dims=True, has_coords=True)
            use_arrow = True
        except Exception:
            query = arr.query()
        self._add_range(query, arr, 'chromosome', chrom, chrom + 1 if isinstance(chrom, int) else chrom + 1 if False else chrom + 1)  # type: ignore[arg-type]
        # For chromosome (categorical) we add as a single-value inclusive range; above line is awkward due to typing.
        # Simpler: re-add explicit logic below for clarity.
        try:
            # Replace previous generic attempt: remove the multi-evaluation side-effects
            if hasattr(query, 'add_range'):
                query.add_range('chromosome', chrom, chrom)
            else:
                # Subarray fallback for chromosome dim (index 0 assumed)
                if hasattr(query, 'subarray'):
                    sub = query.subarray()
                    sub.add_range(0, chrom, chrom)
                else:
                    raise RuntimeError("Missing add_range and subarray for chromosome dim")
        except Exception as e:
            raise RuntimeError(f"Failed to add chromosome range: {e}") from e
        # Position dimension half-open
        if hasattr(query, 'add_range'):
            query.add_range('position', start, end - 1)
        else:
            if hasattr(query, 'subarray'):
                sub = query.subarray()
                sub.add_range(1, start, end - 1)
            else:
                raise RuntimeError("Missing add_range and subarray for position dim")
        data_obj = query[:]  # Arrow Table or dict-like / DataFrame
        if use_arrow:
            try:
                # PyArrow Table path
                val_col = data_obj.column('value') if hasattr(data_obj, 'column') else None
                if val_col is None or val_col.num_chunks == 0 or val_col.length() == 0:
                    return np.empty(0, dtype=np.float32)
                # Combine chunks (usually 1) and convert to numpy
                return val_col.combine_chunks().to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
            except Exception:
                # Fallback to generic extraction path below
                pass
        # Generic fallback expects pandas-like df
        if data_obj is None or len(data_obj) == 0:  # type: ignore[arg-type]
            return np.empty(0, dtype=np.float32)
        try:
            return data_obj['value'].to_numpy(dtype=np.float32)  # pandas DataFrame
        except Exception:
            # Dict-like path
            if isinstance(data_obj, dict) and 'value' in data_obj:
                return np.asarray(data_obj['value'], dtype=np.float32)
        return np.empty(0, dtype=np.float32)

    def _range_query_values_only(self, arr: Any, chrom: str | int, start: int, end: int) -> np.ndarray:
        """Execute value-only range query using Arrow fast path (no coordinates).

        Falls back to generic path if Arrow not available.
        """
        if self._contig_map is not None and isinstance(chrom, str):
            chrom = self._map_chromosome(chrom)
        use_arrow = False
        try:
            q = arr.query(return_arrow=True, use_arrow=True, index_col=False, dims=False, has_coords=False, attrs=['value'])
            use_arrow = True
        except Exception:
            q = arr.query()
        if hasattr(q, 'add_range'):
            q.add_range('chromosome', chrom, chrom)
            q.add_range('position', start, end - 1)
        else:
            if hasattr(q, 'subarray'):
                sub = q.subarray()
                sub.add_range(0, chrom, chrom)
                sub.add_range(1, start, end - 1)
            else:
                raise RuntimeError("TileDB python package missing 'add_range' and subarray fallback API; upgrade TileDB.")
        data_obj = q[:]
        if use_arrow and hasattr(data_obj, 'column'):
            try:
                val_col = data_obj.column('value')
                if val_col.length() == 0:
                    return np.empty(0, dtype=np.float32)
                return val_col.combine_chunks().to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
            except Exception:
                pass
        if data_obj is None or len(data_obj) == 0:  # type: ignore[arg-type]
            return np.empty(0, dtype=np.float32)
        try:
            return data_obj['value'].to_numpy(dtype=np.float32)
        except Exception:
            if isinstance(data_obj, dict) and 'value' in data_obj:
                return np.asarray(data_obj['value'], dtype=np.float32)
        return np.empty(0, dtype=np.float32)

    def _extract_values(self, result: Any) -> np.ndarray:  # retained for API stability
        if result is None:
            return np.empty(0, dtype=np.float32)
        if isinstance(result, dict):  # uncommon path now
            if 'value' not in result:
                raise KeyError("Result dict missing 'value' key")
            return np.asarray(result['value'], dtype=np.float32)
        return np.asarray(result, dtype=np.float32)

    def _map_chromosome(self, chrom: str) -> int:
        if self._contig_map is None:
            raise RuntimeError("Integer contig map not available")
        try:
            return self._contig_map[chrom]  # type: ignore[index]
        except KeyError as e:
            raise KeyError(f"Chromosome '{chrom}' not present in contig map") from e

    def query_multiple_regions(self, regions: List[GenomicRegion]) -> Dict[str, np.ndarray]:
        """Query multiple genomic regions efficiently.

        Args:
            regions: List of GenomicRegion objects to query

        Returns:
            Dictionary mapping region strings to value arrays
        """
        if not regions:
            return {}
        # Normalize & validate regions first
        grouped: dict[int | str, list[GenomicRegion]] = {}
        use_map = self._contig_map is not None
        for r in regions:
            if r.end <= r.start:
                raise ValueError(f"Invalid region {r}: end <= start")
            key: int | str
            if use_map:
                if r.chrom not in self._contig_map:  # type: ignore[operator]
                    raise KeyError(f"Chromosome '{r.chrom}' missing in contig map")
                key = self._contig_map[r.chrom]  # type: ignore[index]
            else:
                key = r.chrom
            grouped.setdefault(key, []).append(r)

        out: Dict[str, np.ndarray] = {}
        with tiledb.open(str(self.tiledb_path), 'r') as arr:
            for chrom_key, sub in grouped.items():
                # Sort regions to enable interval merging
                sub_sorted = sorted(sub, key=lambda r: r.start)
                merged: list[tuple[int,int]] = []
                cur_s, cur_e = sub_sorted[0].start, sub_sorted[0].end
                for r in sub_sorted[1:]:
                    if r.start <= cur_e:  # overlap/adjacent
                        if r.end > cur_e:
                            cur_e = r.end
                    else:
                        merged.append((cur_s, cur_e))
                        cur_s, cur_e = r.start, r.end
                merged.append((cur_s, cur_e))

                # Attempt Arrow path including coordinates for vectorized slicing
                try:
                    q = arr.query(return_arrow=True, use_arrow=True, index_col=False, dims=True, has_coords=True)
                except Exception:
                    q = arr.query()
                if hasattr(q, 'add_range'):
                    q.add_range('chromosome', chrom_key, chrom_key)
                else:
                    if hasattr(q, 'subarray'):
                        sub = q.subarray()
                        sub.add_range(0, chrom_key, chrom_key)
                    else:
                        raise RuntimeError("TileDB python package missing add_range and subarray APIs")
                for (ms, me) in merged:
                    if hasattr(q, 'add_range'):
                        q.add_range('position', ms, me - 1)
                    else:
                        if hasattr(q, 'subarray'):
                            sub = q.subarray()
                            sub.add_range(1, ms, me - 1)
                        else:
                            raise RuntimeError("TileDB python package missing add_range and subarray APIs")
                data_obj = q[:]  # may be Arrow or pandas depending on TileDB build
                if data_obj is None or len(data_obj) == 0:  # type: ignore[arg-type]
                    for r in sub:
                        out[str(r)] = np.empty(0, dtype=np.float32)
                    continue
                # Arrow fast path
                if hasattr(data_obj, 'column'):
                    try:
                        pos_arr = data_obj.column('position').combine_chunks().to_numpy(zero_copy_only=False)
                        val_arr = data_obj.column('value').combine_chunks().to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
                        starts = np.fromiter((r.start for r in sub), count=len(sub), dtype=pos_arr.dtype)
                        ends = np.fromiter((r.end for r in sub), count=len(sub), dtype=pos_arr.dtype)
                        idx_starts = np.searchsorted(pos_arr, starts, side='left')
                        idx_ends = np.searchsorted(pos_arr, ends, side='left')
                        for i, r in enumerate(sub):
                            out[str(r)] = val_arr[idx_starts[i]:idx_ends[i]]
                        continue
                    except Exception:
                        pass
                # Pandas/DataFrame path
                pos = data_obj['position'].to_numpy()
                vals = data_obj['value'].to_numpy(dtype=np.float32)
                starts = np.fromiter((r.start for r in sub), count=len(sub), dtype=pos.dtype)
                ends = np.fromiter((r.end for r in sub), count=len(sub), dtype=pos.dtype)
                idx_starts = np.searchsorted(pos, starts, side='left')
                idx_ends = np.searchsorted(pos, ends, side='left')
                for i, r in enumerate(sub):
                    out[str(r)] = vals[idx_starts[i]:idx_ends[i]]
        return out

    def query_region_aggregate(
        self,
        chrom: str | int,
        start: int,
        end: int,
        aggs: list[str] | dict[str, str | list[str]] | str = ("sum", "count", "mean"),
    ) -> dict[str, float | int]:
        """Server-side aggregation over a single region using TileDB .agg.

        Returns mapping of aggregate name -> value. Normalizes unnamed attribute key ''.
        """
        if end <= start:
            raise ValueError("end must be > start")
        if self._contig_map is not None and isinstance(chrom, str):
            chrom = self._map_chromosome(chrom)
        with tiledb.open(str(self.tiledb_path), 'r') as arr:
            q = arr.query().agg(aggs)
            if hasattr(q, 'add_range'):
                q.add_range('chromosome', chrom, chrom)
                q.add_range('position', start, end - 1)
            else:
                if hasattr(q, 'subarray'):
                    sub = q.subarray()
                    sub.add_range(0, chrom, chrom)
                    sub.add_range(1, start, end - 1)
                else:
                    raise RuntimeError("TileDB python package missing add_range and subarray APIs")
            res = q[:]
            if isinstance(res, (int, float)):
                if isinstance(aggs, str):
                    return {aggs: res}
                return {"value": res}
            if "" in res and isinstance(res[""], dict):
                return res[""]
            if "" in res:
                if isinstance(aggs, (list, tuple)):
                    return {str(aggs[0]): res[""]}
                return {"value": res[""]}
            return res

    def query_regions_aggregate(
        self,
        regions: list[GenomicRegion],
        aggs: list[str] | dict[str, str | list[str]] | str = ("sum", "count", "mean"),
    ) -> dict[str, dict[str, float | int]]:
        """Per-region aggregation using TileDB server-side operations.

        Loops over regions; for many small regions consider batching and client-side reduction.
        Returns mapping of region string -> aggregation results.
        """
        out: dict[str, dict[str, float | int]] = {}
        if not regions:
            return out
        with tiledb.open(str(self.tiledb_path), 'r') as arr:
            for r in regions:
                if r.end <= r.start:
                    raise ValueError(f"Invalid region {r} (end <= start)")
                chrom_key: str | int = r.chrom
                if self._contig_map is not None and isinstance(chrom_key, str):
                    chrom_key = self._map_chromosome(chrom_key)
                q = arr.query().agg(aggs)
                if hasattr(q, 'add_range'):
                    q.add_range('chromosome', chrom_key, chrom_key)
                    q.add_range('position', r.start, r.end - 1)
                else:
                    if hasattr(q, 'subarray'):
                        sub = q.subarray()
                        sub.add_range(0, chrom_key, chrom_key)
                        sub.add_range(1, r.start, r.end - 1)
                    else:
                        raise RuntimeError("TileDB python package missing add_range and subarray APIs")
                res = q[:]
                if isinstance(res, (int, float)):
                    if isinstance(aggs, str):
                        out[str(r)] = {aggs: res}
                    else:
                        out[str(r)] = {"value": res}
                elif "" in res and isinstance(res[""], dict):
                    out[str(r)] = res[""]
                elif "" in res:
                    if isinstance(aggs, (list, tuple)):
                        out[str(r)] = {str(aggs[0]): res[""]}
                    else:
                        out[str(r)] = {"value": res[""]}
                else:
                    out[str(r)] = res  # Already mapping
        return out
    
    def query_multiple_regions_array(self, regions: List[GenomicRegion], n_bins: int) -> np.ndarray:
        """Query multiple regions and return mean-binned matrix (vectorized)."""
        if n_bins <= 0:
            raise ValueError("n_bins must be positive")
        res = np.zeros((len(regions), n_bins), dtype=np.float32)
        with tiledb.open(str(self.tiledb_path), 'r') as arr:
            for i, region in enumerate(regions):
                chrom = region.chrom
                if self._contig_map is not None and isinstance(chrom, str):
                    chrom = self._map_chromosome(chrom)
                vals = self._range_query(arr, chrom, region.start, region.end)
                if vals.size == 0:
                    continue
                edges = np.linspace(0, vals.size, n_bins + 1, dtype=np.int32)
                # Use reduceat for fast bin sums
                bin_sums = np.add.reduceat(vals, edges[:-1])
                counts = (edges[1:] - edges[:-1]).clip(min=1)
                res[i] = bin_sums / counts
        return res

    def get_array_info(self) -> Dict[str, Any]:
        """Get information about the TileDB array.

        Returns:
            Dictionary with array metadata and schema information
        """
        try:
            with tiledb.open(str(self.tiledb_path), 'r') as arr:
                schema = arr.schema
                
                return {
                    'array_type': 'sparse' if schema.sparse else 'dense',
                    'dimensions': [dim.name for dim in schema.domain],
                    'attributes': [attr.name for attr in schema.attrs],
                    'tile_extents': {dim.name: dim.tile for dim in schema.domain},
                    'compression': {
                        attr.name: [str(f) for f in attr.filters] 
                        for attr in schema.attrs
                    },
                    'capacity': schema.capacity,
                }
        except Exception as e:
            logger.error(f"Failed to get array info: {e}")
            return {}

    def bench_queries(self, queries: List[GenomicRegion], iterations: int = 3) -> QueryBenchmark:
        """Benchmark query performance across multiple iterations.

        Args:
            queries: List of genomic regions to query
            iterations: Number of benchmark iterations

        Returns:
            QueryBenchmark object with timing statistics
        """
        if not queries:
            raise ValueError("At least one query region is required")

        times: List[float] = []
        total_values = 0

        logger.info(f"Running benchmark: {len(queries)} queries × {iterations} iterations")

        for iteration in range(iterations):
            iteration_start = time.perf_counter()
            iteration_values = 0
            
            for query in queries:
                try:
                    values = self.query_region(query.chrom, query.start, query.end)
                    iteration_values += len(values)
                except Exception as e:
                    logger.warning(f"Benchmark query failed for {query}: {e}")
            
            iteration_time = time.perf_counter() - iteration_start
            times.append(iteration_time)
            total_values += iteration_values

        return QueryBenchmark(
            total_queries=len(queries),
            iterations=iterations,
            query_times=times,
            total_values_retrieved=total_values
        )

    def estimate_cache_hit_rate(self, queries: List[GenomicRegion], warm_iterations: int = 2) -> float:
        """Estimate cache hit rate by comparing cold vs warm query times.

        Args:
            queries: List of genomic regions to query
            warm_iterations: Number of warm-up iterations

        Returns:
            Estimated cache hit rate as a percentage (0-100)
        """
        if not queries:
            return 0.0

        # Cold run (first query)
        cold_start = time.perf_counter()
        for query in queries:
            self.query_region(query.chrom, query.start, query.end)
        cold_time = time.perf_counter() - cold_start

        # Warm runs (repeated queries)
        warm_times = []
        for _ in range(warm_iterations):
            warm_start = time.perf_counter()
            for query in queries:
                self.query_region(query.chrom, query.start, query.end)
            warm_times.append(time.perf_counter() - warm_start)

        avg_warm_time = np.mean(warm_times)
        
        # Estimate cache hit rate based on time improvement
        if cold_time > 0:
            improvement = max(0, (cold_time - avg_warm_time) / cold_time)
            return min(100.0, improvement * 100.0)
        
        return 0.0


def create_sample_queries(
    chromosomes: List[str], 
    max_position: int = 1_000_000, 
    num_queries: int = 100,
    region_size_range: Tuple[int, int] = (1000, 100_000)
) -> List[GenomicRegion]:
    """Create a set of sample genomic queries for benchmarking.

    Args:
        chromosomes: List of chromosome names
        max_position: Maximum position for random queries  
        num_queries: Number of queries to generate
        region_size_range: Tuple of (min_size, max_size) for region lengths

    Returns:
        List of GenomicRegion objects for benchmarking
    """
    if not chromosomes:
        raise ValueError("At least one chromosome is required")

    queries = []
    np.random.seed(42)  # Reproducible queries
    
    min_size, max_size = region_size_range
    
    for _ in range(num_queries):
        chrom = np.random.choice(chromosomes)
        region_size = np.random.randint(min_size, max_size + 1)
        start = np.random.randint(0, max(1, max_position - region_size))
        end = start + region_size
        
        queries.append(GenomicRegion(chrom=chrom, start=start, end=end))
    
    return queries


def validate_genomic_data_integrity(
    bigwig_path: Path | str, 
    tiledb_path: Path | str, 
    sample_regions: Optional[List[GenomicRegion]] = None,
    tolerance: float = 1e-6
) -> Dict[str, bool]:
    """Validate data integrity between bigWig and TileDB representations.

    Args:
        bigwig_path: Path to original bigWig file
        tiledb_path: Path to TileDB array
        sample_regions: Optional list of regions to check (random if None)
        tolerance: Numerical tolerance for floating point comparisons

    Returns:
        Dictionary with validation results
    """
    if pybigtools is None or tiledb is None:
        raise RuntimeError("Both pybigtools and tiledb are required for validation")

    bigwig_path = Path(bigwig_path)
    tiledb_path = Path(tiledb_path)
    
    if not bigwig_path.exists():
        raise FileNotFoundError(f"BigWig file not found: {bigwig_path}")
    if not tiledb_path.exists():
        raise FileNotFoundError(f"TileDB array not found: {tiledb_path}")

    # Create sample regions if not provided
    if sample_regions is None:
        with pybigtools.open(str(bigwig_path)) as bw:
            chroms = list(bw.chroms().keys())[:5]  # Test first 5 chromosomes
        sample_regions = create_sample_queries(chroms, num_queries=20, region_size_range=(10000, 50000))

    query_engine = TileDBQueryEngine(tiledb_path)
    results = {
        'regions_tested': len(sample_regions),
        'all_passed': True,
        'individual_results': [],
        'summary': {}
    }

    passed = 0
    failed = 0
    
    logger.info(f"Validating data integrity across {len(sample_regions)} regions...")

    with pybigtools.open(str(bigwig_path)) as bw:
        for region in sample_regions:
            try:
                # Get values from both sources
                bw_values = bw.values(region.chrom, region.start, region.end)
                tdb_values = query_engine.query_region(region.chrom, region.start, region.end)
                
                # Normalize both arrays
                if bw_values is not None:
                    bw_cleaned = normalize_genomic_values(bw_values.astype(np.float64))
                    bw_nonzero = bw_cleaned[bw_cleaned != 0.0]
                else:
                    bw_nonzero = np.array([])
                
                tdb_nonzero = tdb_values[tdb_values != 0.0] if len(tdb_values) > 0 else np.array([])
                
                # Compare non-zero values
                values_match = np.allclose(
                    np.sort(bw_nonzero), 
                    np.sort(tdb_nonzero), 
                    rtol=tolerance, 
                    atol=tolerance
                )
                
                region_result = {
                    'region': str(region),
                    'passed': values_match,
                    'bw_count': len(bw_nonzero),
                    'tdb_count': len(tdb_nonzero),
                }
                
                results['individual_results'].append(region_result)
                
                if values_match:
                    passed += 1
                else:
                    failed += 1
                    results['all_passed'] = False
                    logger.warning(f"Validation failed for {region}: "
                                 f"BigWig={len(bw_nonzero)} values, TileDB={len(tdb_nonzero)} values")
                    
            except Exception as e:
                failed += 1
                results['all_passed'] = False
                logger.error(f"Validation error for {region}: {e}")
                results['individual_results'].append({
                    'region': str(region),
                    'passed': False,
                    'error': str(e)
                })

    results['summary'] = {
        'passed': passed,
        'failed': failed,
        'success_rate': (passed / len(sample_regions)) * 100 if sample_regions else 0
    }

    logger.info(f"Validation complete: {passed}/{len(sample_regions)} regions passed "
                f"({results['summary']['success_rate']:.1f}%)")

    return results