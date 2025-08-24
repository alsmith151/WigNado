from __future__ import annotations

"""wignado.engine

Core utilities and conversion helpers for working with genomic
bigWig-like data and (optionally) converting to TileDB-backed stores.

This module intentionally keeps heavy external dependencies at top-level
but many functions are pure and can be unit-tested with lightweight stubs.
"""

import time
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable
import warnings

import bigtools
import numpy as np
import psutil
from loguru import logger
from numba import jit, prange
from pydantic import BaseModel, Field, field_validator
from rich.console import Console

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
    """Converter class that orchestrates reading a bigWig file (via bigtools)

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
            with bigtools.open(str(self.bigwig_path)) as bw:
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
            chromosomes = self._read_chromosome_info_fast()
            total_bp = sum(chrom.length for chrom in chromosomes)

        logger.info(f"Found {len(chromosomes)} chromosomes, {total_bp:,} total bp")

        with console.status("[bold green]Creating TileDB schema..."):
            self._create_optimized_schema(chromosomes)

        stats = self._convert_with_rich_progress(chromosomes)

        total_time = time.perf_counter() - start_time
        stats.total_time_seconds = total_time
        return stats
