from __future__ import annotations
"""Zarr-backed genomic bigWig conversion & query utilities.

Lightweight sibling to the TileDB implementation in `tiledb.py` providing:
  * BigWig -> Zarr conversion (sparse-like via chunked dense per-chrom arrays)
  * Region querying & multi-region binned summaries

Design goals:
  * Minimal dependencies (uses pybigtools + zarr + numpy)
  * Streaming, memory-bounded conversion
  * Intuitive, inspectable Zarr hierarchy:
        <root>/
            .zattrs (metadata: version, contigs)
            chrom_sizes (1D array of contig lengths)
            contigs (1D array of contig names)
            data/<chrom> (1D float array length = chrom length, chunked)

We store each chromosome as an independent Zarr array to allow selective
loading. This keeps implementation simple while still performing well for
random region extraction. Storage is dense (size = genome length * dtype)
so this format is best for moderate-sized genomes or when signal is fairly
continuous. (For highly sparse signals prefer TileDB sparse schema.)
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Sequence
import time

import numpy as np
from loguru import logger
from pydantic import BaseModel
import tqdm

# Optional heavy deps
try:  # pragma: no cover
    import pybigtools  # type: ignore
except ImportError:  # pragma: no cover
    pybigtools = None  # type: ignore

try:  # pragma: no cover
    import zarr  # type: ignore
    from numcodecs import Zstd  # type: ignore
except ImportError:  # pragma: no cover
    zarr = None  # type: ignore

# --------------------------------------------------------------------------------------
# Configuration dataclass
# --------------------------------------------------------------------------------------

class ZarrConverterConfig(BaseModel):
    overwrite: bool = False
    chunk_size: int = 1_000_000          # per-chromosome chunk length in bp
    value_dtype: str = "float32"          # 'float32' or 'float16'
    compression_level: int = 5           # 1-22 (Zstd)
    min_chrom_length: int = 1000         # skip tiny contigs
    zero_fill_value: float = 0.0         # baseline value for missing / invalid
    batch_bp: int = 5_000_000            # streaming window size for reading BigWig
    include_mapping: bool = True         # store contig name -> id mapping


# --------------------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------------------

class BigWigToZarrConverter:
    def __init__(self, bigwig_path: str | Path, zarr_path: str | Path, config: ZarrConverterConfig | None = None):
        self.bigwig_path = Path(bigwig_path)
        self.zarr_path = Path(zarr_path)
        self.cfg = config or ZarrConverterConfig()
        self._validate()

    def _validate(self) -> None:
        if pybigtools is None:
            raise RuntimeError("pybigtools not installed; install with extras 'tiledb' or 'zarr'")
        if zarr is None:
            raise RuntimeError("zarr/numcodecs not installed; install with extras 'zarr'")
        if not self.bigwig_path.exists():
            raise FileNotFoundError(f"BigWig file not found: {self.bigwig_path}")
        with pybigtools.open(str(self.bigwig_path)) as bw:  # sanity
            _ = bw.chroms()
        if self.zarr_path.exists() and self.cfg.overwrite:
            import shutil
            shutil.rmtree(self.zarr_path)
        self.zarr_path.parent.mkdir(parents=True, exist_ok=True)

    # Public API
    def convert(self) -> dict:
        start = time.perf_counter()
        with pybigtools.open(str(self.bigwig_path)) as bw:
            chroms_dict = bw.chroms()
            chrom_meta = [(c, int(length_val)) for c, length_val in chroms_dict.items() if length_val >= self.cfg.min_chrom_length]
        if not chrom_meta:
            raise ValueError("No chromosomes meet minimum length criterion")
        # Create root group

        store = zarr.storage.LocalStore(str(self.zarr_path))
        root = zarr.create_group(store=store, path='/', overwrite=self.cfg.overwrite)
        root.attrs.update({
            'format': 'wignado.zarr.v2',
            'bigwig_source': str(self.bigwig_path),
            'created': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'config': self.cfg.model_dump_json(),
        })
        contigs = [c for c,_ in chrom_meta]
        lengths = np.array([length_val for _, length_val in chrom_meta], dtype=np.uint64)
        root.create_array('contigs', data=np.array(contigs, dtype=str), overwrite=True)
        root.create_array('chrom_sizes', data=lengths, overwrite=True)
        # Mapping name->id stored in attrs for categorical representation
        contig_id_map = {name: i for i, name in enumerate(contigs)} if self.cfg.include_mapping else {}
        if contig_id_map:
            root.attrs['contig_id_map'] = contig_id_map

        total_bp = int(lengths.sum())
        value_dtype = self.cfg.value_dtype
        # Create per-chrom arrays under integer-coded names c{ID}
        g_data = root.require_group('data')
        for chrom, length in chrom_meta:
            chrom_id = contig_id_map.get(chrom, chrom)
            ds_name = f"c{chrom_id}" if isinstance(chrom_id, int) else chrom
            chunks = (min(self.cfg.chunk_size, length),)
            if ds_name in g_data and not self.cfg.overwrite:
                raise FileExistsError(f"Chromosome array already exists: {ds_name}")
            g_data.create_array(
                ds_name,
                shape=(length,),
                chunks=chunks,
                dtype=value_dtype,
                compressor="auto",
                overwrite=True,
                fill_value=self.cfg.zero_fill_value,
            )
        # Stream fill per chromosome
        written_values = 0
        for chrom, length in tqdm.tqdm(chrom_meta, total=len(chrom_meta)):
            chrom_id = root.attrs.get('contig_id_map', {}).get(chrom, chrom)
            ds_name = f"c{chrom_id}" if isinstance(chrom_id, int) else chrom
            arr = g_data[ds_name]
            for start_pos in range(0, length, self.cfg.batch_bp):
                end_pos = min(start_pos + self.cfg.batch_bp, length)
                # Read values (pybigtools returns numpy array length = end-start)
                with pybigtools.open(str(self.bigwig_path)) as bw:  # open inside loop keeps handle short-lived
                    vals = bw.values(chrom, start_pos, end_pos)
                if vals is None or len(vals) == 0:
                    continue
                if vals.dtype != value_dtype:
                    vals = vals.astype(value_dtype, copy=False)
                # Replace invalids
                invalid_mask = ~np.isfinite(vals)
                if invalid_mask.any():
                    vals[invalid_mask] = self.cfg.zero_fill_value
                arr[start_pos:end_pos] = vals
                written_values += (end_pos - start_pos)
        elapsed = time.perf_counter() - start
        root.attrs['conversion_seconds'] = elapsed
        root.attrs['total_basepairs'] = total_bp
        root.attrs['values_written'] = written_values
        logger.info(f"Completed conversion in {elapsed:.1f}s; {total_bp/1e6:.2f} Mbp processed")
        return {
            'chromosomes': len(chrom_meta),
            'total_bp': total_bp,
            'values_written': written_values,
            'seconds': elapsed,
            'zarr_path': str(self.zarr_path),
        }

# --------------------------------------------------------------------------------------
# Query Engine
# --------------------------------------------------------------------------------------

@dataclass(slots=True)
class GenomicRegion:
    chrom: str
    start: int
    end: int
    def length(self) -> int:
        return self.end - self.start
    def __str__(self) -> str:  # pragma: no cover - formatting
        return f"{self.chrom}:{self.start}-{self.end}"

class ZarrQueryEngine:
    def __init__(self, zarr_path: str | Path):
        
        if zarr is None:
            raise RuntimeError("zarr not installed")
        self.zarr_path = Path(zarr_path)
        if not self.zarr_path.exists():
            raise FileNotFoundError(self.zarr_path)
        self.root = zarr.open(str(self.zarr_path), mode='r')
        self.data = self.root['data']
        self.contigs = list(self.root['contigs'][:])
        chrom_sizes = np.asarray(self.root['chrom_sizes'][:], dtype=np.int64)
        self.contig_id_map: dict[str,int] = {
            str(k): int(v) for k,v in self.root.attrs.get('contig_id_map', {name:i for i,name in enumerate(self.contigs)}).items()
        }
        self.id_to_name = [None] * len(self.contig_id_map)
        for name, idx in self.contig_id_map.items():
            if 0 <= idx < len(self.id_to_name):
                self.id_to_name[idx] = name
        self.chrom_sizes = chrom_sizes


    def _resolve(self, chrom: str | int, ignore_missing: bool = True) -> tuple[str,int]:
        if isinstance(chrom, int):
            if chrom < 0 or chrom >= len(self.id_to_name):
                raise KeyError(f"Chromosome id {chrom} out of range")
            name = self.id_to_name[chrom]
            return name, chrom
        if chrom not in self.contig_id_map:
            if ignore_missing:
                logger.debug(f"Chromosome '{chrom}' missing from Zarr store")
                return chrom, None
            raise KeyError(f"Chromosome '{chrom}' missing")
        return chrom, self.contig_id_map[chrom]

    def query_region(self, chrom: str | int, start: int, end: int, *, clamp: bool = True) -> np.ndarray:
        name, cid = self._resolve(chrom)
        if cid is None:
            return np.empty(0, dtype=np.float32)

        length = int(self.chrom_sizes[cid])
        ds_name = f"c{cid}" if f"c{cid}" in self.data else name
        if clamp:
            start = max(0, start)
            end = min(length, end)
        if end <= start:
            return np.empty(0, dtype=self.data[ds_name].dtype)
        return self.data[ds_name][start:end]

    def query_multiple_regions(self, regions: Sequence[GenomicRegion]) -> dict[str, np.ndarray]:
        out = {}
        for r in tqdm.tqdm(regions):
            out[str(r)] = self.query_region(r.chrom, r.start, r.end)
        return out

    def query_binned(self, region: GenomicRegion, n_bins: int) -> np.ndarray:
        arr = self.query_region(region.chrom, region.start, region.end)
        if arr.size == 0:
            return np.zeros(n_bins, dtype=np.float32)
        edges = np.linspace(0, arr.size, n_bins + 1, dtype=np.int32)
        sums = np.add.reduceat(arr, edges[:-1])
        counts = (edges[1:] - edges[:-1]).clip(min=1)
        return (sums / counts).astype(np.float32, copy=False)

    def query_multiple_regions_array(self, regions: Sequence[GenomicRegion], n_bins: int) -> np.ndarray:
        mat = np.zeros((len(regions), n_bins), dtype=np.float32)
        for i, r in tqdm.tqdm(enumerate(regions), total=len(regions)):
            mat[i] = self.query_binned(r, n_bins)
        return mat

# --------------------------------------------------------------------------------------
# Convenience function
# --------------------------------------------------------------------------------------

def bigwig_to_zarr(bigwig: str | Path, out: str | Path, **kwargs) -> dict:
    return BigWigToZarrConverter(bigwig, out, ZarrConverterConfig(**kwargs)).convert()
