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

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import time

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, field_validator, validator
import tqdm
from enum import Enum
import pybigtools  # type: ignore
import zarr  # type: ignore
from numcodecs import Zstd  # type: ignore
import xarray as xr  # type: ignore

from .core import GenomicRegion

# --------------------------------------------------------------------------------------
# Configuration dataclass
# --------------------------------------------------------------------------------------


class ValueDType(str, Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"


class ZarrConverterConfig(BaseModel):
    overwrite: bool = Field(
        False, description="Overwrite existing Zarr store if present"
    )
    chunk_size: int = Field(
        1_000_000, ge=1, description="Per-chromosome chunk length in bp"
    )
    value_dtype: ValueDType = Field(
        ValueDType.FLOAT32, description="Data type for stored signal values"
    )
    compression_level: int = Field(
        5, ge=1, le=22, description="Zstd compression level (1-22)"
    )
    min_chrom_length: int = Field(
        1000, ge=0, description="Skip contigs shorter than this length (bp)"
    )
    zero_fill_value: float = Field(
        0.0, description="Baseline value used for missing / invalid entries"
    )
    batch_bp: int = Field(
        5_000_000, ge=1, description="Streaming window size (bp) for reading BigWig"
    )
    include_mapping: bool = Field(
        True,
        description="Store contig name -> id mapping for categorical representation",
    )


# --------------------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------------------


class BigWigToZarrConverter:
    def __init__(
        self,
        bigwig_path: str | Path,
        zarr_path: str | Path,
        config: ZarrConverterConfig | None = None,
    ):
        self.bigwig_path = Path(bigwig_path)
        self.zarr_path = Path(zarr_path)
        self.cfg = config or ZarrConverterConfig()
        self._validate()

    def _validate(self) -> None:
        if not self.bigwig_path.exists():
            raise FileNotFoundError(f"BigWig file not found: {self.bigwig_path}")
        with pybigtools.open(str(self.bigwig_path)) as bw:  # sanity
            _ = bw.chroms()
        if self.zarr_path.exists() and self.cfg.overwrite:
            import shutil

            shutil.rmtree(self.zarr_path)
        self.zarr_path.parent.mkdir(parents=True, exist_ok=True)

    def _extract_contigs(self) -> list[tuple[str, int]]:
        with pybigtools.open(str(self.bigwig_path)) as bw:
            chroms_dict = bw.chroms()
            chrom_meta = [
                (c, int(length_val))
                for c, length_val in chroms_dict.items()
                if length_val >= self.cfg.min_chrom_length
            ]

        if not chrom_meta:
            raise ValueError("No chromosomes meet minimum length criterion")

        return chrom_meta

    # Public API
    def convert(self) -> dict:
        """Convert BigWig to a Zarr hierarchy.

        Steps:
          1. Extract contig metadata (filtering by min length)
          2. Create store + root group and write metadata
          3. Create per-contig arrays (optionally categorical id names)
          4. Stream values from BigWig into chunked arrays
          5. Persist performance / accounting metadata & return summary
        """
        start = time.perf_counter()
        chrom_meta = self._extract_contigs()

        root = self._create_root_group()
        contig_id_map, total_bp = self._write_metadata_arrays(root, chrom_meta)
        g_data, value_dtype = self._create_data_arrays(root, chrom_meta, contig_id_map)
        written_values = self._stream_fill(root, g_data, chrom_meta, value_dtype)

        elapsed = time.perf_counter() - start
        root.attrs["conversion_seconds"] = elapsed
        root.attrs["total_basepairs"] = total_bp
        root.attrs["values_written"] = written_values
        logger.info(
            f"Completed conversion in {elapsed:.1f}s; {total_bp / 1e6:.2f} Mbp processed"
        )
        return {
            "chromosomes": len(chrom_meta),
            "total_bp": total_bp,
            "values_written": written_values,
            "seconds": elapsed,
            "zarr_path": str(self.zarr_path),
        }

    # ---- helpers -----------------------------------------------------------------
    def _create_root_group(self):
        store = zarr.storage.LocalStore(str(self.zarr_path))
        root = zarr.create_group(store=store, path="/", overwrite=self.cfg.overwrite)
        root.attrs.update(
            {
                "format": "wignado.zarr.v2",
                "bigwig_source": str(self.bigwig_path),
                "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "config": self.cfg.model_dump_json(),
            }
        )
        return root

    def _write_metadata_arrays(self, root, chrom_meta) -> tuple[dict[str, int], int]:
        """
        Write metadata arrays to the Zarr root group.
        """
        contigs = [c for c, _ in chrom_meta]
        lengths = np.array(
            [length_val for _, length_val in chrom_meta], dtype=np.uint64
        )
        root.create_array("contigs", data=np.array(contigs, dtype=str), overwrite=True)
        root.create_array("chrom_sizes", data=lengths, overwrite=True)
        contig_id_map = (
            {name: i for i, name in enumerate(contigs)}
            if self.cfg.include_mapping
            else {}
        )
        if contig_id_map:
            root.attrs["contig_id_map"] = contig_id_map
        total_bp = int(lengths.sum())
        return contig_id_map, total_bp

    def _create_data_arrays(
        self, root, chrom_meta, contig_id_map
    ) -> tuple["zarr.hierarchy.Group", ValueDType]:
        """
        Create data arrays for each chromosome in the Zarr root group.
        """
        g_data = root.require_group("data")
        value_dtype = self.cfg.value_dtype
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
                compressor=Zstd(level=self.cfg.compression_level),
                overwrite=True,
                fill_value=self.cfg.zero_fill_value,
            )
        return g_data, value_dtype

    def _stream_fill(self, root, g_data, chrom_meta, value_dtype):
        """Stream BigWig signal into the per-contig Zarr arrays.

        Parameters
        ----------
        root : zarr.Group
            Root group (used to access contig id mapping attrs).
        g_data : zarr.Group
            Group containing the per-contig target arrays.
        chrom_meta : list[(str, int)]
            Sequence of (chrom_name, chrom_length) tuples to ingest.
        value_dtype : np.dtype / str
            Target dtype (already validated from config).

        Returns
        -------
        int
            Total number of base positions written (sums region spans actually filled).

        Notes
        -----
        * Reads are performed in fixed-size windows of size ``batch_bp`` to bound memory.
        * The BigWig file handle is opened inside the inner loop so that any underlying
          resources are released promptly; if profiling shows this open/close overhead
          is significant, a future optimization could keep one handle per chromosome.
        * Non-finite values (NaN, inf) are replaced with ``zero_fill_value``.
        * Empty responses (no coverage) are skipped, leaving the pre-initialized
          fill value already present in the array.
        """
        written_values = 0
        # Iterate each chromosome / contig
        for chrom, length in tqdm.tqdm(chrom_meta, total=len(chrom_meta)):
            # Resolve dataset name (categorical id mapping if present)
            chrom_id = root.attrs.get("contig_id_map", {}).get(chrom, chrom)
            ds_name = f"c{chrom_id}" if isinstance(chrom_id, int) else chrom
            arr = g_data[ds_name]
            # Walk along chromosome in streaming windows
            for start_pos in range(0, length, self.cfg.batch_bp):
                end_pos = min(start_pos + self.cfg.batch_bp, length)
                # Fetch raw values for this span (exclusive end)
                with pybigtools.open(str(self.bigwig_path)) as bw:
                    vals = bw.values(chrom, start_pos, end_pos)
                # Skip empty / missing blocks (array already has fill value)
                if vals is None or len(vals) == 0:
                    continue
                # Ensure correct dtype without extra copy when already matching
                if vals.dtype != value_dtype:
                    vals = vals.astype(value_dtype, copy=False)
                # Replace any non-finite entries with configured baseline
                invalid_mask = ~np.isfinite(vals)
                if invalid_mask.any():
                    vals[invalid_mask] = self.cfg.zero_fill_value
                # Write slice directly (Zarr handles chunking & compression)
                arr[start_pos:end_pos] = vals
                written_values += end_pos - start_pos
        return written_values


# --------------------------------------------------------------------------------------
# Query Engine
# --------------------------------------------------------------------------------------


class ZarrQueryEngine:
    """Query engine for a WigNado Zarr store.

    Provides:
      * Exact region extraction (optionally clamped to contig bounds)
      * Mean-binned region summaries
      * Multi-region querying (serial & chunk-aware parallel)

    Notes
    -----
    This class assumes the Zarr layout produced by :class:`BigWigToZarrConverter`.
    Public method signatures are kept minimal; internal helpers concentrate
    repeated logic (dataset name resolution, bounds clamping, etc.).
    """

    # ---------------------------- construction ---------------------------------
    def __init__(self, zarr_path: str | Path, region_config: RegionConfig):
        # Validate zarr exists
        self.zarr_path = Path(zarr_path)
        if not self.zarr_path.exists():
            raise FileNotFoundError(self.zarr_path)

        # Config for adjusting regions as they are processed
        self.region_config = region_config

        # Load Zarr store & metadata
        self.root = zarr.open(str(self.zarr_path), mode="r")
        self.data = self.root["data"]
        self.contigs = list(self.root["contigs"][:])
        chrom_sizes = np.asarray(self.root["chrom_sizes"][:], dtype=np.int64)
        self.contig_id_map: dict[str, int] = {
            str(k): int(v)
            for k, v in self.root.attrs.get(
                "contig_id_map", {name: i for i, name in enumerate(self.contigs)}
            ).items()
        }
        # Dense id->name list for fast integer to string resolve
        self.id_to_name = [None] * len(self.contig_id_map)
        for name, idx in self.contig_id_map.items():
            if 0 <= idx < len(self.id_to_name):
                self.id_to_name[idx] = name
        self.chrom_sizes = chrom_sizes

    # --------------------------- internal helpers ------------------------------
    def _resolve(
        self, chrom: str | int, ignore_missing: bool = True
    ) -> tuple[str, int | None]:
        """Return (name, id) for a chromosome or (name, None) if missing & allowed."""
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

    def _dataset_name(self, cid: int, name: str) -> str:
        coded = f"c{cid}"
        return coded if coded in self.data else name

    def _clamp_interval(self, cid: int, start: int, end: int) -> tuple[int, int, int]:
        length = int(self.chrom_sizes[cid])
        start_c = max(0, start)
        end_c = min(length, end)
        return start_c, end_c, length

    def _get_array(self, chrom: str | int):
        name, cid = self._resolve(chrom)
        if cid is None:
            return None, None, None
        ds_name = self._dataset_name(cid, name)
        return self.data[ds_name], cid, ds_name

    # ------------------------------------------------------------------
    # Chunk-aware parallel querying
    # ------------------------------------------------------------------
    def _dataset_and_id(self, chrom: str | int):
        name, cid = self._resolve(chrom)
        ds_name = f"c{cid}" if f"c{cid}" in self.data else name
        return self.data[ds_name], cid, ds_name

    def _split_region_into_chunks(self, region: GenomicRegion):
        arr, cid, ds_name = self._dataset_and_id(region.chrom)
        chunk_len = arr.chunks[0] if hasattr(arr, "chunks") else arr.shape[0]
        start = max(0, region.start)
        end = min(int(self.chrom_sizes[cid]), region.end)
        if end <= start:
            return []
        first_chunk = start // chunk_len
        last_chunk = (end - 1) // chunk_len
        pieces = []
        for chunk_idx in range(first_chunk, last_chunk + 1):
            c_start = chunk_idx * chunk_len
            c_end = c_start + chunk_len
            s = max(start, c_start)
            e = min(end, c_end)
            if e > s:
                pieces.append(
                    {
                        "cid": cid,
                        "ds": ds_name,
                        "chunk_idx": chunk_idx,
                        "chunk_start": c_start,
                        "start": s,
                        "end": e,
                        "region": region,
                    }
                )
        return pieces

    def _plan_regions_by_chunk(self, regions: Sequence[GenomicRegion]):
        """Return a plan grouping region sub-spans by (dataset, chunk_index).

        Each region spanning multiple chunks is split so that we only read
        each chunk once per worker.
        """
        plan: dict[tuple[str, int], list[dict]] = {}
        for r in regions:
            for piece in self._split_region_into_chunks(r):
                key = (piece["ds"], piece["chunk_idx"])
                plan.setdefault(key, []).append(piece)
        return plan

    def _result_to_xarray(
        self,
        regions: Sequence[GenomicRegion],
        result: dict[str, np.ndarray],
        pad_value: float,
    ) -> "xr.Dataset":
        region_ids = [str(r) for r in regions]
        lengths = np.array(
            [len(result.get(rid, [])) for rid in region_ids], dtype=np.int32
        )
        max_len = int(lengths.max(initial=0))
        data = np.full((len(region_ids), max_len), pad_value, dtype=np.float32)
        for i, rid in enumerate(region_ids):
            arr_vals = result.get(rid, np.empty(0, dtype=np.float32))
            if arr_vals.size:
                data[i, : arr_vals.size] = arr_vals.astype(np.float32, copy=False)
        ds = xr.Dataset(
            {
                "signal": (("region", "position"), data),
                "length": (("region",), lengths),
            },
            coords={
                "region": region_ids,
                "position": np.arange(max_len, dtype=np.int32),
            },
            attrs={"source": "wignado.zarrstore", "pad_value": float(pad_value)},
        )
        return ds  # type: ignore[return-value]

    # --------------------------- query API ------------------------------------
    def _query_region(
        self, chrom: str | int, start: int, end: int, *, clamp: bool = True
    ) -> np.ndarray:
        arr, cid, ds_name = self._get_array(chrom)
        if cid is None:
            return np.empty(0, dtype=np.float32)
        if clamp:
            start, end, _ = self._clamp_interval(cid, start, end)
        if end <= start:
            return np.empty(0, dtype=arr.dtype)
        return arr[start:end]

    def _query_multiple_regions(
        self, regions: Sequence[GenomicRegion]
    ) -> dict[str, np.ndarray]:
        out = {}
        for r in tqdm.tqdm(regions):
            out[str(r)] = self._query_region(r.chrom, r.start, r.end)
        return out

    def _query_binned(self, region: GenomicRegion, n_bins: int) -> np.ndarray:
        arr = self._query_region(region.chrom, region.start, region.end)
        if arr.size == 0:
            return np.zeros(n_bins, dtype=np.float32)
        edges = np.linspace(0, arr.size, n_bins + 1, dtype=np.int32)
        sums = np.add.reduceat(arr, edges[:-1])
        counts = (edges[1:] - edges[:-1]).clip(min=1)
        return (sums / counts).astype(np.float32, copy=False)

    def _query_multiple_regions_array(
        self, regions: Sequence[GenomicRegion], n_bins: int
    ) -> np.ndarray:
        mat = np.zeros((len(regions), n_bins), dtype=np.float32)
        for i, r in tqdm.tqdm(enumerate(regions), total=len(regions)):
            mat[i] = self._query_binned(r, n_bins)
        return mat

    def _query_multiple_regions_parallel(
        self,
        regions: Sequence[GenomicRegion],
        max_workers: int = 4,
        prefer_distinct_chunks: bool = True,
        as_xarray: bool = False,
        pad_value: float = np.nan,
    ) -> dict[str, np.ndarray]:
        """Parallel region querying minimizing overlapping chunk reads.

        Strategy:
          1. Split regions into per-chunk pieces.
          2. Group pieces by chunk.
          3. Schedule groups across a thread pool so workers mostly handle
             disjoint chunks (round-robin) to reduce simultaneous access to
             same underlying storage.

                Returns:
                        dict(region_str -> np.ndarray) by default.
                        If as_xarray=True (and xarray installed), returns an xarray.Dataset
                        with variable-length regions padded to 'pad_value'. Dimensions:
                            - region: region index (with region string coordinate labels)
                            - position: 0..(max_region_length-1) offset within region
                        Dataset variables:
                            - signal: (region, position) float32
                            - length: (region,) original lengths
        """
        if not regions:
            return {}
        from concurrent.futures import ThreadPoolExecutor, as_completed

        plan = self._plan_regions_by_chunk(regions)
        # Pre-allocate containers for region assembly
        region_parts: dict[str, list[tuple[int, np.ndarray]]] = {}

        # Ordering of chunk groups: sort by dataset then chunk index
        chunk_groups = sorted(plan.items(), key=lambda kv: (kv[0][0], kv[0][1]))
        if prefer_distinct_chunks:
            # Interleave groups to spread out adjacent chunk indices
            even = chunk_groups[0::2]
            odd = chunk_groups[1::2]
            chunk_groups = even + odd

        def process_group(key, pieces):
            ds_name, chunk_idx = key
            arr = self.data[ds_name]
            chunk_len = arr.chunks[0] if hasattr(arr, "chunks") else arr.shape[0]
            c_start = chunk_idx * chunk_len
            c_end = min(c_start + chunk_len, arr.shape[0])
            # Load chunk slice once
            chunk_view = arr[c_start:c_end]
            out_local = []
            for p in pieces:
                local_s = p["start"] - c_start
                local_e = p["end"] - c_start
                sub = chunk_view[local_s:local_e]
                out_local.append((p["region"], p["start"], sub))
            return out_local

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(process_group, key, pieces) for key, pieces in chunk_groups
            ]
            for fut in as_completed(futures):
                for region_obj, abs_start, sub_vals in fut.result():
                    rid = str(region_obj)
                    region_parts.setdefault(rid, []).append((abs_start, sub_vals))

        # Assemble full arrays per region (pieces are non-overlapping)
        result: dict[str, np.ndarray] = {}
        for r in regions:
            rid = str(r)
            if rid not in region_parts:
                result[rid] = np.empty(0, dtype=np.float32)
                continue
            pieces = sorted(region_parts[rid], key=lambda t: t[0])
            vals = (
                np.concatenate([p[1] for p in pieces])
                if len(pieces) > 1
                else pieces[0][1]
            )
            # Ensure float32 output
            result[rid] = vals.astype(np.float32, copy=False)
        if not as_xarray:
            return result
        else:
            return self._result_to_xarray(regions, result, pad_value)

    def query_region(
        self, region: GenomicRegion, *, n_bins: int | None = None
    ) -> np.ndarray:
        if n_bins is not None:
            return self._query_binned(region, n_bins)
        return self._query_region(region.chrom, region.start, region.end)

    def query_regions(
        self,
        regions: Sequence[GenomicRegion],
        *,
        n_threads: int | None = None,
    ) -> dict[str, np.ndarray]:
        pass


class ReferencePoint(Enum):
    START = "start"
    CENTER = "center"
    END = "end"
    SCALE = "scale"


class RegionConfig(BaseModel):
    bin_size: int | None = None
    n_bins: int | None = None
    reference_point: ReferencePoint = ReferencePoint.CENTER
    bp_before: int = Field(0, ge=0)
    bp_after: int = Field(0, ge=0)

    # Validate that either bin_size or n_bins is set
    @field_validator("bin_size", "n_bins", pre=True, always=True)
    def validate_bin_size_or_n_bins(cls, v, values, field):
        if field.name == "bin_size" and v is None and values.get("n_bins") is None:
            raise ValueError("Either bin_size or n_bins must be set.")
        if field.name == "n_bins" and v is None and values.get("bin_size") is None:
            raise ValueError("Either bin_size or n_bins must be set.")
        return v
