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
from typing import Literal
import time

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, model_validator, computed_field
import tqdm
from enum import Enum
import pybigtools  # type: ignore
import zarr  # type: ignore
from numcodecs import Zstd  # type: ignore

from .core import GenomicRegion, RegionConfig, QueryConfig, ReferencePoint

# --------------------------------------------------------------------------------------
# Configuration dataclass
# --------------------------------------------------------------------------------------


class ValueDType(str, Enum):
    """Enum of supported on-disk value dtypes.

    Keeping as string Enum so values map cleanly to numpy dtype names.
    """
    FLOAT32 = "float32"
    FLOAT16 = "float16"


class ZarrConverterConfig(BaseModel):
    """Configuration controlling BigWig -> Zarr conversion.

    Parameters govern chunk sizing, compression, dtype, and filtering of
    contigs. All values are validated by Pydantic. This object is serialised
    verbatim into the Zarr root attrs (key: ``config``) for provenance.
    """
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
    """Stream a BigWig into a dense-per-contig Zarr hierarchy.

    Typical usage:

        >>> cfg = ZarrConverterConfig(chunk_size=500_000)
        >>> BigWigToZarrConverter("signal.bw", "out.zarr", cfg).convert()

    The resulting layout under ``out.zarr`` is:

        contigs        (1D, str)
        chrom_sizes    (1D, uint64)
        data/<contig or cID>  (1D float array per chromosome)

    Attributes
    ----------
    bigwig_path : Path
        Source BigWig file.
    zarr_path : Path
        Destination directory (created if missing / overwritten if configured).
    cfg : ZarrConverterConfig
        Active conversion configuration.
    """
    def __init__(
        self,
        bigwig_path: str | Path,
        zarr_path: str | Path,
        config: ZarrConverterConfig | None = None,
    ):
        self.bigwig_path = Path(bigwig_path)
        self.zarr_path = Path(zarr_path)
        self.cfg = config or ZarrConverterConfig()
        self._validate_inputs()

        self.compressor = zarr.codecs.BloscCodec(cname='zstd', clevel=self.cfg.compression_level, shuffle=zarr.codecs.BloscShuffle.bitshuffle)

    def _validate_inputs(self) -> None:
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
                compressors=self.compressor,
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
    """Read/query interface for a WigNado Zarr store."""

    # ---------------------------- construction ---------------------------------
    def __init__(
        self,
        zarr_path: str | Path,
        region_config: RegionConfig,
        query_config: QueryConfig,
    ):
        # Validate zarr exists
        self.zarr_path = Path(zarr_path)
        if not self.zarr_path.exists():
            raise FileNotFoundError(self.zarr_path)

        # Config for adjusting regions as they are processed
        self.region_config = region_config

        # Config for query behavior
        self.query_config = query_config

        # Load Zarr store & metadata
        self.root = zarr.open(str(self.zarr_path), mode="r")
        self.data = self.root["data"]

        # Load contig metadata
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

    # ------------------------------------------------------------------
    # Basic metadata helpers
    # ------------------------------------------------------------------
    def _resolve_chromosome(self, chrom: str | int) -> tuple[str, int | None]:
        """Resolve a chromosome identifier.

        Parameters
        ----------
        chrom : str | int
            Chromosome name or numeric contig id.

        Returns
        -------
        (name, id | None)
            Returns the normalized name and its integer id. If the chromosome
            is missing and missing chromosomes are tolerated, returns (name, None).
        """
        if isinstance(chrom, int):
            # Resolve integer chromosome ID
            if chrom < 0 or chrom >= len(self.id_to_name):
                raise KeyError(f"Chromosome id {chrom} out of range")

            name = self.id_to_name[chrom]
            # Return the resolved chromosome name and ID
            return name, chrom

        if chrom not in self.contig_id_map:
            if self.query_config.error_on_missing_chromosome:
                raise KeyError(f"Chromosome '{chrom}' missing from Zarr store")
            return chrom, None

        # Return the resolved chromosome name and ID
        return chrom, self.contig_id_map[chrom]

    def get_chromosome_length(self, chrom: str | int) -> int | None:
        """Return chrom length in bp or None if missing (and missing allowed)."""
        _, chrom_id = self._resolve_chromosome(chrom)
        if chrom_id is None:
            return None
        return int(self.chrom_sizes[chrom_id])

    def _expand_region(self, region: GenomicRegion) -> GenomicRegion | None:
        """Transform an input region according to RegionConfig.

        For reference-point mode the window is centered / anchored relative to
        the supplied region (interpreted as the raw genomic interval). For
        scale-regions mode we extend outward by flanks + internal unscaled
        segments producing a larger window from which we will later segment
        and (optionally) scale the original body.
        """
        chrom_name, chrom_id = self._resolve_chromosome(region.chrom)
        if chrom_id is None:
            if self.query_config.error_on_missing_chromosome:
                raise KeyError(f"Chromosome '{region.chrom}' missing from Zarr store")
            return None
        if self.region_config.mode == "reference-point":
            if self.region_config.reference_point == ReferencePoint.START:
                anchor = region.start
            elif self.region_config.reference_point == ReferencePoint.END:
                anchor = region.end
            else:  # center / default
                anchor = (region.start + region.end) // 2
            
            new_start = anchor - self.region_config.bp_before
            new_end = anchor + self.region_config.bp_after
            
            if new_end < new_start:
                new_end = new_start + 1

            return GenomicRegion(chrom=chrom_name, start=new_start, end=new_end, name=region.name)

        # scale-regions: build outer window spanning flanks + body + internal unscaled segments
        new_start = region.start - (
            self.region_config.upstream + self.region_config.unscaled_5_prime
        )
        new_end = region.end + (
            self.region_config.downstream + self.region_config.unscaled_3_prime
        )
        if new_end < new_start:
            new_end = new_start + 1
        return GenomicRegion(chrom=chrom_name, start=new_start, end=new_end, name=region.name)

    # ------------------------------------------------------------------
    # Low-level data access
    # ------------------------------------------------------------------
    def _dataset_name(self, chrom_id: int | None, chrom_name: str) -> str:
        """Return concrete dataset name for a contig (id-coded if available)."""
        if chrom_id is None:
            return chrom_name
        coded = f"c{chrom_id}"
        return coded if coded in self.data else chrom_name

    def _fetch_region(
        self, region: GenomicRegion, *, expand: bool = True
    ) -> tuple[np.ndarray, GenomicRegion | None]:
        """Return (array, effective_region).

        Parameters
        ----------
        region : GenomicRegion
            User-supplied region or already-expanded region if expand=False.
        expand : bool, default True
            Whether to apply configured expansion / transformation first.

        Notes
        -----
        The returned array is always a view (copy-on-write) of the stored
        per-chromosome dense array (float32). Coordinates are clamped to
        chromosome bounds. Missing chromosomes yield (empty_array, None).
        """
        effective = self._expand_region(region) if expand else region
        if effective is None:
            return np.empty(0, dtype=np.float32), None
        chrom_name, chrom_id = self._resolve_chromosome(effective.chrom)
        ds_name = self._dataset_name(chrom_id, chrom_name)
        chrom_length = int(self.chrom_sizes[chrom_id])  # type: ignore[index]
        start = max(0, effective.start)
        end = min(effective.end, chrom_length)
        if end <= start:
            return np.empty(0, dtype=np.float32), effective
        arr = self.data[ds_name]
        return arr[start:end], effective

    def _query_region(self, region: GenomicRegion, expand: bool = False) -> np.ndarray:
        """Public-compatible private method: return expanded region values.

        Provided for backward-compatibility with earlier internal usage where
        "query" implied applying RegionConfig. Prefer using ``_fetch_region``
        when you need both values and the effective coordinates.
        """
        arr, _ = self._fetch_region(region, expand=False)
        return arr

    # --------------------------- binning helpers -----------------------------
    def _mean_bins_from_edges(self, arr: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Return mean of slices defined by half-open bin edges.

        Assumes ``edges`` are monotonically non-decreasing. Last edge is clamped
        to array length. Empty bins yield NaN (later caller may keep or replace).
        """
        if arr.size == 0 or edges.size <= 1:
            return np.zeros(max(edges.size - 1, 0), dtype=np.float32)
        e = edges.astype(np.int64)
        e[-1] = min(e[-1], arr.size)
        if np.any(e[1:] < e[:-1]):
            raise ValueError("Non-monotonic bin edges")
        starts = e[:-1]
        ends = e[1:]
        # Compute means (loop acceptable; bin counts typically modest & explicit keeps clarity)
        out = np.empty(len(starts), dtype=np.float32)
        for i, (s, end) in enumerate(zip(starts, ends)):
            if end > s:
                out[i] = float(np.mean(arr[s:end]))
            else:
                out[i] = np.nan
        return out

    # Unified helper -------------------------------------------------------
    def _edges_for_fixed(self, length: int, *, n_bins: int | None = None, bin_size: int | None = None) -> np.ndarray:
        if n_bins is not None:
            return np.linspace(0, length, n_bins + 1)
        if bin_size and bin_size > 0:
            n = int(np.ceil(length / bin_size))
            return np.arange(0, n * bin_size + 1, bin_size)
        return np.array([0, length], dtype=np.int64)

    def _bin_fixed_window(self, arr: np.ndarray) -> np.ndarray:
        rc = self.region_config
        edges = self._edges_for_fixed(arr.size, n_bins=rc.n_bins, bin_size=rc.bin_size)
        if edges.size == 2 and edges[1] - edges[0] == arr.size and rc.n_bins is None and (not rc.bin_size or rc.bin_size == 0):
            return arr.astype(np.float32, copy=False)
        edges[-1] = max(edges[-1], arr.size)  # clamp
        return self._mean_bins_from_edges(arr, edges)

    def _bin_scale_regions(self, region: GenomicRegion, arr: np.ndarray, expanded: GenomicRegion) -> np.ndarray:
        rc = self.region_config
        bin_size = rc.bin_size
        if bin_size <= 0:
            raise ValueError("bin_size must be >0 in scale-regions mode for binning")

        # --- Segment index calculation (expanded coordinates) ---
        orig_body_len = max(0, region.end - region.start)
        upstream = rc.upstream
        un5 = rc.unscaled_5_prime
        un3 = rc.unscaled_3_prime
        downstream = rc.downstream

        idx_up_end = upstream
        idx_un5_end = idx_up_end + un5
        idx_body_end = idx_un5_end + orig_body_len
        idx_un3_end = idx_body_end + un3
        idx_down_end = idx_un3_end + downstream

        def slice_safe(a: int, b: int) -> np.ndarray:
            return arr[max(0, a): min(len(arr), b)]

        upstream_arr = slice_safe(0, idx_up_end)
        un5_arr = slice_safe(idx_up_end, idx_un5_end)
        body_arr = slice_safe(idx_un5_end, idx_body_end)
        un3_arr = slice_safe(idx_body_end, idx_un3_end)
        downstream_arr = slice_safe(idx_un3_end, idx_down_end)

        # Helper for simple fixed-size binning of a segment
        def bin_segment(seg: np.ndarray) -> np.ndarray:
            if seg.size == 0:
                return np.zeros(0, dtype=np.float32)
            edges = self._edges_for_fixed(seg.size, bin_size=bin_size)
            edges[-1] = max(edges[-1], seg.size)
            return self._mean_bins_from_edges(seg, edges)

        # Body scaling: proportional mapping into target number of bins
        body_target = rc.body
        if body_target % bin_size != 0:
            raise ValueError("body must be multiple of bin_size for scale-regions binning")
        target_bins = body_target // bin_size
        if target_bins > 0:
            if body_arr.size == 0:
                body_binned = np.zeros(target_bins, dtype=np.float32)
            else:
                # Map original body to target bins via linspace of edges
                edges = np.linspace(0, body_arr.size, target_bins + 1)
                body_binned = self._mean_bins_from_edges(body_arr, edges)
        else:
            body_binned = np.zeros(0, dtype=np.float32)

        return np.concatenate([
            bin_segment(upstream_arr),
            bin_segment(un5_arr),
            body_binned,
            bin_segment(un3_arr),
            bin_segment(downstream_arr),
        ])

    # --------------------------- public binning API ---------------------------
    def query_region(self, region: GenomicRegion, binned: bool = False) -> np.ndarray:
        """Return raw or binned signal for a single region.

        Parameters
        ----------
        region : GenomicRegion
            Input genomic interval (unexpanded coordinates).
        binned : bool, default False
            If True apply binning logic according to RegionConfig.

        Notes
        -----
        For scale-regions mode the original region defines the "body" that is
        proportionally scaled; we therefore expand once to obtain the full
        window and must avoid a second expansion (a previous implementation
        performed an inadvertent double expansion). This method now uses
        ``_fetch_region(expand=True)`` once and, for scale-regions, separately
        obtains the expanded coordinates to segment the array.
        """
        if not binned:
            return self._query_region(region, expand=False)

        if self.region_config.mode == 'reference-point':
            raw, _ = self._fetch_region(region, expand=True)
            return self._bin_fixed_window(raw)

        # scale-regions: expand exactly once, fetch expanded raw without re-expanding
        expanded = self._expand_region(region)
        if expanded is None:
            return np.empty(0, dtype=np.float32)
        expanded_raw, _ = self._fetch_region(expanded, expand=False)
        return self._bin_scale_regions(region, expanded_raw, expanded)

    # --------------------------- multi-region API ---------------------------
    def query_regions(
        self,
        regions: list[GenomicRegion] | tuple[GenomicRegion, ...],
        *,
        binned: bool = False,
        stack: bool = False,
        pad_value: float = np.nan,
        show_progress: bool = False,
        as_xarray: bool = False,
    ):
        """Query multiple regions.

        Parameters
        ----------
        regions : sequence[GenomicRegion]
            Regions to extract (interpreted as unexpanded coordinates).
        binned : bool, default False
            Apply binning per RegionConfig.
        stack : bool, default False
            If True, return a 2D ndarray (n_regions, width) when widths are
            uniform (binned=True) or a padded matrix (raw) using `pad_value`.
        pad_value : float, default NaN
            Fill value for padding when stacking variable length raw regions.
        show_progress : bool, default False
            Display tqdm progress bar.
        as_xarray : bool, default False
            If True and xarray is installed, return an xarray Dataset with
            dimensions (region, position) or (region, bin).

        Returns
        -------
        dict | np.ndarray | xr.Dataset
            By default a dict mapping region string -> np.ndarray. If stack=True
            returns (matrix, region_ids, lengths) unless as_xarray=True.
        """
        if not regions:
            return {} if not stack else (np.zeros((0, 0), dtype=np.float32), [], [])

        iterator = regions
        if show_progress:
            iterator = tqdm.tqdm(regions, desc="regions")  # type: ignore

        out: dict[str, np.ndarray] = {}
        for r in iterator:  # type: ignore
            try:
                arr = self.query_region(r, binned=binned)
            except KeyError:
                if self.query_config.error_on_missing_chromosome:
                    raise
                arr = np.empty(0, dtype=np.float32)
            out[str(r)] = arr.astype(np.float32, copy=False)

        if not stack and not as_xarray:
            return out

        # Determine uniform length expectation
        region_ids = list(out.keys())
        arrays = [out[rid] for rid in region_ids]
        lengths = np.array([a.size for a in arrays], dtype=np.int32)
        uniform = np.all(lengths == lengths[0]) if lengths.size else True

        if binned and not uniform:
            # Defensive: binned outputs should be uniform; raise to warn user.
            raise ValueError("Binned outputs have differing lengths; check config")

        if uniform:
            mat = (
                np.vstack([a for a in arrays])
                if lengths.size and lengths[0] > 0
                else np.zeros((len(arrays), 0), dtype=np.float32)
            )
        else:
            max_len = int(lengths.max(initial=0))
            mat = np.full((len(arrays), max_len), pad_value, dtype=np.float32)
            for i, a in enumerate(arrays):
                if a.size:
                    mat[i, : a.size] = a

        if as_xarray:
            try:
                import xarray as xr  # type: ignore
            except Exception:  # pragma: no cover
                raise ImportError("xarray not installed; install to use as_xarray=True")
            dims = ("region", "bin" if binned else "position")
            ds = xr.Dataset(
                {
                    "signal": (dims, mat),
                    "length": (("region",), lengths),
                },
                coords={
                    "region": region_ids,
                    dims[1]: np.arange(mat.shape[1], dtype=np.int32),
                },
                attrs={
                    "source": "wignado.zarrstore",
                    "binned": binned,
                    "pad_value": float(pad_value),
                },
            )
            return ds

        return (mat, region_ids, lengths) if stack else out

