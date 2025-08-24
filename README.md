# WigNado

High-performance mixed Rust/Python genomic toolkit. Currently provides:

* BigWig -> TileDB sparse conversion + fast region queries (see `wignado.tiledb`)
* BigWig -> Zarr dense-per-chrom conversion + region / binned queries (see `wignado.zarrstore`)

## Installation (development)

Build the extension and install Python package (editable):

```bash
pip install -e .[tiledb,zarr]
```

## BigWig → Zarr Example

```python
from wignado.zarrstore import bigwig_to_zarr, ZarrQueryEngine, GenomicRegion

# Convert (creates directory 'example.zarr')
stats = bigwig_to_zarr("example.bigWig", "example.zarr", overwrite=True, chunk_size=2_000_000)
print(stats)

# Query
qe = ZarrQueryEngine("example.zarr")
region = GenomicRegion(chrom="chr1", start=1_000_000, end=1_100_000)
values = qe.query_region(region.chrom, region.start, region.end)
print(values.mean())

# Binned matrix across multiple regions
regions = [GenomicRegion("chr1", 0, 200_000), GenomicRegion("chr1", 500_000, 700_000)]
mat = qe.query_multiple_regions_array(regions, n_bins=50)
print(mat.shape)
```

## BigWig → TileDB (brief)

```python
from wignado.tiledb import ConverterConfig, BigwigToTileDBConverter, TileDBQueryEngine, GenomicRegion

cfg = ConverterConfig(overwrite=True, tile_size=10_000)
converter = BigwigToTileDBConverter("example.bigWig", "example_tiledb", config=cfg)
metrics = converter.convert()
print(metrics)

qe = TileDBQueryEngine("example_tiledb")
vals = qe.query_region("chr1", 1_000_000, 1_010_000)
print(vals.mean())
```

## License

MIT (adjust as appropriate).
