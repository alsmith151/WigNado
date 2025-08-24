"""wignado package (src layout)

This package intentionally delays importing the heavy `engine` module. Use
`wignado.load_engine()` to access it. The Rust extension (if built) is
available as `wignado._wignado`.
"""
try:
    from . import _wignado as _rust
except Exception:  # pragma: no cover - extension optional
    _rust = None

from . import utils
try:  # optional
    from .zarrstore import (
        bigwig_to_zarr,
        BigWigToZarrConverter,
        ZarrQueryEngine,
        ZarrConverterConfig,
    )
    __all__ = [
        "load_engine",
        "utils",
        "bigwig_to_zarr",
        "BigWigToZarrConverter",
        "ZarrQueryEngine",
        "ZarrConverterConfig",
    ]
except Exception:  # pragma: no cover
    __all__ = ["load_engine", "utils"]


def load_engine():
    import importlib

    return importlib.import_module("wignado.engine")
