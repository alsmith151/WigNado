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

__all__ = ["load_engine", "utils"]


def load_engine():
    import importlib

    return importlib.import_module("wignado.engine")
