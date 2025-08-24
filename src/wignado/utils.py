"""Small utility helpers for wignado used at runtime without heavy deps."""
from __future__ import annotations

from typing import Any


def add(a: int, b: int) -> int:
    """Simple deterministic addition helper used in tests; prefer using the
    Rust implementation when available as `wignado._wignado.add`.
    """
    return int(a) + int(b)


def safe_div(a: int | float, b: int | float, default: Any = 0) -> float | Any:
    try:
        return a / b
    except Exception:
        return default
