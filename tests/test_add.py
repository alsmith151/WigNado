import pytest
from wignado.utils import add


def test_add_python_fallback():
    # Works regardless of Rust extension
    assert add(2, 3) == 5


def test_add_uses_rust_if_available():
    # If compiled extension is available, ensure it returns correct result.
    try:
        # rust extension will be importable as wignado._wignado when built/installed
        from wignado import _wignado as _rust
    except Exception:
        pytest.skip("Rust extension not built")

    assert _rust.add(10, 7) == 17
