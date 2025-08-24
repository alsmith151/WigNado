import importlib
import importlib.util
import sys
import types
import numpy as np


def _ensure_stub(module_name: str, attrs: dict):
    """If a module is missing, insert a minimal stub into sys.modules."""
    if importlib.util.find_spec(module_name) is None:
        mod = types.ModuleType(module_name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[module_name] = mod


def import_engine_with_stubs():
    # Provide minimal stubs for heavy/external deps so importing engine is reliable in CI
    _ensure_stub(
        "numba",
        {
            "jit": (lambda *a, **k: (lambda f: f)),
            "prange": (lambda n: range(n)),
        },
    )

    def _base_init(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    _ensure_stub(
        "pydantic",
        {
            "BaseModel": type("BaseModel", (object,), {"__init__": _base_init}),
            "Field": (lambda *a, **k: None),
            "field_validator": (lambda *a, **k: (lambda f: f)),
        },
    )

    # Engine imports pybigtools (not bigtools); provide stub if missing
    _ensure_stub("pybigtools", {"open": lambda *a, **k: types.SimpleNamespace(chroms=lambda: {})})

    class _LoggerStub:
        def info(self, *a, **k):
            pass
        def warning(self, *a, **k):
            pass
        def error(self, *a, **k):
            pass
        def remove(self, *a, **k):
            pass
        def add(self, *a, **k):
            pass

    _ensure_stub("loguru", {"logger": _LoggerStub()})

    # Provide minimal psutil.cpu_count if psutil missing
    if importlib.util.find_spec("psutil") is None:
        _ensure_stub("psutil", {"cpu_count": (lambda logical=True: 2)})

    # Import the engine module from package
    engine = importlib.import_module("wignado.engine")
    return engine


def test_clean_genomic_values_simd():
    engine = import_engine_with_stubs()
    arr = np.array([1.0, np.nan, np.inf, -2000.0, 5.0], dtype=np.float32)
    out = engine.normalize_genomic_values(arr)
    assert out.dtype == np.float32
    assert out[0] == 1.0
    assert out[1] == 0.0
    assert out[2] == 0.0
    assert out[3] == 0.0
    assert out[4] == 5.0


def test_calculate_chunk_stats():
    engine = import_engine_with_stubs()
    arr = np.array([1.0, 2.0, np.nan, 3.0, np.inf], dtype=np.float64)
    mean, ma, mi, cnt = engine.compute_chunk_statistics(arr)
    assert cnt == 3
    assert mean == np.mean([1.0, 2.0, 3.0])
    assert ma == 3.0
    assert mi == 1.0


def test_genomic_interval_and_chromosomeinfo():
    engine = import_engine_with_stubs()
    gi = engine.GenomicRegion(chrom="chr1", start=1000, end=2000)
    assert gi.length == 1000
    s = str(gi)
    assert "chr1" in s and ":" in s and "-" in s

    ci = engine.ContigInfo(name="chr1", length=248956422, gc_content=0.41)
    s2 = str(ci)
    assert "chr1" in s2 and "GC" in s2


def test_benchmark_results_properties():
    engine = import_engine_with_stubs()
    br = engine.QueryBenchmark(
        total_queries=10, iterations=1, query_times=[0.1, 0.2, 0.15], total_values_retrieved=100
    )
    assert abs(br.avg_time_seconds - np.mean([0.1, 0.2, 0.15])) < 1e-12
    assert br.ms_per_query > 0