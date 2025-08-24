use pyo3::prelude::*;

/// A simple function exposed to Python.
#[pyfunction]
fn add(a: i64, b: i64) -> PyResult<i64> {
    Ok(a + b)
}

/// Rust Python module named `_wignado`.
#[pymodule]
#[pyo3(name = "_wignado")]
fn wignado(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add("__doc__", "Rust extension module")?;
    Ok(())
}
