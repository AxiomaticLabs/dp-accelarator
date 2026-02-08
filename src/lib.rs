use pyo3::prelude::*;

mod accounting;
mod math;

#[pyfunction]
fn compute_epsilon_batch(
    q: f64,
    noise_multiplier: f64,
    steps_list: Vec<i64>,
    orders: Vec<f64>,
    delta: f64,
) -> PyResult<Vec<f64>> {
    Ok(accounting::compute_epsilon_batch_impl(
        q,
        noise_multiplier,
        &steps_list,
        &orders,
        delta,
    ))
}

#[pymodule]
fn _core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_epsilon_batch, m)?)?;
    Ok(())
}