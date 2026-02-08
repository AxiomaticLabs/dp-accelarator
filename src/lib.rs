#![allow(non_local_definitions)] // pyo3 0.20 macro; remove after upgrading pyo3

use pyo3::prelude::*;

mod accounting;
mod gaussian;
mod math;
mod pld;

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

// ── Gaussian mechanism (Balle & Wang) ────────────────────────────

#[pyfunction]
fn get_epsilon_gaussian(sigma: f64, delta: f64, tol: f64) -> PyResult<f64> {
    Ok(gaussian::get_epsilon_gaussian_impl(sigma, delta, tol))
}

#[pyfunction]
fn get_sigma_gaussian(epsilon: f64, delta: f64, tol: f64) -> PyResult<f64> {
    Ok(gaussian::get_sigma_gaussian_impl(epsilon, delta, tol))
}

// ── RDP primitives ───────────────────────────────────────────────

#[pyfunction]
fn compute_rdp_poisson_subsampled_gaussian(
    q: f64,
    sigma: f64,
    orders: Vec<f64>,
) -> PyResult<Vec<f64>> {
    Ok(orders
        .iter()
        .map(|&a| accounting::compute_rdp_single_order(q, sigma, a))
        .collect())
}

#[pyfunction]
fn compute_rdp_sample_wor_gaussian(q: f64, sigma: f64, orders: Vec<f64>) -> PyResult<Vec<f64>> {
    Ok(orders
        .iter()
        .map(|&a| accounting::compute_rdp_sample_wor_single(q, sigma, a))
        .collect())
}

#[pyfunction]
fn compute_rdp_laplace(pure_eps: f64, orders: Vec<f64>) -> PyResult<Vec<f64>> {
    Ok(orders
        .iter()
        .map(|&a| accounting::laplace_rdp(pure_eps, a))
        .collect())
}

/// Returns (epsilon, optimal_order).
#[pyfunction]
fn rdp_to_epsilon_vec(orders: Vec<f64>, rdp_values: Vec<f64>, delta: f64) -> PyResult<(f64, f64)> {
    let mut best_eps = f64::INFINITY;
    let mut best_order = orders[0];
    for (i, (&a, &r)) in orders.iter().zip(rdp_values.iter()).enumerate() {
        let eps = accounting::rdp_to_epsilon(a, r, delta);
        if eps < best_eps {
            best_eps = eps;
            best_order = orders[i];
        }
    }
    Ok((f64::max(0.0, best_eps), best_order))
}

/// Returns (delta, optimal_order).
#[pyfunction]
fn rdp_to_delta_vec(orders: Vec<f64>, rdp_values: Vec<f64>, epsilon: f64) -> PyResult<(f64, f64)> {
    let mut best_delta = f64::INFINITY;
    let mut best_order = orders[0];
    for (i, (&a, &r)) in orders.iter().zip(rdp_values.iter()).enumerate() {
        let d = accounting::rdp_to_delta_single(a, r, epsilon);
        if d < best_delta {
            best_delta = d;
            best_order = orders[i];
        }
    }
    Ok((f64::min(1.0, best_delta), best_order))
}

#[pymodule]
fn _core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_epsilon_batch, m)?)?;
    m.add_function(wrap_pyfunction!(get_epsilon_gaussian, m)?)?;
    m.add_function(wrap_pyfunction!(get_sigma_gaussian, m)?)?;
    m.add_function(wrap_pyfunction!(
        compute_rdp_poisson_subsampled_gaussian,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(compute_rdp_sample_wor_gaussian, m)?)?;
    m.add_function(wrap_pyfunction!(compute_rdp_laplace, m)?)?;
    m.add_function(wrap_pyfunction!(rdp_to_epsilon_vec, m)?)?;
    m.add_function(wrap_pyfunction!(rdp_to_delta_vec, m)?)?;
    m.add_class::<pld::RustPldPmf>()?;
    Ok(())
}
