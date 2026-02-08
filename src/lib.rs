use pyo3::prelude::*;
use statrs::function::erf::erfc;
use statrs::function::gamma::ln_gamma;
use std::f64::consts::{PI, SQRT_2};

// ============================================================================
// Exact port of Google's dp_accounting/rdp/rdp_privacy_accountant.py
// ============================================================================

const MAX_STEPS_LOG_A_FRAC: i32 = 1000;

// ---------------------------------------------------------------------------
// Core math helpers (exact clones of Google's functions)
// ---------------------------------------------------------------------------

/// Exact clone of Google's `_log_add`.
/// Adds two numbers in log space: log(exp(logx) + exp(logy)).
fn log_add(logx: f64, logy: f64) -> f64 {
    let (a, b) = if logx < logy { (logx, logy) } else { (logy, logx) };
    if a == f64::NEG_INFINITY {
        return b;
    }
    // log(exp(a) + exp(b)) = log((exp(a-b) + 1) * exp(b)) = log1p(exp(a-b)) + b
    (a - b).exp().ln_1p() + b
}

/// log(|Gamma(x)|) that handles negative non-integer x via reflection.
/// Matches scipy.special.gammaln behavior.
fn log_abs_gamma(x: f64) -> f64 {
    if x > 0.0 {
        ln_gamma(x)
    } else if x == x.floor() {
        // Pole at non-positive integers: |Gamma| → ∞, log → ∞
        f64::INFINITY
    } else {
        // Reflection formula: Gamma(x)*Gamma(1-x) = π/sin(πx)
        // ⇒ log|Gamma(x)| = log(π) - log|sin(πx)| - log|Gamma(1-x)|
        PI.ln() - (PI * x).sin().abs().ln() - ln_gamma(1.0 - x)
    }
}

/// Exact clone of Google's `_log_comb`.
/// Computes log of the (generalized) binomial coefficient C(n, k).
fn log_comb(n: f64, k: f64) -> f64 {
    log_abs_gamma(n + 1.0) - log_abs_gamma(k + 1.0) - log_abs_gamma(n - k + 1.0)
}

/// Exact clone of Google's `_log_erfc`.
/// Computes log(erfc(x)) with high accuracy for large x.
/// Uses: erfc(x) = 2·Φ(−x√2), so log(erfc(x)) = log(2) + log_ndtr(−x√2).
/// Falls back to Laurent series when erfc underflows to 0.
fn log_erfc(x: f64) -> f64 {
    let r = erfc(x);
    if r == 0.0 {
        // Laurent series at infinity for the tail of erfc:
        //   erfc(x) ~ exp(-x²-0.5/x²+0.625/x⁴) / (x·√π)
        -PI.ln() / 2.0
            - x.ln()
            - x * x
            - 0.5 * x.powi(-2)
            + 0.625 * x.powi(-4)
            - 37.0 / 24.0 * x.powi(-6)
            + 353.0 / 64.0 * x.powi(-8)
    } else {
        r.ln()
    }
}

// ---------------------------------------------------------------------------
// RDP computation for the Poisson Subsampled Gaussian Mechanism
// ---------------------------------------------------------------------------

/// Exact clone of Google's `_compute_log_a_int`.
/// Computes log(A_alpha) for INTEGER alpha, 0 < q < 1.
fn compute_log_a_int(q: f64, sigma: f64, alpha: i64) -> f64 {
    let mut log_a = f64::NEG_INFINITY;
    let log1mq = (-q).ln_1p(); // log(1 − q), using ln_1p for accuracy
    let log_q = q.ln();
    let two_sigma_sq = 2.0 * sigma * sigma;

    for i in 0..=alpha {
        let i_f = i as f64;
        let alpha_f = alpha as f64;
        let log_coef_i = log_comb(alpha_f, i_f)
            + i_f * log_q
            + (alpha_f - i_f) * log1mq;
        let s = log_coef_i + (i_f * i_f - i_f) / two_sigma_sq;
        log_a = log_add(log_a, s);
    }
    log_a
}

/// Exact clone of Google's `_compute_log_a_frac`.
/// Computes log(A_alpha) for FRACTIONAL alpha, 0 < q < 1.
/// Derived from Sec 3.3 of https://arxiv.org/pdf/1908.10530.
fn compute_log_a_frac(q: f64, sigma: f64, alpha: f64) -> f64 {
    let mut log_a0 = f64::NEG_INFINITY;
    let mut log_a1 = f64::NEG_INFINITY;
    let z0 = sigma * sigma * (1.0 / q - 1.0).ln() + 0.5;
    let log1mq = (-q).ln_1p(); // log(1 − q)
    let log_q = q.ln();
    let two_sigma_sq = 2.0 * sigma * sigma;
    let sqrt2_sigma = SQRT_2 * sigma;

    let mut last_s0 = f64::NEG_INFINITY;
    let mut last_s1 = f64::NEG_INFINITY;

    for i in 0..MAX_STEPS_LOG_A_FRAC {
        let i_f = i as f64;
        let log_coef = log_comb(alpha, i_f);
        let j = alpha - i_f;

        let log_t0 = log_coef + i_f * log_q + j * log1mq;
        let log_t1 = log_coef + j * log_q + i_f * log1mq;

        // log(0.5) + log_erfc(·) matches Google's formula exactly
        let log_e0 = 0.5_f64.ln() + log_erfc((i_f - z0) / sqrt2_sigma);
        let log_e1 = 0.5_f64.ln() + log_erfc((z0 - j) / sqrt2_sigma);

        let log_s0 = log_t0 + (i_f * i_f - i_f) / two_sigma_sq + log_e0;
        let log_s1 = log_t1 + (j * j - j) / two_sigma_sq + log_e1;

        log_a0 = log_add(log_a0, log_s0);
        log_a1 = log_add(log_a1, log_s1);

        let total = log_add(log_a0, log_a1);

        // Convergence: terminate when both s0 and s1 are decreasing and
        // sufficiently small relative to the total (matches Google exactly).
        if log_s0 < last_s0
            && log_s1 < last_s1
            && f64::max(log_s0, log_s1) < total - 30.0
        {
            return total;
        }

        last_s0 = log_s0;
        last_s1 = log_s1;
    }

    // Failed to converge → exclude this order by returning infinity
    f64::INFINITY
}

/// Exact clone of Google's `_compute_log_a`.
/// Dispatches to int or frac variant based on whether alpha is integer.
fn compute_log_a(q: f64, sigma: f64, alpha: f64) -> f64 {
    if alpha == alpha.floor() && alpha.is_finite() {
        compute_log_a_int(q, sigma, alpha as i64)
    } else {
        compute_log_a_frac(q, sigma, alpha)
    }
}

/// Exact clone of Google's `_compute_rdp_poisson_subsampled_gaussian`
/// for a single order.
fn compute_rdp_single_order(q: f64, sigma: f64, alpha: f64) -> f64 {
    if q == 0.0 {
        return 0.0;
    }
    if alpha.is_infinite() || sigma == 0.0 {
        return f64::INFINITY;
    }
    if q == 1.0 {
        return alpha / (2.0 * sigma * sigma);
    }
    compute_log_a(q, sigma, alpha) / (alpha - 1.0)
}

// ---------------------------------------------------------------------------
// RDP → ε conversion (improved bound, Proposition 12 of arXiv:2004.00010)
// ---------------------------------------------------------------------------

/// Exact clone of Google's `compute_epsilon` for a single (order, rdp) pair.
fn rdp_to_epsilon(alpha: f64, rdp: f64, delta: f64) -> f64 {
    if alpha < 1.0 {
        return f64::INFINITY;
    }
    if rdp < 0.0 {
        return 0.0;
    }
    if rdp == 0.0 {
        return 0.0;
    }

    // KL divergence bound: delta ≤ √(1 − exp(−KL))
    // If delta² + expm1(−rdp) > 0, epsilon = 0
    if delta * delta + (-rdp).exp_m1() > 0.0 {
        return 0.0;
    }

    if alpha > 1.01 {
        // Improved bound (Proposition 12):
        // ε = rdp + log(1 − 1/α) − log(δ·α) / (α − 1)
        rdp + (1.0 - 1.0 / alpha).ln() - (delta * alpha).ln() / (alpha - 1.0)
    } else {
        // Numerically unstable for α close to 1 — exclude
        f64::INFINITY
    }
}

// ---------------------------------------------------------------------------
// PyO3-exported batch computation
// ---------------------------------------------------------------------------

/// Batch computation of ε for multiple step counts.
/// Pre-computes per-step RDP once (the expensive part), then vectorises
/// the composition + ε conversion across all step counts.
#[pyfunction]
fn compute_epsilon_batch(
    q: f64,
    noise_multiplier: f64,
    steps_list: Vec<i64>,
    orders: Vec<f64>,
    delta: f64,
) -> PyResult<Vec<f64>> {
    // 1. Pre-compute per-step RDP for every order (most expensive work)
    let rdp_per_step: Vec<f64> = orders
        .iter()
        .map(|&alpha| compute_rdp_single_order(q, noise_multiplier, alpha))
        .collect();

    // 2. For each step count: compose (multiply) then convert RDP → ε
    let results: Vec<f64> = steps_list
        .iter()
        .map(|&steps| {
            let steps_f = steps as f64;
            let mut min_eps = f64::INFINITY;
            for (i, &alpha) in orders.iter().enumerate() {
                let total_rdp = rdp_per_step[i] * steps_f;
                let eps = rdp_to_epsilon(alpha, total_rdp, delta);
                if eps < min_eps {
                    min_eps = eps;
                }
            }
            f64::max(0.0, min_eps)
        })
        .collect();

    Ok(results)
}

#[pymodule]
fn _core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_epsilon_batch, m)?)?;
    Ok(())
}
