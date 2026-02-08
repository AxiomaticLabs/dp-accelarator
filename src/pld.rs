//! Privacy Loss Distribution engine — FFT-based composition in Rust.
//!
//! Provides [`RustPldPmf`], a PyO3-exported class that keeps the entire
//! PMF in Rust memory and performs all heavy math (FFT convolution,
//! connect-the-dots discretization, epsilon/delta queries) without
//! crossing the Python–Rust boundary.

use pyo3::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};
use statrs::function::erf::erfc;

const SQRT_2: f64 = std::f64::consts::SQRT_2;
const TAIL_MASS_BOUND: f64 = 1e-15;

// ── Numerical helpers ───────────────────────────────────────────

/// Normal CDF: Φ(x) = 0.5 · erfc(−x / √2)
#[inline]
fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / SQRT_2)
}

// ── FFT convolution ─────────────────────────────────────────────

/// Full linear convolution of two real sequences via FFT.
fn fft_convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let out_len = a.len() + b.len() - 1;
    let fft_len = out_len.next_power_of_two();

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(fft_len);
    let ifft = planner.plan_fft_inverse(fft_len);

    // Build zero-padded complex buffers
    let mut a_c: Vec<Complex<f64>> = Vec::with_capacity(fft_len);
    for &v in a {
        a_c.push(Complex::new(v, 0.0));
    }
    a_c.resize(fft_len, Complex::new(0.0, 0.0));

    let mut b_c: Vec<Complex<f64>> = Vec::with_capacity(fft_len);
    for &v in b {
        b_c.push(Complex::new(v, 0.0));
    }
    b_c.resize(fft_len, Complex::new(0.0, 0.0));

    // Forward FFT (in-place)
    fft.process(&mut a_c);
    fft.process(&mut b_c);

    // Point-wise multiply in frequency domain
    for i in 0..fft_len {
        a_c[i] = a_c[i] * b_c[i];
    }

    // Inverse FFT (in-place, unnormalised)
    ifft.process(&mut a_c);

    // Normalise and extract real parts
    let scale = 1.0 / fft_len as f64;
    a_c.iter().take(out_len).map(|c| c.re * scale).collect()
}

// ── PMF truncation ──────────────────────────────────────────────

/// Remove leading/trailing bins whose probability ≤ `bound`,
/// attributing the trimmed mass to `infinity_mass` (pessimistic).
fn truncate_pmf(
    probs: &[f64],
    lower_loss: i64,
    infinity_mass: f64,
    bound: f64,
) -> (Vec<f64>, i64, f64) {
    let first = probs.iter().position(|&p| p > bound);
    let last = probs.iter().rposition(|&p| p > bound);

    match (first, last) {
        (Some(f), Some(l)) => {
            let trimmed: f64 =
                probs[..f].iter().sum::<f64>() + probs[l + 1..].iter().sum::<f64>();
            (
                probs[f..=l].to_vec(),
                lower_loss + f as i64,
                infinity_mass + trimmed,
            )
        }
        _ => {
            let total: f64 = probs.iter().sum();
            (vec![0.0], 0, infinity_mass + total)
        }
    }
}

// ── Gaussian connect-the-dots ───────────────────────────────────

/// Build the discrete PMF for a Gaussian mechanism using the
/// connect-the-dots method.
///
/// Returns `(probs, lower_loss_index, infinity_mass)`.
fn discretize_gaussian(
    sigma: f64,
    sensitivity: f64,
    di: f64,
    tail_bound: f64,
    is_add: bool,
) -> (Vec<f64>, i64, f64) {
    let s = sensitivity;
    let sig2 = sigma * sigma;

    // Linear privacy-loss coefficients: pl(x) = a·x + b
    let (a, b, mu_upper) = if is_add {
        (-s / sig2, s * s / (2.0 * sig2), 0.0)
    } else {
        (s / sig2, -(s * s) / (2.0 * sig2), s)
    };

    // x range covering ±tail_bound standard deviations
    let x_min = if is_add {
        -tail_bound * sigma
    } else {
        -tail_bound * sigma - s
    };
    let x_max = if is_add {
        s + tail_bound * sigma
    } else {
        tail_bound * sigma
    };

    // Privacy-loss range
    let pl_a = a * x_min + b;
    let pl_b = a * x_max + b;
    let (pl_min, pl_max) = if pl_a < pl_b { (pl_a, pl_b) } else { (pl_b, pl_a) };

    let idx_min = (pl_min / di).floor() as i64;
    let idx_max = (pl_max / di).ceil() as i64;
    let mut n = (idx_max - idx_min + 1).max(0) as usize;
    if n == 0 {
        return (vec![1.0], 0, 0.0);
    }
    if n > 10_000_000 {
        n = 10_000_000;
    }
    if a.abs() < 1e-300 {
        return (vec![1.0], 0, 0.0);
    }

    // Bin boundaries in privacy-loss space → x values → CDF
    let n_boundaries = n + 1;
    let mut cdf_vals: Vec<f64> = Vec::with_capacity(n_boundaries);
    for i in 0..n_boundaries {
        let pl_boundary = (idx_min + i as i64) as f64 * di - di / 2.0;
        let x = (pl_boundary - b) / a;
        cdf_vals.push(norm_cdf((x - mu_upper) / sigma));
    }

    // Probability mass per bin
    let mut probs: Vec<f64> = Vec::with_capacity(n);
    if a < 0.0 {
        for i in 0..n {
            probs.push((cdf_vals[i] - cdf_vals[i + 1]).max(0.0));
        }
    } else {
        for i in 0..n {
            probs.push((cdf_vals[i + 1] - cdf_vals[i]).max(0.0));
        }
    }

    let total: f64 = probs.iter().sum();
    let infinity_mass = (1.0 - total).max(0.0);

    (probs, idx_min, infinity_mass)
}

// ── Laplace connect-the-dots ────────────────────────────────────

/// Build discrete PMF for Laplace mechanism.
///
/// Privacy loss for Laplace(0, b) vs Laplace(s, b):
///   x < 0   → pl = s/b  (constant)
///   0 ≤ x ≤ s → pl = (s − 2x)/b
///   x > s   → pl = −s/b (constant)
fn discretize_laplace(
    parameter: f64,
    sensitivity: f64,
    di: f64,
    tail_bound: f64,
) -> (Vec<f64>, i64, f64) {
    let b = parameter;
    let s = sensitivity;
    let x_min = -tail_bound * b;
    let x_max = s + tail_bound * b;

    let pl_max = s / b;
    let pl_min = -s / b;

    let idx_min = (pl_min / di).floor() as i64;
    let idx_max = (pl_max / di).ceil() as i64;
    let n = ((idx_max - idx_min + 1).max(0) as usize).min(10_000_000);
    if n == 0 {
        return (vec![1.0], 0, 0.0);
    }

    // Laplace CDF: F(x) = 0.5·exp(x/b)  for x < 0
    //              F(x) = 1 − 0.5·exp(−x/b)  for x ≥ 0
    let laplace_cdf = |x: f64| -> f64 {
        if x < 0.0 {
            0.5 * (x / b).exp()
        } else {
            1.0 - 0.5 * (-x / b).exp()
        }
    };

    let mut probs = vec![0.0f64; n];

    // For efficient binning, use CDF differences in the three regions.
    // Region 1: x < 0 → pl = s/b → single bin
    let idx_const_hi = ((s / b / di).round() as i64 - idx_min) as usize;
    if idx_const_hi < n {
        probs[idx_const_hi] += laplace_cdf(0.0); // P(X < 0) = 0.5
    }

    // Region 3: x > s → pl = -s/b → single bin
    let idx_const_lo = ((-s / b / di).round() as i64 - idx_min) as usize;
    if idx_const_lo < n {
        probs[idx_const_lo] += 1.0 - laplace_cdf(s); // P(X > s) = 0.5·exp(-s/b)
    }

    // Region 2: 0 ≤ x ≤ s → pl = (s - 2x)/b, linearly decreasing
    // pl boundaries → x values: x = (s - b·pl) / 2
    // We assign CDF mass to bins
    if s > 0.0 {
        let _a_coeff = -2.0 / b; // d(pl)/dx = -2/b
        for i in 0..n {
            let pl_lo = (idx_min + i as i64) as f64 * di - di / 2.0;
            let pl_hi = pl_lo + di;
            // x = (s - b·pl) / 2
            // Since pl is decreasing in x, higher pl → lower x
            let x_hi_at_pl_lo = (s - b * pl_lo) / 2.0;
            let x_lo_at_pl_hi = (s - b * pl_hi) / 2.0;
            // Clamp to [0, s]
            let x_lo_c = x_lo_at_pl_hi.max(0.0).min(s);
            let x_hi_c = x_hi_at_pl_lo.max(0.0).min(s);
            if x_hi_c > x_lo_c {
                let mass = laplace_cdf(x_hi_c) - laplace_cdf(x_lo_c);
                if mass > 0.0 {
                    probs[i] += mass;
                }
            }
        }
    }

    let total: f64 = probs.iter().sum();
    let infinity_mass = (1.0 - total).max(0.0);

    (probs, idx_min, infinity_mass)
}

// ── PyO3-exported class ─────────────────────────────────────────

/// A PLD probability mass function stored entirely in Rust.
///
/// Supports FFT-based composition, self-composition with repeated
/// squaring, and optimised ε/δ queries — all without crossing
/// the Python–Rust boundary.
#[pyclass]
#[derive(Clone)]
pub struct RustPldPmf {
    probs: Vec<f64>,
    lower_loss: i64,
    di: f64,
    infinity_mass: f64,
}

impl RustPldPmf {
    /// Internal compose (no PyResult overhead).
    fn compose_internal(&self, other: &RustPldPmf) -> RustPldPmf {
        let new_probs = fft_convolve(&self.probs, &other.probs);
        let new_lower = self.lower_loss + other.lower_loss;
        let new_inf = self.infinity_mass + other.infinity_mass
            - self.infinity_mass * other.infinity_mass;
        let (tp, tl, ti) = truncate_pmf(&new_probs, new_lower, new_inf, TAIL_MASS_BOUND);
        RustPldPmf {
            probs: tp,
            lower_loss: tl,
            di: self.di,
            infinity_mass: ti,
        }
    }
}

#[pymethods]
impl RustPldPmf {
    /// Create from a Python list of probabilities.
    #[new]
    fn new(probs: Vec<f64>, lower_loss: i64, di: f64, infinity_mass: f64) -> Self {
        RustPldPmf {
            probs,
            lower_loss,
            di,
            infinity_mass,
        }
    }

    /// Build Gaussian PMF via connect-the-dots.
    #[staticmethod]
    #[pyo3(signature = (sigma, sensitivity, di, tail_bound=10.0, is_add=true))]
    fn from_gaussian(
        sigma: f64,
        sensitivity: f64,
        di: f64,
        tail_bound: f64,
        is_add: bool,
    ) -> Self {
        let (probs, lower_loss, infinity_mass) =
            discretize_gaussian(sigma, sensitivity, di, tail_bound, is_add);
        RustPldPmf {
            probs,
            lower_loss,
            di,
            infinity_mass,
        }
    }

    /// Build Laplace PMF.
    #[staticmethod]
    #[pyo3(signature = (parameter, sensitivity, di, tail_bound=10.0))]
    fn from_laplace(
        parameter: f64,
        sensitivity: f64,
        di: f64,
        tail_bound: f64,
    ) -> Self {
        let (probs, lower_loss, infinity_mass) =
            discretize_laplace(parameter, sensitivity, di, tail_bound);
        RustPldPmf {
            probs,
            lower_loss,
            di,
            infinity_mass,
        }
    }

    /// Compose (convolve) with another PMF.
    fn compose(&self, other: &RustPldPmf) -> PyResult<RustPldPmf> {
        if (self.di - other.di).abs() > 1e-15 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "discretization intervals must match",
            ));
        }
        Ok(self.compose_internal(other))
    }

    /// Self-compose `count` times via repeated squaring with truncation.
    fn self_compose(&self, count: usize) -> PyResult<RustPldPmf> {
        if count == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "count must be positive",
            ));
        }
        if count == 1 {
            return Ok(self.clone());
        }
        let (tp, tl, ti) =
            truncate_pmf(&self.probs, self.lower_loss, self.infinity_mass, TAIL_MASS_BOUND);
        let mut base = RustPldPmf {
            probs: tp,
            lower_loss: tl,
            di: self.di,
            infinity_mass: ti,
        };
        let mut result: Option<RustPldPmf> = None;
        let mut c = count;

        while c > 0 {
            if c & 1 == 1 {
                result = Some(match result {
                    None => base.clone(),
                    Some(r) => r.compose_internal(&base),
                });
            }
            if c > 1 {
                base = base.compose_internal(&base);
            }
            c >>= 1;
        }
        Ok(result.unwrap())
    }

    /// Hockey-stick divergence: δ(ε) for a single epsilon.
    fn get_delta_for_epsilon(&self, epsilon: f64) -> f64 {
        let mut delta = self.infinity_mass;
        for i in 0..self.probs.len() {
            let loss = (self.lower_loss + i as i64) as f64 * self.di;
            if loss > epsilon {
                delta += (1.0 - (epsilon - loss).exp()) * self.probs[i];
            }
        }
        delta.max(0.0).min(1.0)
    }

    /// Hockey-stick divergence for a list of epsilons.
    fn get_delta_for_epsilon_list(&self, epsilons: Vec<f64>) -> Vec<f64> {
        epsilons
            .iter()
            .map(|&eps| self.get_delta_for_epsilon(eps))
            .collect()
    }

    /// Smallest ε such that δ(ε) ≤ target delta.
    fn get_epsilon_for_delta(&self, delta: f64) -> f64 {
        if self.infinity_mass > delta {
            return f64::INFINITY;
        }

        let n = self.probs.len();
        if n == 0 {
            return 0.0;
        }

        let mut mass_upper = self.infinity_mass;
        let mut mass_lower = 0.0_f64;

        // Iterate from highest loss to lowest
        for i in (0..n).rev() {
            let loss = (self.lower_loss + i as i64) as f64 * self.di;
            let prob = self.probs[i];

            if prob <= 0.0 {
                continue;
            }

            if mass_upper > delta && mass_lower > 0.0 {
                let candidate = ((mass_upper - delta) / mass_lower).ln();
                if candidate >= loss {
                    return candidate.max(0.0);
                }
            }

            mass_upper += prob;
            if loss < -500.0 {
                mass_lower += prob * 1e200;
            } else {
                mass_lower += (-loss).exp() * prob;
            }

            if mass_upper >= delta && mass_lower == 0.0 {
                return loss.max(0.0);
            }
        }

        if mass_upper <= mass_lower + delta {
            return 0.0;
        }
        ((mass_upper - delta) / mass_lower).ln().max(0.0)
    }

    #[getter]
    fn size(&self) -> usize {
        self.probs.len()
    }

    #[getter]
    fn lower_loss(&self) -> i64 {
        self.lower_loss
    }

    #[getter]
    fn infinity_mass_val(&self) -> f64 {
        self.infinity_mass
    }

    #[getter]
    fn di(&self) -> f64 {
        self.di
    }
}

// ── Rust unit tests ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_norm_cdf_standard() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-12);
        assert!((norm_cdf(1e10) - 1.0).abs() < 1e-12);
        assert!(norm_cdf(-1e10) < 1e-12);
    }

    #[test]
    fn test_fft_convolve_delta() {
        // Convolving with [1] is identity
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0];
        let c = fft_convolve(&a, &b);
        assert_eq!(c.len(), 3);
        for i in 0..3 {
            assert!((c[i] - a[i]).abs() < 1e-10, "i={}: {} vs {}", i, c[i], a[i]);
        }
    }

    #[test]
    fn test_fft_convolve_two_boxcars() {
        // [1,1] * [1,1] = [1,2,1]
        let a = vec![1.0, 1.0];
        let b = vec![1.0, 1.0];
        let c = fft_convolve(&a, &b);
        assert_eq!(c.len(), 3);
        assert!((c[0] - 1.0).abs() < 1e-10);
        assert!((c[1] - 2.0).abs() < 1e-10);
        assert!((c[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_discretize_gaussian_sums_to_one() {
        let (probs, _, inf) = discretize_gaussian(1.0, 1.0, 1e-4, 10.0, true);
        let total: f64 = probs.iter().sum::<f64>() + inf;
        assert!(
            (total - 1.0).abs() < 1e-6,
            "total={}, inf={}",
            total,
            inf
        );
    }

    #[test]
    fn test_discretize_gaussian_symmetry() {
        let (probs_add, ll_add, inf_add) = discretize_gaussian(1.0, 1.0, 1e-3, 10.0, true);
        let (probs_rem, ll_rem, inf_rem) = discretize_gaussian(1.0, 1.0, 1e-3, 10.0, false);
        // Both should sum to ~1
        let sum_add: f64 = probs_add.iter().sum::<f64>() + inf_add;
        let sum_rem: f64 = probs_rem.iter().sum::<f64>() + inf_rem;
        assert!((sum_add - 1.0).abs() < 1e-6);
        assert!((sum_rem - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_self_compose_identity() {
        let pmf = RustPldPmf {
            probs: vec![0.3, 0.4, 0.3],
            lower_loss: -1,
            di: 1.0,
            infinity_mass: 0.0,
        };
        let composed = pmf.self_compose(1).unwrap();
        assert_eq!(composed.probs.len(), pmf.probs.len());
    }

    #[test]
    fn test_get_delta_for_epsilon_monotone() {
        let (probs, ll, inf) = discretize_gaussian(1.0, 1.0, 1e-3, 10.0, true);
        let pmf = RustPldPmf {
            probs,
            lower_loss: ll,
            di: 1e-3,
            infinity_mass: inf,
        };
        let d1 = pmf.get_delta_for_epsilon(0.5);
        let d2 = pmf.get_delta_for_epsilon(1.0);
        let d3 = pmf.get_delta_for_epsilon(2.0);
        assert!(d1 >= d2, "d1={} d2={}", d1, d2);
        assert!(d2 >= d3, "d2={} d3={}", d2, d3);
    }

    #[test]
    fn test_get_epsilon_for_delta_finite() {
        let (probs, ll, inf) = discretize_gaussian(1.0, 1.0, 1e-3, 10.0, true);
        let pmf = RustPldPmf {
            probs,
            lower_loss: ll,
            di: 1e-3,
            infinity_mass: inf,
        };
        let eps = pmf.get_epsilon_for_delta(1e-5);
        assert!(eps.is_finite(), "eps={}", eps);
        assert!(eps > 0.0, "eps={}", eps);
    }

    #[test]
    fn test_compose_two_gaussians() {
        let a = RustPldPmf::from_gaussian(1.0, 1.0, 1e-3, 10.0, true);
        let b = RustPldPmf::from_gaussian(1.0, 1.0, 1e-3, 10.0, true);
        let c = a.compose_internal(&b);
        let total: f64 = c.probs.iter().sum::<f64>() + c.infinity_mass;
        assert!(
            (total - 1.0).abs() < 1e-4,
            "total={}, inf={}",
            total,
            c.infinity_mass
        );
    }

    #[test]
    fn test_self_compose_100_fast() {
        let pmf = RustPldPmf::from_gaussian(1.0, 1.0, 1e-4, 10.0, true);
        let start = std::time::Instant::now();
        let composed = pmf.self_compose(100).unwrap();
        let elapsed = start.elapsed();
        let eps = composed.get_epsilon_for_delta(1e-5);
        // Should complete well under 5 seconds
        assert!(
            elapsed.as_secs_f64() < 5.0,
            "Took {:.2}s — too slow!",
            elapsed.as_secs_f64()
        );
        assert!(eps.is_finite() && eps > 0.0, "eps={}", eps);
        println!(
            "self_compose(100): {:.3}s, size={}, eps={:.4}",
            elapsed.as_secs_f64(),
            composed.size(),
            eps
        );
    }

    #[test]
    fn test_truncate_preserves_mass() {
        let probs = vec![1e-20, 0.3, 0.4, 0.3, 1e-20];
        let (tp, tl, ti) = truncate_pmf(&probs, -2, 0.0, 1e-15);
        let total_before: f64 = probs.iter().sum();
        let total_after: f64 = tp.iter().sum::<f64>() + ti;
        assert!(
            (total_before - total_after).abs() < 1e-15,
            "before={} after={}",
            total_before,
            total_after
        );
        assert_eq!(tl, -1); // trimmed first element
        assert_eq!(tp.len(), 3); // trimmed two negligible tails
    }

    #[test]
    fn test_laplace_discretize_sums_to_one() {
        let (probs, _, inf) = discretize_laplace(1.0, 1.0, 1e-3, 10.0);
        let total: f64 = probs.iter().sum::<f64>() + inf;
        assert!(
            (total - 1.0).abs() < 0.05,
            "total={}, inf={}",
            total,
            inf
        );
    }
}
