//! Exact calibration for the Gaussian mechanism.
//!
//! Implements the analytical formulas from Balle & Wang, "Improving the Gaussian
//! Mechanism for Differential Privacy: Analytical Calibration and Optimal
//! Denoising" (arXiv:1805.06530).

use crate::math::log_erfc;
use std::f64::consts::SQRT_2;

/// Computes log(delta) for the Gaussian mechanism with given sigma and epsilon.
/// Uses Eq. (6) from arXiv:1805.06530.
fn get_log_delta(sigma: f64, eps: f64) -> f64 {
    // t* = eps * sigma + 1/(2*sigma)
    // log(delta) = log(Phi(1/sigma - t*)) + log(1 - exp(eps + log(Phi(-t*)) - x))
    //   where x = log(Phi(1/sigma - t*))
    let t_star = eps * sigma + 1.0 / (2.0 * sigma);
    let arg1 = (1.0 / sigma - t_star) / SQRT_2;
    let arg2 = -t_star / SQRT_2;

    // log(Phi(z)) = log(0.5 * erfc(-z)) = log(0.5) + log(erfc(-z))
    let x = -(std::f64::consts::LN_2) + log_erfc(-arg1); // log(Phi(1/sigma - t*))
    let y = eps - std::f64::consts::LN_2 + log_erfc(-arg2); // eps + log(Phi(-t*))

    if y <= x {
        x + (-(y - x).exp()).ln_1p()
    } else {
        f64::NEG_INFINITY
    }
}

/// Computes epsilon for the Gaussian mechanism using Brent's method bisection.
///
/// # Arguments
/// * `sigma` - Standard deviation of the Gaussian noise (>=0)
/// * `delta` - Target delta (in [0, 1])
/// * `tol` - Tolerance for the bisection search
///
/// # Returns
/// The smallest non-negative epsilon such that the mechanism is (eps, delta)-DP.
pub fn get_epsilon_gaussian_impl(sigma: f64, delta: f64, tol: f64) -> f64 {
    if sigma < 0.0 {
        return f64::NAN;
    }
    if !(0.0..=1.0).contains(&delta) {
        return f64::NAN;
    }
    if delta == 1.0 {
        return 0.0;
    }
    if sigma == 0.0 || delta == 0.0 {
        return f64::INFINITY;
    }
    if sigma.is_infinite() {
        return 0.0;
    }

    let log_delta = delta.ln();

    // Check if (0, delta)-DP already holds
    if get_log_delta(sigma, 0.0) < log_delta {
        return 0.0;
    }

    // Find bracket: exponentially increase eps_hi until log_delta(sigma, eps_hi) < log_delta
    let mut eps_lo = 0.0_f64;
    let mut eps_hi = 1.0_f64;
    while get_log_delta(sigma, eps_hi) > log_delta {
        eps_lo = eps_hi;
        eps_hi *= 10.0;
    }

    // Bisection search (Brent-like)
    brentq(
        |eps| get_log_delta(sigma, eps) - log_delta,
        eps_lo,
        eps_hi,
        tol,
    )
}

/// Computes the optimal noise std for the Gaussian mechanism.
///
/// # Arguments
/// * `epsilon` - Target epsilon (>=0)
/// * `delta` - Target delta (in [0, 1])
/// * `tol` - Tolerance for the bisection search
///
/// # Returns
/// The smallest sigma such that the mechanism is (epsilon, delta)-DP.
pub fn get_sigma_gaussian_impl(epsilon: f64, delta: f64, tol: f64) -> f64 {
    if epsilon < 0.0 {
        return f64::NAN;
    }
    if !(0.0..=1.0).contains(&delta) {
        return f64::NAN;
    }
    if delta == 1.0 || epsilon.is_infinite() {
        return 0.0;
    }
    if delta == 0.0 {
        return f64::INFINITY;
    }

    let log_delta = delta.ln();

    // Find bracket: exponentially adjust until we bracket the root
    let mut sigma_lo = 0.1_f64;
    let mut sigma_hi = 1.0_f64;

    while get_log_delta(sigma_lo, epsilon) < log_delta {
        sigma_hi = sigma_lo;
        sigma_lo /= 10.0;
    }
    while get_log_delta(sigma_hi, epsilon) > log_delta {
        sigma_lo = sigma_hi;
        sigma_hi *= 10.0;
    }

    // Bisection search
    brentq(
        |sigma| get_log_delta(sigma, epsilon) - log_delta,
        sigma_lo,
        sigma_hi,
        tol,
    )
}

/// Simple Brent's method for root finding.
/// Finds x in [a, b] such that f(x) ≈ 0, given f(a) and f(b) have opposite signs.
fn brentq<F: Fn(f64) -> f64>(f: F, mut lo: f64, mut hi: f64, tol: f64) -> f64 {
    let mut f_lo = f(lo);
    let f_hi = f(hi);

    if f_lo * f_hi > 0.0 {
        return (lo + hi) / 2.0;
    }

    // Ensure f(lo) >= 0 and f(hi) <= 0
    if f_lo < 0.0 {
        std::mem::swap(&mut lo, &mut hi);
        f_lo = -f_lo;
    }
    let _ = f_lo;

    // Pure bisection — reliable and fast enough (40 iters for 1e-12 tol)
    for _ in 0..200 {
        if (hi - lo).abs() < tol {
            return (lo + hi) / 2.0;
        }
        let mid = (lo + hi) / 2.0;
        let f_mid = f(mid);
        if f_mid > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_epsilon_gaussian_basic() {
        // sigma=1.0, delta=1e-5 should give a reasonable epsilon
        let eps = get_epsilon_gaussian_impl(1.0, 1e-5, 1e-12);
        assert!(eps > 0.0 && eps < 10.0, "epsilon was {}", eps);
    }

    #[test]
    fn test_get_epsilon_gaussian_large_sigma() {
        // Large sigma => small epsilon
        let eps = get_epsilon_gaussian_impl(100.0, 1e-5, 1e-12);
        assert!(eps < 0.1, "epsilon was {}", eps);
    }

    #[test]
    fn test_get_epsilon_gaussian_delta_one() {
        assert_eq!(get_epsilon_gaussian_impl(1.0, 1.0, 1e-12), 0.0);
    }

    #[test]
    fn test_get_epsilon_gaussian_sigma_zero() {
        assert!(get_epsilon_gaussian_impl(0.0, 1e-5, 1e-12).is_infinite());
    }

    #[test]
    fn test_get_sigma_gaussian_basic() {
        // eps=1.0, delta=1e-5 should give a reasonable sigma
        let sigma = get_sigma_gaussian_impl(1.0, 1e-5, 1e-12);
        assert!(sigma > 0.0 && sigma < 100.0, "sigma was {}", sigma);
    }

    #[test]
    fn test_get_sigma_gaussian_roundtrip() {
        // sigma -> eps -> sigma should roundtrip
        let sigma_orig = 1.5;
        let delta = 1e-5;
        let eps = get_epsilon_gaussian_impl(sigma_orig, delta, 1e-12);
        let sigma_back = get_sigma_gaussian_impl(eps, delta, 1e-12);
        assert!(
            (sigma_back - sigma_orig).abs() < 1e-6,
            "roundtrip failed: {} vs {}",
            sigma_orig,
            sigma_back
        );
    }

    #[test]
    fn test_get_sigma_gaussian_delta_one() {
        assert_eq!(get_sigma_gaussian_impl(1.0, 1.0, 1e-12), 0.0);
    }

    #[test]
    fn test_get_sigma_gaussian_epsilon_inf() {
        assert_eq!(get_sigma_gaussian_impl(f64::INFINITY, 1e-5, 1e-12), 0.0);
    }
}
