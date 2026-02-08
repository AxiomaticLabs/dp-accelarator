use std::f64::consts::PI;

/// Stable log-sum-exp: log(exp(a) + exp(b)).
pub fn log_add(a: f64, b: f64) -> f64 {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    if lo == f64::NEG_INFINITY {
        return hi;
    }
    (lo - hi).exp().ln_1p() + hi
}

/// Compute |sin(π*x)| accurately by reducing x mod 2 first,
/// avoiding large-argument precision loss in sin().
fn abs_sinpi(x: f64) -> f64 {
    let r = x.abs().rem_euclid(2.0); // r in [0, 2)
                                     // sin(π*r) for r in [0, 2) is non-negative in [0, 1] and non-positive in [1, 2)
                                     // |sin(π*r)| = sin(π * min(r, 2-r)) for r mapped to [0, 1]
    let s = if r <= 1.0 { r } else { 2.0 - r };
    // Now s in [0, 1], |sin(π*r)| = sin(π*s)
    (PI * s).sin()
}

/// log|Gamma(x)| using libm for better precision than statrs.
pub fn log_abs_gamma(x: f64) -> f64 {
    if x > 0.0 {
        libm::lgamma_r(x).0
    } else if x == x.floor() {
        f64::INFINITY
    } else {
        // Reflection formula: |Gamma(x)| = π / (|sin(πx)| * Gamma(1-x))
        PI.ln() - abs_sinpi(x).ln() - libm::lgamma_r(1.0 - x).0
    }
}

/// Log of the generalised binomial coefficient C(n, k).
pub fn log_comb(n: f64, k: f64) -> f64 {
    log_abs_gamma(n + 1.0) - log_abs_gamma(k + 1.0) - log_abs_gamma(n - k + 1.0)
}

/// Robust log(erfc(x)) using libm::erfc for better precision.
/// Falls back to asymptotic expansion when erfc returns subnormal values
/// (which have too few mantissa bits for accurate logarithm).
pub fn log_erfc(x: f64) -> f64 {
    let r = libm::erfc(x);
    if r < f64::MIN_POSITIVE {
        // erfc underflowed to subnormal or zero — use asymptotic expansion.
        // log(erfc(x)) ≈ -x² - ln(x) - ln(√π) + ln(1 - 1/(2x²) + ...)
        // The coefficients below expand the logarithm of the asymptotic series.
        -PI.ln() / 2.0 - x.ln() - x * x - 0.5 * x.powi(-2) + 0.625 * x.powi(-4)
            - 37.0 / 24.0 * x.powi(-6)
            + 353.0 / 64.0 * x.powi(-8)
    } else {
        r.ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_add_basic() {
        let result = log_add(0.0, 0.0);
        assert!((result - 2.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn log_add_neg_infinity() {
        assert!((log_add(f64::NEG_INFINITY, 5.0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn log_abs_gamma_positive() {
        // Gamma(5) = 4! = 24, ln(24) ≈ 3.178
        assert!((log_abs_gamma(5.0) - 24.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn log_comb_integers() {
        // C(10, 3) = 120
        assert!((log_comb(10.0, 3.0) - 120.0_f64.ln()).abs() < 1e-8);
    }

    #[test]
    fn log_erfc_small_x() {
        // erfc(0) = 1, log(1) = 0
        assert!((log_erfc(0.0)).abs() < 1e-10);
    }
}
