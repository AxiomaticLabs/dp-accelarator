use statrs::function::erf::erfc;
use statrs::function::gamma::ln_gamma;
use std::f64::consts::PI;

/// Stable log-sum-exp: log(exp(a) + exp(b)).
pub fn log_add(a: f64, b: f64) -> f64 {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    if lo == f64::NEG_INFINITY {
        return hi;
    }
    (lo - hi).exp().ln_1p() + hi
}

/// log|Gamma(x)| with reflection for negative non-integer x.
pub fn log_abs_gamma(x: f64) -> f64 {
    if x > 0.0 {
        ln_gamma(x)
    } else if x == x.floor() {
        f64::INFINITY
    } else {
        PI.ln() - (PI * x).sin().abs().ln() - ln_gamma(1.0 - x)
    }
}

/// Log of the generalised binomial coefficient C(n, k).
pub fn log_comb(n: f64, k: f64) -> f64 {
    log_abs_gamma(n + 1.0) - log_abs_gamma(k + 1.0) - log_abs_gamma(n - k + 1.0)
}

/// Robust log(erfc(x)), falling back to Laurent series when erfc underflows.
pub fn log_erfc(x: f64) -> f64 {
    let r = erfc(x);
    if r == 0.0 {
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