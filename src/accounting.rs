use crate::math::{log_add, log_comb, log_erfc};
use std::f64::consts::SQRT_2;

const MAX_STEPS_LOG_A_FRAC: i32 = 1000;

/// Computes log(A_alpha) for INTEGER alpha, 0 < q < 1.
fn compute_log_a_int(q: f64, sigma: f64, alpha: i64) -> f64 {
    let mut log_a = f64::NEG_INFINITY;
    let log1mq = (-q).ln_1p();
    let log_q = q.ln();
    let two_sigma_sq = 2.0 * sigma * sigma;

    for i in 0..=alpha {
        let i_f = i as f64;
        let alpha_f = alpha as f64;
        let log_coef_i = log_comb(alpha_f, i_f) + i_f * log_q + (alpha_f - i_f) * log1mq;
        let s = log_coef_i + (i_f * i_f - i_f) / two_sigma_sq;
        log_a = log_add(log_a, s);
    }
    log_a
}

/// Computes log(A_alpha) for FRACTIONAL alpha, 0 < q < 1.
fn compute_log_a_frac(q: f64, sigma: f64, alpha: f64) -> f64 {
    let mut log_a0 = f64::NEG_INFINITY;
    let mut log_a1 = f64::NEG_INFINITY;
    let z0 = sigma * sigma * (1.0 / q - 1.0).ln() + 0.5;
    let log1mq = (-q).ln_1p();
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

        let log_e0 = 0.5_f64.ln() + log_erfc((i_f - z0) / sqrt2_sigma);
        let log_e1 = 0.5_f64.ln() + log_erfc((z0 - j) / sqrt2_sigma);

        let log_s0 = log_t0 + (i_f * i_f - i_f) / two_sigma_sq + log_e0;
        let log_s1 = log_t1 + (j * j - j) / two_sigma_sq + log_e1;

        log_a0 = log_add(log_a0, log_s0);
        log_a1 = log_add(log_a1, log_s1);

        let total = log_add(log_a0, log_a1);

        if log_s0 < last_s0 && log_s1 < last_s1 && f64::max(log_s0, log_s1) < total - 30.0 {
            return total;
        }

        last_s0 = log_s0;
        last_s1 = log_s1;
    }

    f64::INFINITY
}

/// Dispatches to int or frac variant based on whether alpha is integer.
pub fn compute_log_a(q: f64, sigma: f64, alpha: f64) -> f64 {
    if alpha == alpha.floor() && alpha.is_finite() {
        compute_log_a_int(q, sigma, alpha as i64)
    } else {
        compute_log_a_frac(q, sigma, alpha)
    }
}

/// Computes RDP for the Poisson subsampled Gaussian mechanism (single order).
pub fn compute_rdp_single_order(q: f64, sigma: f64, alpha: f64) -> f64 {
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

/// Converts a single (order, rdp) pair to epsilon (Proposition 12, arXiv:2004.00010).
pub fn rdp_to_epsilon(alpha: f64, rdp: f64, delta: f64) -> f64 {
    if alpha < 1.0 {
        return f64::INFINITY;
    }
    if rdp <= 0.0 {
        return 0.0;
    }
    if delta * delta + (-rdp).exp_m1() > 0.0 {
        return 0.0;
    }
    if alpha > 1.01 {
        rdp + (1.0 - 1.0 / alpha).ln() - (delta * alpha).ln() / (alpha - 1.0)
    } else {
        f64::INFINITY
    }
}

/// Batch epsilon computation: pre-computes per-step RDP, then vectorises
/// composition + conversion across all step counts.
pub fn compute_epsilon_batch_impl(
    q: f64,
    sigma: f64,
    steps_list: &[i64],
    orders: &[f64],
    delta: f64,
) -> Vec<f64> {
    let rdp_per_step: Vec<f64> = orders
        .iter()
        .map(|&alpha| compute_rdp_single_order(q, sigma, alpha))
        .collect();

    steps_list
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
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_log_a_finite() {
        let result = compute_log_a(0.5, 10.0, 2.0);
        assert!(result.is_finite());
        assert!(result > 0.0);
    }

    #[test]
    fn rdp_zero_q() {
        assert_eq!(compute_rdp_single_order(0.0, 1.0, 5.0), 0.0);
    }

    #[test]
    fn rdp_full_batch() {
        let result = compute_rdp_single_order(1.0, 1.0, 5.0);
        assert!((result - 2.5).abs() < 1e-6);
    }

    #[test]
    fn rdp_infinite_alpha() {
        assert!(compute_rdp_single_order(0.5, 1.0, f64::INFINITY).is_infinite());
    }

    #[test]
    fn rdp_zero_sigma() {
        assert!(compute_rdp_single_order(0.5, 0.0, 5.0).is_infinite());
    }

    #[test]
    fn epsilon_zero_rdp() {
        assert_eq!(rdp_to_epsilon(2.0, 0.0, 1e-5), 0.0);
    }

    #[test]
    fn batch_empty_steps() {
        let res = compute_epsilon_batch_impl(0.01, 1.0, &[], &[2.0], 1e-5);
        assert!(res.is_empty());
    }

    #[test]
    fn batch_monotone_in_steps() {
        let res = compute_epsilon_batch_impl(0.01, 1.0, &[100, 500, 1000], &[2.0, 5.0, 10.0], 1e-5);
        assert!(res[0] < res[1]);
        assert!(res[1] < res[2]);
    }
}