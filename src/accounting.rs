use crate::math::{log_add, log_comb, log_erfc};
use std::f64::consts::SQRT_2;

const MAX_STEPS_LOG_A_FRAC: i32 = 10000;

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

// ── Sampling without replacement ─────────────────────────────────

/// Returns (sign, log|exp(logx) - exp(logy)|)
fn log_sub_sign(logx: f64, logy: f64) -> (bool, f64) {
    if logx > logy {
        (true, logx + (-(logy - logx).exp()).ln_1p())
    } else if logx < logy {
        (false, logy + (-(logx - logy).exp()).ln_1p())
    } else {
        (true, f64::NEG_INFINITY)
    }
}

/// In-place forward difference in log space with signs.
fn stable_inplace_diff_in_log(vec: &mut [f64], signs: &mut [bool], n: usize) {
    for j in 0..n {
        if signs[j] == signs[j + 1] {
            let (s, mag) = log_sub_sign(vec[j + 1], vec[j]);
            let new_sign = if signs[j + 1] { s } else { !s };
            signs[j] = new_sign;
            vec[j] = mag;
        } else {
            vec[j] = log_add(vec[j], vec[j + 1]);
            signs[j] = signs[j + 1];
        }
    }
}

/// Compute forward differences of cgf(x) = x*(x+1)/(2*sigma^2).
/// Returns the log-abs deltas array.
fn get_forward_diffs_cgf(sigma: f64, n: usize) -> Vec<f64> {
    let two_sig_sq = 2.0 * sigma * sigma;
    let size = n + 3;

    let mut func_vec = vec![0.0_f64; size];
    let mut signs = vec![true; size];

    for (i, item) in func_vec.iter_mut().enumerate().skip(1) {
        let x = (i - 1) as f64;
        *item = x * (x + 1.0) / two_sig_sq;
    }

    let mut deltas = vec![0.0_f64; n + 2];

    for (i, item) in deltas.iter_mut().enumerate() {
        stable_inplace_diff_in_log(&mut func_vec, &mut signs, n + 2 - i);
        *item = func_vec[0];
    }

    deltas
}

/// RDP for sampling-without-replacement Gaussian mechanism (single integer order).
/// Returns log(A) (NOT divided by alpha-1).
fn compute_rdp_sample_wor_gaussian_int(q: f64, sigma: f64, alpha: usize) -> f64 {
    if alpha <= 1 {
        return 0.0;
    }

    let two_sig_sq = 2.0 * sigma * sigma;

    let func2 = 2.0 / two_sig_sq; // = 1/sigma^2
                                  // log_f2m1 = func(2) + log(1 - exp(-func(2)))
    let log_f2m1 = func2 + (-((-func2).exp())).ln_1p();

    let max_alpha = 256;
    let mut log_a = 0.0_f64; // log(1) = 0

    if alpha <= max_alpha {
        let deltas = get_forward_diffs_cgf(sigma, alpha);

        for i in 2..=alpha {
            let i_f = i as f64;
            let alpha_f = alpha as f64;
            let s = if i == 2 {
                let a = (4.0_f64).ln() + log_f2m1;
                let b = func2 + (2.0_f64).ln();
                2.0 * q.ln() + log_comb(alpha_f, 2.0) + a.min(b)
            } else {
                let delta_lo_idx = 2 * (i / 2) - 1;
                let delta_hi_idx = 2 * i.div_ceil(2) - 1;
                let delta_lo = deltas[delta_lo_idx];
                let delta_hi = deltas[delta_hi_idx];
                let a = (4.0_f64).ln() + 0.5 * (delta_lo + delta_hi);
                let b = (2.0_f64).ln() + i_f * (i_f - 1.0) / two_sig_sq; // cgf(i-1) = (i-1)*i/(2*sigma^2)
                a.min(b) + i_f * q.ln() + log_comb(alpha_f, i_f)
            };
            log_a = log_add(log_a, s);
        }
    } else {
        for i in 2..=alpha {
            let i_f = i as f64;
            let alpha_f = alpha as f64;
            let s = if i == 2 {
                let a = (4.0_f64).ln() + log_f2m1;
                let b = func2 + (2.0_f64).ln();
                2.0 * q.ln() + log_comb(alpha_f, 2.0) + a.min(b)
            } else {
                let cgf_im1 = (i_f - 1.0) * i_f / two_sig_sq;
                (2.0_f64).ln() + cgf_im1 + i_f * q.ln() + log_comb(alpha_f, i_f)
            };
            log_a = log_add(log_a, s);
        }
    }
    log_a
}

/// RDP for sampling-without-replacement Gaussian mechanism (single order).
pub fn compute_rdp_sample_wor_single(q: f64, sigma: f64, alpha: f64) -> f64 {
    if q == 0.0 {
        return 0.0;
    }
    if q == 1.0 {
        return alpha / (2.0 * sigma * sigma);
    }
    if alpha.is_infinite() {
        return f64::INFINITY;
    }
    if alpha < 1.0 {
        return f64::INFINITY;
    }

    if alpha == alpha.floor() && alpha.is_finite() {
        let a = alpha as usize;
        if a <= 1 {
            return 0.0;
        }
        compute_rdp_sample_wor_gaussian_int(q, sigma, a) / (alpha - 1.0)
    } else {
        // Interpolate between floor and ceil
        let alpha_f = alpha.floor() as usize;
        let alpha_c = alpha.ceil() as usize;
        if alpha_f <= 1 && alpha_c <= 1 {
            return 0.0;
        }
        let x = if alpha_f <= 1 {
            0.0
        } else {
            compute_rdp_sample_wor_gaussian_int(q, sigma, alpha_f)
        };
        let y = if alpha_c <= 1 {
            0.0
        } else {
            compute_rdp_sample_wor_gaussian_int(q, sigma, alpha_c)
        };
        let t = alpha - alpha.floor();
        ((1.0 - t) * x + t * y) / (alpha - 1.0)
    }
}

// ── Laplace mechanism ────────────────────────────────────────────

/// RDP for the Laplace mechanism with pure-DP parameter `pure_eps`.
/// Matches dp_accounting._laplace_rdp exactly.
pub fn laplace_rdp(pure_eps: f64, alpha: f64) -> f64 {
    if pure_eps == 0.0 {
        return 0.0;
    }
    if alpha == 1.0 {
        // KL divergence: eps + exp(-eps) - 1
        return pure_eps + (-pure_eps).exp() - 1.0;
    }
    if alpha.is_infinite() {
        return pure_eps;
    }
    if alpha < 1.0 {
        return f64::INFINITY;
    }
    let a = alpha;
    let e = pure_eps;
    if a <= 1.1 {
        // For alpha near 1, use series expansion for numerical stability
        // c = expm1(eps*(1-2a)) / (2a-1)
        // v = -c * (a-1)
        // result = eps + c * sum(v^(k-1)/k for k=1..99)
        let c = (e * (1.0 - 2.0 * a)).exp_m1() / (2.0 * a - 1.0);
        let v = -c * (a - 1.0);
        let mut series_sum = 0.0_f64;
        let mut v_pow = 1.0_f64; // v^(k-1) starting at k=1
        for k in 1..100 {
            series_sum += v_pow / (k as f64);
            v_pow *= v;
        }
        (e + c * series_sum).max(0.0)
    } else {
        // Standard formula for alpha > 1.1:
        // eps + log1p((a-1) * expm1(eps*(1-2a)) / (2a-1)) / (a-1)
        let inner = (a - 1.0) * (e * (1.0 - 2.0 * a)).exp_m1() / (2.0 * a - 1.0);
        (e + inner.ln_1p() / (a - 1.0)).max(0.0)
    }
}

// ── RDP to epsilon / delta conversions ───────────────────────────

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

/// Converts a single (order, rdp) pair to delta.
/// Matches dp_accounting compute_delta: uses two bounds and takes the tighter.
pub fn rdp_to_delta_single(alpha: f64, rdp: f64, epsilon: f64) -> f64 {
    if alpha < 1.0 {
        return 1.0;
    }
    if rdp <= 0.0 {
        return 0.0;
    }

    // Bound 1: basic from the definition of Rényi divergence
    // log_delta = 0.5 * log(1 - exp(-rdp))
    let log_delta_basic = 0.5 * (-((-rdp).exp())).ln_1p();

    // Bound 2: improved bound for alpha > 1.01 (Balle et al., 2020)
    // log_delta = (alpha - 1) * (rdp - epsilon + log(1 - 1/alpha)) - log(alpha)
    let log_delta_improved = if alpha > 1.01 {
        (alpha - 1.0) * (rdp - epsilon + (1.0 - 1.0 / alpha).ln()) - alpha.ln()
    } else {
        log_delta_basic
    };

    let log_delta = log_delta_basic.min(log_delta_improved);
    log_delta.exp().min(1.0)
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
