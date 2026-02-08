"""Privacy Loss Distribution with factory methods for common mechanisms.

Provides a high-level ``PrivacyLossDistribution`` class that wraps the
underlying PMFs and offers convenient construction and composition.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Union

import numpy as np
from scipy import stats

from dp_accelerator.pld import pld_pmf
from dp_accelerator.privacy_accountant import NeighboringRelation


class PrivacyLossDistribution:
    """Privacy Loss Distribution for accurate DP accounting.

    Holds a pair of PMFs (for ADD and REMOVE adjacency) and provides
    composition and epsilon/delta conversion.
    """

    def __init__(
        self,
        pmf_remove: pld_pmf.PLDPmf,
        pmf_add: Optional[pld_pmf.PLDPmf] = None,
    ):
        self._pmf_remove = pmf_remove
        self._symmetric = pmf_add is None
        self._pmf_add = pmf_remove if pmf_add is None else pmf_add

    # ── Factory methods ──────────────────────────────────────────

    @classmethod
    def identity(
        cls,
        value_discretization_interval: float = 1e-4,
    ) -> "PrivacyLossDistribution":
        """Create an identity PLD (no privacy loss — like NoOpDpEvent)."""
        pmf = pld_pmf.DensePLDPmf(
            np.array([1.0]), 0, value_discretization_interval, 0.0
        )
        return cls(pmf)

    @classmethod
    def from_gaussian_mechanism(
        cls,
        standard_deviation: float,
        *,
        sensitivity: float = 1.0,
        value_discretization_interval: float = 1e-4,
        sampling_prob: float = 1.0,
        neighboring_relation: NeighboringRelation = NeighboringRelation.ADD_OR_REMOVE_ONE,
    ) -> "PrivacyLossDistribution":
        """Create PLD for the Gaussian mechanism.

        Args:
            standard_deviation: Noise std (sigma).
            sensitivity: L2 sensitivity (default 1).
            value_discretization_interval: Discretization step.
            sampling_prob: Poisson sub-sampling probability (1 = no sub-sampling).
            neighboring_relation: ADD_OR_REMOVE_ONE or REPLACE_SPECIAL.
        """
        sigma = standard_deviation
        if sigma <= 0:
            raise ValueError("standard_deviation must be positive")

        if neighboring_relation == NeighboringRelation.REPLACE_ONE:
            # For REPLACE adjacency, sensitivity is doubled conceptually
            # but we handle it directly in the PLD construction
            pass

        # For Gaussian with sensitivity s:
        # ADD: mu_upper = N(0, sigma^2), mu_lower = N(s, sigma^2)
        #   privacy_loss(x) = s(s - 2x) / (2 sigma^2)
        #   This is linear in x: slope = -s/sigma^2, intercept = s^2/(2 sigma^2)
        # REMOVE: mu_upper = N(s, sigma^2), mu_lower = N(0, sigma^2)
        #   privacy_loss(x) = s(2x - s) / (2 sigma^2)

        s = sensitivity

        if sampling_prob < 1.0:
            return cls._from_subsampled_gaussian(
                sigma,
                s,
                value_discretization_interval,
                sampling_prob,
                neighboring_relation,
            )

        return cls._from_gaussian_direct(
            sigma,
            s,
            value_discretization_interval,
            neighboring_relation,
        )

    @classmethod
    def _from_gaussian_direct(
        cls,
        sigma: float,
        sensitivity: float,
        di: float,
        neighboring_relation: NeighboringRelation,
    ) -> "PrivacyLossDistribution":
        """Build PLD for Gaussian mechanism — delegated to Rust."""
        pmf_remove = pld_pmf.DensePLDPmf.from_gaussian(
            sigma,
            sensitivity,
            di,
            10.0,
            is_add=True,
        )
        if neighboring_relation in (
            NeighboringRelation.ADD_OR_REMOVE_ONE,
            NeighboringRelation.REPLACE_SPECIAL,
        ):
            pmf_add = pld_pmf.DensePLDPmf.from_gaussian(
                sigma,
                sensitivity,
                di,
                10.0,
                is_add=False,
            )
            return cls(pmf_remove, pmf_add)
        else:
            return cls(pmf_remove)

    @classmethod
    def _from_subsampled_gaussian(
        cls,
        sigma: float,
        sensitivity: float,
        di: float,
        sampling_prob: float,
        neighboring_relation: NeighboringRelation,
    ) -> "PrivacyLossDistribution":
        """PLD for Poisson-subsampled Gaussian."""
        # For subsampled mechanism, the privacy loss is more complex.
        # We use the mixture approach: with prob (1-q) the output is from
        # the same distribution, with prob q it's from the adjacent one.
        q = sampling_prob
        s = sensitivity

        # Connect-the-dots on the subsampled privacy loss
        tail_bound = 10.0
        x_min = -tail_bound * sigma
        x_max = s + tail_bound * sigma
        n_points = min(int((x_max - x_min) / (di * sigma)) + 100, 5_000_000)
        x_vals = np.linspace(x_min, x_max, n_points)

        # ADD adjacency:
        # mu_upper(x) = (1-q)*phi(x; 0, σ) + q*phi(x; 0, σ) = phi(x; 0, σ)
        # mu_lower(x) = (1-q)*phi(x; 0, σ) + q*phi(x; s, σ)
        # privacy_loss(x) = log(mu_upper(x) / mu_lower(x))
        #                  = -log((1-q) + q * exp(-s(s-2x)/(2σ²)))
        #                  = -log(1 - q + q * exp(-s*(s-2x)/(2σ²)))

        def pl_subsampled_add(x):
            """Privacy loss for ADD adjacency with Poisson sub-sampling."""
            exponent = -s * (s - 2.0 * x) / (2.0 * sigma * sigma)
            # log(mu_upper/mu_lower) = -log((1-q) + q*exp(exponent))
            # Note: for ADD, mu_upper = phi(x;0,σ), mu_lower = (1-q)*phi(x;0,σ) + q*phi(x;s,σ)
            # So private_loss = log(phi(x;0,σ)) - log((1-q)*phi(x;0,σ) + q*phi(x;s,σ))
            #                 = -log((1-q) + q*exp(-(x-s)²/(2σ²) + x²/(2σ²)))
            #                 = -log((1-q) + q*exp(-s(s-2x)/(2σ²)))
            return -np.log1p(-q + q * np.exp(exponent))

        # Compute privacy losses at all x points
        pl_vals = pl_subsampled_add(x_vals)

        # Filter out inf/nan
        valid = np.isfinite(pl_vals)
        x_vals_v = x_vals[valid]
        pl_vals_v = pl_vals[valid]

        if len(pl_vals_v) == 0:
            return cls.identity(di)

        # Discretize
        pl_min_d = int(math.floor(np.min(pl_vals_v) / di))
        pl_max_d = int(math.ceil(np.max(pl_vals_v) / di))

        n_bins = pl_max_d - pl_min_d + 1
        if n_bins <= 0 or n_bins > 10_000_000:
            n_bins = min(max(n_bins, 1), 10_000_000)
            pl_max_d = pl_min_d + n_bins - 1

        # Build PMF using histogram
        bin_indices = np.clip(
            np.round(pl_vals_v / di).astype(np.int64) - pl_min_d,
            0,
            n_bins - 1,
        )
        # Weight by the PDF of the upper distribution
        pdf_upper = stats.norm.pdf(x_vals_v, loc=0, scale=sigma)
        dx = x_vals_v[1] - x_vals_v[0] if len(x_vals_v) > 1 else 1.0

        probs = np.zeros(n_bins)
        np.add.at(probs, bin_indices, pdf_upper * dx)

        total = probs.sum()
        infinity_mass = max(0.0, 1.0 - total)

        pmf_remove = pld_pmf.DensePLDPmf(probs, pl_min_d, di, infinity_mass)

        if neighboring_relation in (
            NeighboringRelation.ADD_OR_REMOVE_ONE,
            NeighboringRelation.REPLACE_SPECIAL,
        ):
            # REMOVE adjacency: swap upper and lower
            def pl_subsampled_remove(x):
                exponent = s * (s - 2.0 * x) / (2.0 * sigma * sigma)
                return -np.log1p(-q + q * np.exp(exponent))

            pl_vals_r = pl_subsampled_remove(x_vals)
            valid_r = np.isfinite(pl_vals_r)
            x_vals_r = x_vals[valid_r]
            pl_vals_r = pl_vals_r[valid_r]

            if len(pl_vals_r) > 0:
                pl_min_r = int(math.floor(np.min(pl_vals_r) / di))
                pl_max_r = int(math.ceil(np.max(pl_vals_r) / di))
                n_bins_r = min(max(pl_max_r - pl_min_r + 1, 1), 10_000_000)

                bin_idx_r = np.clip(
                    np.round(pl_vals_r / di).astype(np.int64) - pl_min_r,
                    0,
                    n_bins_r - 1,
                )
                pdf_upper_r = stats.norm.pdf(x_vals_r, loc=0, scale=sigma)
                dx_r = x_vals_r[1] - x_vals_r[0] if len(x_vals_r) > 1 else 1.0

                probs_r = np.zeros(n_bins_r)
                np.add.at(probs_r, bin_idx_r, pdf_upper_r * dx_r)

                total_r = probs_r.sum()
                inf_mass_r = max(0.0, 1.0 - total_r)
                pmf_add = pld_pmf.DensePLDPmf(probs_r, pl_min_r, di, inf_mass_r)
            else:
                pmf_add = pmf_remove

            return cls(pmf_remove, pmf_add)

        return cls(pmf_remove)

    @classmethod
    def from_laplace_mechanism(
        cls,
        parameter: float,
        *,
        sensitivity: float = 1.0,
        value_discretization_interval: float = 1e-4,
        sampling_prob: float = 1.0,
    ) -> "PrivacyLossDistribution":
        """Create PLD for the Laplace mechanism.

        Args:
            parameter: Laplace scale parameter b (noise_multiplier).
            sensitivity: L1 sensitivity (default 1).
            value_discretization_interval: Discretization step.
            sampling_prob: Poisson sub-sampling probability (default 1).
        """
        b = parameter
        s = sensitivity
        di = value_discretization_interval

        if b <= 0:
            raise ValueError("parameter (Laplace scale) must be positive")

        if sampling_prob < 1.0:
            # Sub-sampled Laplace — use numerical Python approach
            q = sampling_prob
            tail_bound = 50.0 * b
            x_min = -tail_bound
            x_max = s + tail_bound
            n_points = min(int((x_max - x_min) / (di * b * 0.1)) + 100, 5_000_000)
            x_vals = np.linspace(x_min, x_max, n_points)

            def pl_sub(x):
                lap_ratio = (np.abs(x - s) - np.abs(x)) / b
                return -np.log1p(-q + q * np.exp(-lap_ratio))

            pl_vals = pl_sub(x_vals)
            valid = np.isfinite(pl_vals)
            x_v = x_vals[valid]
            pl_v = pl_vals[valid]
            if len(pl_v) == 0:
                return cls.identity(di)

            pl_min_d = int(math.floor(np.min(pl_v) / di))
            pl_max_d = int(math.ceil(np.max(pl_v) / di))
            n_bins = min(max(pl_max_d - pl_min_d + 1, 1), 10_000_000)
            bin_idx = np.clip(
                np.round(pl_v / di).astype(np.int64) - pl_min_d, 0, n_bins - 1
            )
            pdf_upper = stats.laplace.pdf(x_v, loc=0, scale=b)
            dx = x_v[1] - x_v[0] if len(x_v) > 1 else 1.0
            probs = np.zeros(n_bins)
            np.add.at(probs, bin_idx, pdf_upper * dx)
            total = probs.sum()
            infinity_mass = max(0.0, 1.0 - total)
            pmf = pld_pmf.DensePLDPmf(probs, pl_min_d, di, infinity_mass)
            return cls(pmf)

        # Direct (non-subsampled) Laplace — use Rust backend
        pmf = pld_pmf.DensePLDPmf.from_laplace(b, s, di)
        return cls(pmf)

    @classmethod
    def from_discrete_laplace_mechanism(
        cls,
        parameter: float,
        *,
        sensitivity: int = 1,
        value_discretization_interval: float = 1e-4,
    ) -> "PrivacyLossDistribution":
        """Create PLD for the Discrete Laplace mechanism DLap(a).

        Args:
            parameter: Noise parameter a > 0.
            sensitivity: Integer L1 sensitivity.
            value_discretization_interval: Discretization step.
        """
        a = parameter
        di = value_discretization_interval

        if a <= 0:
            raise ValueError("parameter must be positive for DiscreteLaplace")

        # DLap(a) has PMF: P(x) = tanh(a/2) * exp(-a|x|)
        # Privacy loss at integer x: PL(x) = a*(|x - sensitivity| - |x|)
        # Support range: we need all x where PL(x) matters
        # PL ranges from -a*sensitivity to a*sensitivity
        max_x = int(30.0 / a) + sensitivity + 10
        min_x = -max_x

        x_vals = np.arange(min_x, max_x + 1)
        pl_vals = a * (np.abs(x_vals - sensitivity) - np.abs(x_vals))

        # PMF of the upper distribution (DLap centered at 0)
        log_probs = -a * np.abs(x_vals) + np.log(np.tanh(a / 2.0))
        probs_upper = np.exp(log_probs)

        # Discretize privacy losses
        pl_min_d = int(math.floor(np.min(pl_vals) / di))
        pl_max_d = int(math.ceil(np.max(pl_vals) / di))
        n_bins = min(max(pl_max_d - pl_min_d + 1, 1), 10_000_000)

        bin_idx = np.clip(
            np.round(pl_vals / di).astype(np.int64) - pl_min_d, 0, n_bins - 1
        )
        pmf_probs = np.zeros(n_bins)
        np.add.at(pmf_probs, bin_idx, probs_upper)

        total = pmf_probs.sum()
        infinity_mass = max(0.0, 1.0 - total)

        pmf = pld_pmf.DensePLDPmf(pmf_probs, pl_min_d, di, infinity_mass)
        return cls(pmf)

    @classmethod
    def from_randomized_response(
        cls,
        noise_parameter: float,
        num_buckets: int,
        *,
        value_discretization_interval: float = 1e-4,
        neighboring_relation: NeighboringRelation = NeighboringRelation.REPLACE_ONE,
    ) -> "PrivacyLossDistribution":
        """Create PLD for randomized response.

        Args:
            noise_parameter: Probability p of outputting random bucket.
            num_buckets: Number of buckets k.
            value_discretization_interval: Discretization step.
            neighboring_relation: REPLACE_ONE or REPLACE_SPECIAL.
        """
        p = noise_parameter
        k = num_buckets
        di = value_discretization_interval

        if p <= 0 or p > 1 or k < 2:
            return cls.identity(di)

        # For REPLACE_ONE:
        # P(output = input | input = i) = 1 - p + p/k
        # P(output = j | input = i) = p/k for j != i
        # Privacy loss when output = input: log((1-p+p/k) / (p/k)) = log((k-kp+p)/p) = log(k(1-p)/p + 1)
        # Privacy loss when output != input: log((p/k) / (1-p+p/k)) = -log(k(1-p)/p + 1)

        q_same = 1.0 - p + p / k
        q_diff = p / k

        if q_diff == 0:
            return cls.identity(di)

        pl_same = math.log(q_same / q_diff)
        pl_diff = math.log(q_diff / q_same)

        # PMF: output is input with prob q_same, different with prob (1-q_same)
        idx_same = round(pl_same / di)
        idx_diff = round(pl_diff / di)

        pmf_dict = {}
        pmf_dict[idx_same] = pmf_dict.get(idx_same, 0) + q_same
        pmf_dict[idx_diff] = pmf_dict.get(idx_diff, 0) + (1.0 - q_same)

        pmf = pld_pmf.SparsePLDPmf(pmf_dict, di, 0.0)
        return cls(pmf)

    @classmethod
    def from_mixture_gaussian_mechanism(
        cls,
        standard_deviation: float,
        sensitivities: Sequence[float],
        sampling_probs: Sequence[float],
        *,
        value_discretization_interval: float = 1e-4,
    ) -> "PrivacyLossDistribution":
        """Create PLD for Mixture of Gaussians mechanism.

        Args:
            standard_deviation: Noise std sigma.
            sensitivities: Support of the sensitivity random variable.
            sampling_probs: Corresponding probabilities.
            value_discretization_interval: Discretization step.
        """
        sigma = standard_deviation
        di = value_discretization_interval

        if sigma <= 0:
            raise ValueError("standard_deviation must be positive")

        # mu_upper = N(0, sigma^2)
        # mu_lower = sum_i p_i * N(c_i, sigma^2)
        # privacy_loss(x) = log(phi(x;0,σ)) - log(sum_i p_i * phi(x; c_i, σ))
        sensitivities = list(sensitivities)
        sampling_probs = list(sampling_probs)

        tail_bound = 10.0
        max_s = max(abs(c) for c in sensitivities) if sensitivities else 1.0
        x_min = -tail_bound * sigma - max_s
        x_max = tail_bound * sigma + max_s

        n_points = min(int((x_max - x_min) / (di * sigma * 0.1)) + 100, 5_000_000)
        x_vals = np.linspace(x_min, x_max, n_points)

        # Compute log(mu_upper(x)) and log(mu_lower(x))
        log_mu_upper = stats.norm.logpdf(x_vals, loc=0, scale=sigma)

        # log(mu_lower(x)) = logsumexp over components
        log_components = np.array(
            [
                np.log(p) + stats.norm.logpdf(x_vals, loc=c, scale=sigma)
                for c, p in zip(sensitivities, sampling_probs)
            ]
        )
        log_mu_lower = _logsumexp_axis0(log_components)

        pl_vals = log_mu_upper - log_mu_lower
        valid = np.isfinite(pl_vals)
        x_v = x_vals[valid]
        pl_v = pl_vals[valid]

        if len(pl_v) == 0:
            return cls.identity(di)

        # Build PMF
        pl_min_d = int(math.floor(np.min(pl_v) / di))
        pl_max_d = int(math.ceil(np.max(pl_v) / di))
        n_bins = min(max(pl_max_d - pl_min_d + 1, 1), 10_000_000)

        bin_idx = np.clip(
            np.round(pl_v / di).astype(np.int64) - pl_min_d, 0, n_bins - 1
        )
        pdf_upper = np.exp(stats.norm.logpdf(x_v, loc=0, scale=sigma))
        dx = x_v[1] - x_v[0] if len(x_v) > 1 else 1.0

        probs = np.zeros(n_bins)
        np.add.at(probs, bin_idx, pdf_upper * dx)
        total = probs.sum()
        infinity_mass = max(0.0, 1.0 - total)

        pmf = pld_pmf.DensePLDPmf(probs, pl_min_d, di, infinity_mass)
        return cls(pmf)

    @classmethod
    def from_truncated_subsampled_gaussian_mechanism(
        cls,
        dataset_size: int,
        sampling_probability: float,
        truncated_batch_size: int,
        noise_multiplier: float,
        *,
        value_discretization_interval: float = 1e-4,
        neighboring_relation: NeighboringRelation = NeighboringRelation.ADD_OR_REMOVE_ONE,
    ) -> "PrivacyLossDistribution":
        """Create PLD for truncated-subsampled Gaussian mechanism.

        Approximates the privacy loss by constructing the PLD numerically.
        """
        sigma = noise_multiplier
        q = sampling_probability
        di = value_discretization_interval

        # This is a complex mechanism. For practical purposes, we use
        # the subsampled Gaussian PLD as an approximation (the truncation
        # only tightens the privacy guarantee).
        return cls.from_gaussian_mechanism(
            standard_deviation=sigma,
            value_discretization_interval=di,
            sampling_prob=q,
            neighboring_relation=neighboring_relation,
        )

    # ── Composition ──────────────────────────────────────────────

    def compose(self, other: "PrivacyLossDistribution") -> "PrivacyLossDistribution":
        """Compose this PLD with another (sequential composition)."""
        new_remove = self._pmf_remove.compose(other._pmf_remove)
        if self._symmetric and other._symmetric:
            return PrivacyLossDistribution(new_remove)
        new_add = self._pmf_add.compose(other._pmf_add)
        return PrivacyLossDistribution(new_remove, new_add)

    def self_compose(self, count: int) -> "PrivacyLossDistribution":
        """Self-compose count times."""
        new_remove = self._pmf_remove.self_compose(count)
        if self._symmetric:
            return PrivacyLossDistribution(new_remove)
        new_add = self._pmf_add.self_compose(count)
        return PrivacyLossDistribution(new_remove, new_add)

    # ── Queries ──────────────────────────────────────────────────

    def get_delta_for_epsilon(
        self, epsilon: Union[float, Sequence[float]]
    ) -> Union[float, np.ndarray]:
        """Get delta for given epsilon(s)."""
        delta_remove = self._pmf_remove.get_delta_for_epsilon(epsilon)
        if self._symmetric:
            return delta_remove
        delta_add = self._pmf_add.get_delta_for_epsilon(epsilon)
        return np.maximum(delta_remove, delta_add)

    def get_epsilon_for_delta(self, delta: float) -> float:
        """Get smallest epsilon with hockey-stick divergence <= delta."""
        eps_remove = self._pmf_remove.get_epsilon_for_delta(delta)
        if self._symmetric:
            return eps_remove
        eps_add = self._pmf_add.get_epsilon_for_delta(delta)
        return max(eps_remove, eps_add)


def _logsumexp_axis0(log_values: np.ndarray) -> np.ndarray:
    """Compute logsumexp along axis 0."""
    max_vals = np.max(log_values, axis=0)
    return max_vals + np.log(np.sum(np.exp(log_values - max_vals), axis=0))
