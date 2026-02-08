"""RDP (Rényi Differential Privacy) accounting primitives.

Provides Rust-accelerated RDP computation for the Poisson-subsampled Gaussian
mechanism, Laplace, randomized response, zCDP, tree aggregation, sampling
without replacement, and repeat-and-select.

Also provides an ``RdpAccountant`` convenience class that mirrors the
``dp_accounting.rdp.RdpAccountant`` API.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from dp_accelerator._core import (
    compute_rdp_poisson_subsampled_gaussian as _rdp_poisson_gaussian_rust,
    compute_rdp_sample_wor_gaussian as _rdp_sample_wor_rust,
    compute_rdp_laplace as _rdp_laplace_rust,
    rdp_to_epsilon_vec as _rdp_to_eps_rust,
    rdp_to_delta_vec as _rdp_to_delta_rust,
)
from dp_accelerator.dp_event import (
    GaussianDpEvent,
    PoissonSampledDpEvent,
    SelfComposedDpEvent,
    ComposedDpEvent,
    NoOpDpEvent,
)
from dp_accelerator.privacy_accountant import (
    NeighboringRelation,
    PrivacyAccountant,
)

# ── Default RDP orders (same as Google dp_accounting) ─────────────

DEFAULT_RDP_ORDERS: Tuple[float, ...] = tuple(
    np.concatenate(
        (
            np.linspace(1.01, 8, num=50),
            np.arange(8, 64, dtype=float),
            np.linspace(65, 512, num=10, dtype=float),
        )
    ).tolist()
)

# ═══════════════════════════════════════════════════════════════════
#  Low-level RDP computation functions
# ═══════════════════════════════════════════════════════════════════


def compute_rdp_poisson_subsampled_gaussian(
    q: float, sigma: float, orders: Sequence[float]
) -> List[float]:
    """RDP for the Poisson-subsampled Gaussian mechanism (Rust-accelerated)."""
    return _rdp_poisson_gaussian_rust(q, sigma, list(orders))


def compute_rdp_sample_wor_gaussian(
    q: float, sigma: float, orders: Sequence[float]
) -> List[float]:
    """RDP for sampling-without-replacement Gaussian (Rust-accelerated)."""
    return _rdp_sample_wor_rust(q, sigma, list(orders))


def compute_rdp_laplace(pure_eps: float, orders: Sequence[float]) -> List[float]:
    """RDP for the Laplace mechanism (Rust-accelerated)."""
    return _rdp_laplace_rust(pure_eps, list(orders))


def compute_rdp_tree_aggregation(
    sigma: float,
    step_counts: Union[int, List[int]],
    orders: Sequence[float],
) -> List[float]:
    """RDP for single-epoch tree aggregation.

    Follows the analysis from "Practical and Private (Deep) Learning without
    Sampling or Shuffling" (https://arxiv.org/abs/2103.00039).

    The maximum depth of a balanced binary tree is ceil(log2(max(step_counts)+1)),
    and each level contributes alpha / (2 * sigma^2).
    """
    if isinstance(step_counts, int):
        step_counts = [step_counts]

    if not step_counts:
        return [0.0] * len(list(orders))

    max_steps = max(step_counts)
    if max_steps <= 0:
        return [0.0] * len(list(orders))

    if sigma == 0.0:
        return [float("inf")] * len(list(orders))

    max_depth = math.ceil(math.log2(max_steps + 1))
    two_sigma_sq = 2.0 * sigma * sigma

    return [alpha * max_depth / two_sigma_sq for alpha in orders]


def compute_rdp_randomized_response(
    noise_parameter: float,
    num_buckets: int,
    orders: Sequence[float],
    replace_one: bool = False,
) -> List[float]:
    """RDP for randomized response.

    Args:
        noise_parameter: Probability of outputting a random bucket (p).
        num_buckets: Number of possible output buckets (k).
        orders: RDP orders.
        replace_one: If True, use replace-one adjacency; else replace-special.
    """
    p = noise_parameter
    k = num_buckets

    # Trivial cases: no privacy loss
    if p >= 1.0 or k <= 1:
        return [0.0] * len(list(orders))

    result = []
    for alpha in orders:
        if replace_one:
            rdp = _rr_rdp_replace_one(p, k, alpha)
        else:
            rdp = _rr_rdp_replace_special(p, k, alpha)
        result.append(rdp)
    return result


def _rr_rdp_replace_special(p: float, k: int, alpha: float) -> float:
    """RDP for randomized response under replace-special adjacency.

    Matches dp_accounting._randomized_response_rdp_replace_special.
    """
    if alpha <= 1.0:
        return 0.0
    if p >= 1.0 or k <= 1:
        return 0.0
    if p == 0.0:
        return float("inf")

    from scipy.special import logsumexp

    log_1 = math.log(k * (1 - p) + p)
    log_2 = math.log(p)

    if math.isinf(alpha):
        return max(log_1, -log_2)

    # Compute bounds for both orderings
    bound1 = logsumexp(
        [alpha * log_1, alpha * log_2],
        b=[1.0 / k, 1.0 - 1.0 / k],
    )
    bound2 = logsumexp(
        [(1 - alpha) * log_1, (1 - alpha) * log_2],
        b=[1.0 / k, 1.0 - 1.0 / k],
    )
    return float(max(bound1, bound2) / (alpha - 1.0))


def _rr_rdp_replace_one(p: float, k: int, alpha: float) -> float:
    """RDP for randomized response under replace-one adjacency.

    Matches dp_accounting._randomized_response_rdp_replace_one.
    The RDP is the Rényi divergence between:
      P = [ 1-p+p/k, p/k, p/k, ..., p/k ]
      Q = [ p/k, 1-p+p/k, p/k, ..., p/k ]
    """
    if alpha <= 1.0:
        return 0.0
    if p >= 1.0 or k <= 1:
        return 0.0
    if p == 0.0:
        return float("inf")

    from scipy.special import logsumexp

    log_1 = math.log(k / p - k + 1)  # log((k*(1-p)+p) / (p/k)) = log((1-p+p/k)/(p/k))

    if math.isinf(alpha):
        return log_1

    return float(
        logsumexp(
            a=[alpha * log_1, -alpha * log_1, 0.0],
            b=[p / k, 1 - p + p / k, (1 - 2.0 / k) * p],
        )
        / (alpha - 1.0)
    )


def compute_rdp_zcdp(xi: float, rho: float, orders: Sequence[float]) -> List[float]:
    """RDP for a mechanism satisfying (xi, rho)-zCDP.

    RDP(alpha) = xi + rho * alpha.
    """
    return [xi + rho * alpha for alpha in orders]


def compute_rdp_repeat_and_select(
    orders: Sequence[float],
    rdp_single: Sequence[float],
    mean: float,
    shape: float,
) -> List[float]:
    """RDP for repeat-and-select (arXiv:2110.03620).

    Matches dp_accounting._compute_rdp_repeat_and_select.

    Args:
        orders: RDP orders.
        rdp_single: Per-run RDP values (one per order).
        mean: Mean number of repetitions.
        shape: Distribution shape (inf=Poisson, 1=Geometric, 0=Logarithmic).
    """
    orders_arr = np.array(list(orders), dtype=float)
    rdp_arr = np.array(list(rdp_single), dtype=float)
    rdp_out = np.full_like(orders_arr, float("inf"))

    if shape == float("inf"):
        # Poisson distribution
        for i in range(len(orders_arr)):
            if orders_arr[i] <= 1.0:
                continue
            epshat = math.log1p(1.0 / (orders_arr[i] - 1.0))
            deltahat, _ = rdp_to_epsilon(list(orders_arr), list(rdp_arr), epshat)
            # deltahat is actually a delta here — we need compute_delta
            deltahat_val = _compute_delta_for_repeat_select(
                list(orders_arr), list(rdp_arr), epshat
            )
            rdp_out[i] = (
                rdp_arr[i]
                + mean * deltahat_val
                + math.log(mean) / (orders_arr[i] - 1.0)
            )
    else:
        # Truncated Negative Binomial (includes Geometric & Logarithmic)
        gamma = _gamma_truncated_negative_binomial(shape, mean)
        c = (1 + shape) * np.min(
            (1.0 - 1.0 / orders_arr[orders_arr > 1]) * rdp_arr[orders_arr > 1]
            - math.log(gamma) / orders_arr[orders_arr > 1]
        )
        for i in range(len(orders_arr)):
            if orders_arr[i] > 1.0:
                rdp_out[i] = rdp_arr[i] + math.log(mean) / (orders_arr[i] - 1.0) + c
        # Apply monotonicity
        for i in range(len(orders_arr)):
            rdp_out[i] = min(
                rdp_out[j]
                for j in range(len(orders_arr))
                if orders_arr[i] <= orders_arr[j]
            )
    return rdp_out.tolist()


def _compute_delta_for_repeat_select(
    orders: List[float], rdp: List[float], epsilon: float
) -> float:
    """Internal: compute delta from RDP for repeat-and-select."""
    log_deltas = []
    for a, r in zip(orders, rdp):
        if r <= 0:
            log_deltas.append(float("-inf"))
            continue
        log_delta = 0.5 * math.log1p(-math.exp(-r)) if r < 700 else 0.0
        if a > 1.01:
            rdp_bound = (a - 1.0) * (r - epsilon + math.log1p(-1.0 / a)) - math.log(a)
            log_delta = min(log_delta, rdp_bound)
        log_deltas.append(log_delta)
    best = min(log_deltas)
    return min(math.exp(best), 1.0)


def _expm1_over_x(x: float) -> float:
    """Compute (exp(x) - 1) / x stably."""
    if abs(x) < 1e-8:
        return 1.0 + x / 2.0 + x * x / 6.0
    return math.expm1(x) / x


def _logx_over_xm1(x: float) -> float:
    """Compute log(x) / (x - 1) stably."""
    if abs(x - 1.0) < 1e-8:
        return 1.0 - (x - 1.0) / 2.0
    return math.log(x) / (x - 1.0)


def _truncated_negative_binomial_mean(gamma: float, shape: float) -> float:
    """Mean of the truncated negative binomial distribution."""
    if shape == 0:
        return -1.0 / math.log1p(-gamma)
    return shape * gamma * _expm1_over_x(shape * math.log1p(-gamma)) / (1 - gamma)


def _gamma_truncated_negative_binomial(shape: float, mean: float) -> float:
    """Find gamma for the truncated negative binomial with given mean."""
    # Binary search for gamma in (0, 1)
    lo, hi = 1e-15, 1.0 - 1e-15
    for _ in range(200):
        mid = (lo + hi) / 2.0
        m = _truncated_negative_binomial_mean(mid, shape)
        if m < mean:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ═══════════════════════════════════════════════════════════════════
#  RDP ↔ (ε,δ) conversions
# ═══════════════════════════════════════════════════════════════════


def rdp_to_epsilon(
    orders: Sequence[float],
    rdp_values: Sequence[float],
    delta: float,
) -> Tuple[float, float]:
    """Convert RDP guarantees to (epsilon, delta)-DP (Rust-accelerated).

    Returns:
        (epsilon, optimal_order).
    """
    return _rdp_to_eps_rust(list(orders), list(rdp_values), delta)


def rdp_to_delta(
    orders: Sequence[float],
    rdp_values: Sequence[float],
    epsilon: float,
) -> Tuple[float, float]:
    """Convert RDP guarantees to delta for a given epsilon (Rust-accelerated).

    Returns:
        (delta, optimal_order).
    """
    return _rdp_to_delta_rust(list(orders), list(rdp_values), epsilon)


# ═══════════════════════════════════════════════════════════════════
#  RdpAccountant
# ═══════════════════════════════════════════════════════════════════


class RdpAccountant(PrivacyAccountant):
    """Rust-accelerated RDP privacy accountant.

    API-compatible with ``dp_accounting.rdp.RdpAccountant``.
    """

    def __init__(
        self,
        orders: Optional[Sequence[float]] = None,
        neighboring_relation: NeighboringRelation = NeighboringRelation.ADD_OR_REMOVE_ONE,
    ):
        super().__init__(neighboring_relation)
        self._orders = list(orders) if orders is not None else list(DEFAULT_RDP_ORDERS)
        # Accumulated RDP values, one per order
        self._rdp = [0.0] * len(self._orders)

    # ── Composition helpers (direct API) ──────────────────────────

    def compose_poisson_subsampled_gaussian(
        self, q: float, sigma: float, count: int = 1
    ) -> "RdpAccountant":
        """Compose Poisson-subsampled Gaussian mechanisms."""
        rdp_per_step = compute_rdp_poisson_subsampled_gaussian(q, sigma, self._orders)
        for i in range(len(self._rdp)):
            self._rdp[i] += rdp_per_step[i] * count
        return self

    # ── DpEvent-based composition ─────────────────────────────────

    def _maybe_compose(self, event, count, do_compose):
        """Traverse and optionally compose a DpEvent."""
        if isinstance(event, NoOpDpEvent):
            return None

        if isinstance(event, GaussianDpEvent):
            if do_compose:
                rdp = [
                    alpha / (2.0 * event.noise_multiplier**2) for alpha in self._orders
                ]
                for i in range(len(self._rdp)):
                    self._rdp[i] += rdp[i] * count
            return None

        if isinstance(event, PoissonSampledDpEvent):
            if isinstance(event.event, GaussianDpEvent):
                if do_compose:
                    rdp = compute_rdp_poisson_subsampled_gaussian(
                        event.sampling_probability,
                        event.event.noise_multiplier,
                        self._orders,
                    )
                    for i in range(len(self._rdp)):
                        self._rdp[i] += rdp[i] * count
                return None

        if isinstance(event, SelfComposedDpEvent):
            return self._maybe_compose(event.event, count * event.count, do_compose)

        if isinstance(event, ComposedDpEvent):
            for sub in event.events:
                err = self._maybe_compose(sub, count, do_compose)
                if err is not None:
                    return err
            return None

        return self.CompositionErrorDetails(
            invalid_event=event,
            error_message=f"Unsupported event type: {type(event).__name__}",
        )

    # ── Epsilon / delta queries ───────────────────────────────────

    def get_epsilon(self, target_delta: float = 1e-5) -> float:
        """Compute the best epsilon for the given delta."""
        eps, _ = rdp_to_epsilon(self._orders, self._rdp, target_delta)
        return eps

    def get_delta(self, target_epsilon: float) -> float:
        """Compute the best delta for the given epsilon."""
        delta, _ = rdp_to_delta(self._orders, self._rdp, target_epsilon)
        return delta

    @property
    def orders(self):
        return self._orders
