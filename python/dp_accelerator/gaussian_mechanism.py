"""Exact calibration for the Gaussian mechanism.

Implements the analytical formulas from Balle & Wang (arXiv:1805.06530).
All computation runs in Rust for maximum speed.

API-compatible with ``dp_accounting.gaussian_mechanism``.
"""

from dp_accelerator._core import (
    get_epsilon_gaussian as _get_epsilon_gaussian_rust,
    get_sigma_gaussian as _get_sigma_gaussian_rust,
)


def get_epsilon_gaussian(sigma: float, delta: float, tol: float = 1e-12) -> float:
    """Compute epsilon for the Gaussian mechanism.

    Uses the analytical method from https://arxiv.org/pdf/1805.06530.

    Args:
        sigma: Standard deviation of the Gaussian noise (>= 0).
        delta: Target delta (in [0, 1]).
        tol: Error tolerance for root-finding search.

    Returns:
        The smallest non-negative epsilon such that the Gaussian mechanism
        with the given sigma is (epsilon, delta)-DP.

    Raises:
        ValueError: If sigma < 0 or delta is not in [0, 1].
    """
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got sigma={sigma}.")
    if not 0 <= delta <= 1:
        raise ValueError(f"delta must be in [0, 1], got delta={delta}.")
    return _get_epsilon_gaussian_rust(sigma, delta, tol)


def get_sigma_gaussian(epsilon: float, delta: float, tol: float = 1e-12) -> float:
    """Compute the noise std for the Gaussian mechanism.

    Uses the analytical method from https://arxiv.org/pdf/1805.06530.

    Args:
        epsilon: Target epsilon (>= 0).
        delta: Target delta (in [0, 1]).
        tol: Error tolerance for root-finding search.

    Returns:
        The smallest sigma such that the Gaussian mechanism is
        (epsilon, delta)-DP.

    Raises:
        ValueError: If epsilon < 0 or delta is not in [0, 1].
    """
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got epsilon={epsilon}.")
    if not 0 <= delta <= 1:
        raise ValueError(f"delta must be in [0, 1], got delta={delta}.")
    return _get_sigma_gaussian_rust(epsilon, delta, tol)
