"""
JAX Privacy Adapter for Rusty DP Engine

This module provides drop-in replacements for JAX Privacy's accounting functions,
using the high-performance Rust backend.
"""

from . import DPSGDAccountant


def compute_epsilon(
    noise_multiplier: float,
    batch_size: int,
    dataset_size: int,
    num_steps: int,
    delta: float = 1e-5,
) -> float:
    """
    Drop-in replacement for JAX Privacy's compute_epsilon function.

    Args:
        noise_multiplier: The noise multiplier (sigma) added to gradients
        batch_size: Size of each training batch
        dataset_size: Total size of the training dataset
        num_steps: Number of training steps
        delta: Target delta for (epsilon, delta)-DP

    Returns:
        The epsilon value that satisfies (epsilon, delta)-DP
    """
    accountant = DPSGDAccountant(noise_multiplier, batch_size, dataset_size)
    return accountant.get_epsilon(num_steps, delta)


def compute_rdp_curve(
    noise_multiplier: float, batch_size: int, dataset_size: int, num_steps: int
) -> dict:
    """
    Compute the full RDP curve for analysis.

    Args:
        noise_multiplier: The noise multiplier (sigma) added to gradients
        batch_size: Size of each training batch
        dataset_size: Total size of the training dataset
        num_steps: Number of training steps

    Returns:
        Dictionary mapping RDP orders to RDP values
    """
    accountant = DPSGDAccountant(noise_multiplier, batch_size, dataset_size)
    return accountant.get_rdp_curve(num_steps)
