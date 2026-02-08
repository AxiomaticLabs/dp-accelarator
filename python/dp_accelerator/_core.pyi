"""
Type hints for the Rust _core module.
Provides VS Code autocompletion and type checking.
"""

from typing import List

def compute_rdp(q: float, noise_multiplier: float, steps: int, orders: List[float]) -> List[float]:
    """
    Computes RDP (Rényi Differential Privacy) for subsampled Gaussian mechanism.

    Args:
        q: Sampling probability (batch_size / dataset_size)
        noise_multiplier: The noise multiplier (sigma)
        steps: Number of composition steps
        orders: RDP orders (alpha values) to compute

    Returns:
        List of RDP values, one for each order
    """
    ...

def get_epsilon(rdp_values: List[float], orders: List[float], delta: float) -> float:
    """
    Converts RDP values to (epsilon, delta)-DP guarantee.

    Args:
        rdp_values: RDP values for each order
        orders: Corresponding RDP orders (alpha values)
        delta: Target delta for the DP guarantee

    Returns:
        The minimum epsilon that satisfies (epsilon, delta)-DP
    """
    ...

def compute_epsilon_batch(q: float, noise_multiplier: float, steps_list: List[int], orders: List[float], delta: float) -> List[float]:
    """
    Computes a list of epsilons for a list of steps (Vectorized Batching).

    Args:
        q: Sampling probability (batch_size / dataset_size)
        noise_multiplier: The noise multiplier (sigma)
        steps_list: List of training step counts
        orders: RDP orders (alpha values) to compute
        delta: Target delta for the DP guarantee

    Returns:
        List of epsilon values, one for each step count
    """
    ...