from typing import List, Tuple

def compute_epsilon_batch(
    q: float,
    noise_multiplier: float,
    steps_list: List[int],
    orders: List[float],
    delta: float,
) -> List[float]:
    """Compute epsilon values for multiple step counts using the Rust backend."""
    ...

def get_epsilon_gaussian(sigma: float, delta: float, tol: float) -> float:
    """Compute epsilon for the Gaussian mechanism (Balle & Wang)."""
    ...

def get_sigma_gaussian(epsilon: float, delta: float, tol: float) -> float:
    """Compute optimal noise std for the Gaussian mechanism."""
    ...

def compute_rdp_poisson_subsampled_gaussian(
    q: float, sigma: float, orders: List[float]
) -> List[float]:
    """RDP for Poisson-subsampled Gaussian mechanism."""
    ...

def compute_rdp_sample_wor_gaussian(
    q: float, sigma: float, orders: List[float]
) -> List[float]:
    """RDP for sampling-without-replacement Gaussian mechanism."""
    ...

def compute_rdp_laplace(pure_eps: float, orders: List[float]) -> List[float]:
    """RDP for the Laplace mechanism."""
    ...

def rdp_to_epsilon_vec(
    orders: List[float], rdp_values: List[float], delta: float
) -> Tuple[float, float]:
    """Convert RDP to epsilon. Returns (epsilon, optimal_order)."""
    ...

def rdp_to_delta_vec(
    orders: List[float], rdp_values: List[float], epsilon: float
) -> Tuple[float, float]:
    """Convert RDP to delta. Returns (delta, optimal_order)."""
    ...
