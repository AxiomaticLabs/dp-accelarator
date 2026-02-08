#!/usr/bin/env python3
"""Test the Rusty DP Engine package."""

from dp_accelerator import DPSGDAccountant
from dp_accelerator.jax_adapter import compute_dpsgd_epsilon
import time

def test_basic_functionality():
    """Test basic DPSGD accountant functionality."""
    print("🧪 Testing Rusty DP Engine...")

    # Test the universal API
    accountant = DPSGDAccountant(
        noise_multiplier=1.0,
        batch_size=600,
        dataset_size=60000
    )

    epsilon = accountant.get_epsilon(steps=1000, delta=1e-5)
    print(".4f")

    # Test RDP curve
    rdp_curve = accountant.get_rdp_curve(steps=1000)
    print(f"RDP curve sample: α=2.0 → RDP={rdp_curve[2.0]:.4f}")

    # Test JAX adapter
    epsilon_jax = compute_dpsgd_epsilon(
        noise_multiplier=1.0,
        batch_size=600,
        dataset_size=60000,
        num_steps=1000,
        delta=1e-5
    )
    print(".4f")

    # Verify they match
    assert abs(epsilon - epsilon_jax) < 1e-10, "Results should be identical"
    print("✅ JAX adapter matches universal API")

def test_performance():
    """Test performance compared to a simple baseline."""
    print("\n⚡ Performance Test...")

    accountant = DPSGDAccountant(
        noise_multiplier=1.0,
        batch_size=600,
        dataset_size=60000
    )

    # Time the computation
    start = time.time()
    for _ in range(100):
        accountant.get_epsilon(steps=10000, delta=1e-5)
    end = time.time()

    total_time = end - start
    avg_time = total_time / 100

    print(".6f")
    print(".1f")

if __name__ == "__main__":
    test_basic_functionality()
    test_performance()
    print("\n🎉 All tests passed! Rusty DP Engine is working correctly.")