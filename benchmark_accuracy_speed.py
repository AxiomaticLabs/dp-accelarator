#!/usr/bin/env python3
"""
Comprehensive Benchmark: dp-accelerator (Rust) vs dp_accounting (Python)
========================================================================

Tests both ACCURACY (do they agree on epsilon values?) and SPEED
(how much faster is the Rust implementation?) across a wide range
of realistic DP-SGD configurations.
"""

import time
import statistics
import numpy as np
import dp_accounting
from dp_accelerator import DPSGDAccountant, compute_epsilon_batch

# ─────────────────────────────────────────────────────────────────
# Shared RDP orders (same in both implementations)
# ─────────────────────────────────────────────────────────────────
ORDERS = np.concatenate((
    np.linspace(1.01, 8, num=50),
    np.arange(8, 64),
    np.linspace(65, 512, num=10, dtype=int),
)).tolist()


# ═════════════════════════════════════════════════════════════════
#  Helper: Python (dp_accounting) baseline
# ═════════════════════════════════════════════════════════════════
def python_compute_epsilon(q, noise_multiplier, steps, delta, orders=ORDERS):
    """Compute epsilon using Google's dp_accounting (pure Python)."""
    accountant = dp_accounting.rdp.RdpAccountant(
        orders=orders,
        neighboring_relation=dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE,
    )
    event = dp_accounting.PoissonSampledDpEvent(
        q, dp_accounting.GaussianDpEvent(noise_multiplier)
    )
    accountant.compose(event, steps)
    return accountant.get_epsilon(target_delta=delta)


def python_compute_epsilon_batch(q, noise_multiplier, steps_list, delta, orders=ORDERS):
    """Compute epsilon for multiple step counts using Python (one at a time)."""
    return [python_compute_epsilon(q, noise_multiplier, s, delta, orders) for s in steps_list]


# ═════════════════════════════════════════════════════════════════
#  Helper: Rust (dp-accelerator) implementation
# ═════════════════════════════════════════════════════════════════
def rust_compute_epsilon(q, noise_multiplier, steps, delta, orders=ORDERS):
    """Compute epsilon using dp-accelerator (Rust)."""
    result = compute_epsilon_batch(q, noise_multiplier, [steps], orders, delta)
    return result[0]


def rust_compute_epsilon_batch(q, noise_multiplier, steps_list, delta, orders=ORDERS):
    """Compute epsilon for multiple step counts using Rust vectorized."""
    return compute_epsilon_batch(q, noise_multiplier, steps_list, orders, delta)


# ═════════════════════════════════════════════════════════════════
#  Timing utility
# ═════════════════════════════════════════════════════════════════
def benchmark_fn(fn, *args, warmup=3, repeats=10, **kwargs):
    """Time a function with warmup runs. Returns (result, median_time_sec)."""
    # Warmup
    for _ in range(warmup):
        result = fn(*args, **kwargs)
    # Timed runs
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return result, statistics.median(times)


# ═════════════════════════════════════════════════════════════════
#  TEST CONFIGURATIONS
# ═════════════════════════════════════════════════════════════════
CONFIGS = [
    # (name, noise_multiplier, batch_size, dataset_size, steps, delta)
    ("MNIST-small",       1.0,  600,   60_000,    1_000,  1e-5),
    ("MNIST-typical",     1.0,  600,   60_000,   10_000,  1e-5),
    ("MNIST-long",        1.0,  600,   60_000,   50_000,  1e-5),
    ("ImageNet-like",     0.5,  256,  1_200_000,  90_000,  1e-6),
    ("Low-noise",         0.3,  128,   50_000,    5_000,  1e-5),
    ("High-noise",        4.0,  512,  100_000,   20_000,  1e-5),
    ("Tiny-dataset",      1.0,   32,    1_000,    2_000,  1e-3),
    ("Large-dataset",     1.5, 1024, 10_000_000, 100_000,  1e-7),
    ("Very-high-sigma",  10.0,  256,   60_000,   10_000,  1e-5),
    ("Edge-high-q",       1.0, 5000,   10_000,    1_000,  1e-5),
]


# ═════════════════════════════════════════════════════════════════
#  PART 1: ACCURACY COMPARISON
# ═════════════════════════════════════════════════════════════════
def run_accuracy_tests():
    """Compare epsilon values between Python and Rust implementations."""
    print("=" * 85)
    print("PART 1: ACCURACY COMPARISON — Python (dp_accounting) vs Rust (dp-accelerator)")
    print("=" * 85)
    print()
    print(f"{'Config':<20} {'Python ε':>14} {'Rust ε':>14} {'Abs Diff':>12} {'Rel Diff':>12} {'Match?':>8}")
    print("─" * 85)

    all_pass = True
    max_rel_diff = 0.0

    for name, sigma, bs, ds, steps, delta in CONFIGS:
        q = bs / ds
        py_eps = python_compute_epsilon(q, sigma, steps, delta)
        rs_eps = rust_compute_epsilon(q, sigma, steps, delta)

        abs_diff = abs(py_eps - rs_eps)
        rel_diff = abs_diff / max(abs(py_eps), 1e-15)
        max_rel_diff = max(max_rel_diff, rel_diff)

        # Allow up to 1e-6 relative tolerance (floating-point differences
        # between Rust's statrs and Python's mpmath are expected)
        match = rel_diff < 1e-6
        if not match:
            all_pass = False

        print(
            f"{name:<20} {py_eps:>14.10f} {rs_eps:>14.10f} "
            f"{abs_diff:>12.2e} {rel_diff:>12.2e} {'  ✓' if match else '  ✗':>8}"
        )

    print("─" * 85)
    print(f"Max relative difference: {max_rel_diff:.2e}")
    if all_pass:
        print("✓ ALL ACCURACY TESTS PASSED (relative diff < 1e-6)")
    else:
        print("✗ SOME ACCURACY TESTS FAILED — investigate differences")
    print()

    return all_pass, max_rel_diff


# ═════════════════════════════════════════════════════════════════
#  PART 2: SINGLE-CALL SPEED COMPARISON
# ═════════════════════════════════════════════════════════════════
def run_speed_single():
    """Benchmark single epsilon computation."""
    print("=" * 85)
    print("PART 2: SPEED — Single epsilon computation")
    print("=" * 85)
    print()
    print(f"{'Config':<20} {'Python (ms)':>12} {'Rust (ms)':>12} {'Speedup':>10}")
    print("─" * 60)

    speedups = []

    for name, sigma, bs, ds, steps, delta in CONFIGS:
        q = bs / ds

        _, py_time = benchmark_fn(python_compute_epsilon, q, sigma, steps, delta)
        _, rs_time = benchmark_fn(rust_compute_epsilon, q, sigma, steps, delta)

        speedup = py_time / rs_time if rs_time > 0 else float('inf')
        speedups.append(speedup)

        print(
            f"{name:<20} {py_time*1000:>12.3f} {rs_time*1000:>12.4f} {speedup:>9.0f}x"
        )

    print("─" * 60)
    print(f"Median speedup: {statistics.median(speedups):.0f}x")
    print(f"Min speedup:    {min(speedups):.0f}x")
    print(f"Max speedup:    {max(speedups):.0f}x")
    print()

    return speedups


# ═════════════════════════════════════════════════════════════════
#  PART 3: BATCH SPEED COMPARISON
# ═════════════════════════════════════════════════════════════════
def run_speed_batch():
    """Benchmark batch epsilon computation (multiple step counts at once)."""
    print("=" * 85)
    print("PART 3: SPEED — Batch epsilon computation (100 step counts)")
    print("=" * 85)
    print()

    batch_sizes_to_test = [10, 50, 100, 200]
    sigma = 1.0
    q = 600 / 60000
    delta = 1e-5

    print(f"{'Batch Size':<12} {'Python (ms)':>12} {'Rust (ms)':>12} {'Speedup':>10}")
    print("─" * 50)

    for n in batch_sizes_to_test:
        steps_list = list(range(100, 100 + n * 100, 100))  # [100, 200, ..., n*100]

        _, py_time = benchmark_fn(python_compute_epsilon_batch, q, sigma, steps_list, delta, warmup=2, repeats=5)
        _, rs_time = benchmark_fn(rust_compute_epsilon_batch, q, sigma, steps_list, delta, warmup=2, repeats=5)

        speedup = py_time / rs_time if rs_time > 0 else float('inf')

        print(
            f"{n:<12} {py_time*1000:>12.2f} {rs_time*1000:>12.4f} {speedup:>9.0f}x"
        )

    print()


# ═════════════════════════════════════════════════════════════════
#  PART 4: SCALING — How speed changes with number of RDP orders
# ═════════════════════════════════════════════════════════════════
def run_scaling_orders():
    """Benchmark how performance scales with number of RDP orders."""
    print("=" * 85)
    print("PART 4: SCALING — Performance vs number of RDP orders")
    print("=" * 85)
    print()

    q = 600 / 60000
    sigma = 1.0
    steps = 10_000
    delta = 1e-5

    order_counts = [10, 50, 100, 200, 500]

    print(f"{'# Orders':<12} {'Python (ms)':>12} {'Rust (ms)':>12} {'Speedup':>10}")
    print("─" * 50)

    for n_orders in order_counts:
        orders = np.linspace(1.01, 512, num=n_orders).tolist()

        _, py_time = benchmark_fn(python_compute_epsilon, q, sigma, steps, delta, orders=orders, warmup=2, repeats=5)
        _, rs_time = benchmark_fn(rust_compute_epsilon, q, sigma, steps, delta, orders=orders, warmup=2, repeats=5)

        speedup = py_time / rs_time if rs_time > 0 else float('inf')

        print(
            f"{n_orders:<12} {py_time*1000:>12.3f} {rs_time*1000:>12.4f} {speedup:>9.0f}x"
        )

    print()


# ═════════════════════════════════════════════════════════════════
#  PART 5: ACCURACY SWEEP — epsilon over training steps
# ═════════════════════════════════════════════════════════════════
def run_accuracy_sweep():
    """Compare epsilon curves over many training steps."""
    print("=" * 85)
    print("PART 5: ACCURACY SWEEP — Epsilon at every 1000 steps up to 50k")
    print("=" * 85)
    print()

    q = 600 / 60000
    sigma = 1.0
    delta = 1e-5
    step_range = list(range(1000, 50_001, 1000))

    # Rust batch
    rs_epsilons = rust_compute_epsilon_batch(q, sigma, step_range, delta)

    # Python one-by-one
    py_epsilons = [python_compute_epsilon(q, sigma, s, delta) for s in step_range]

    diffs = [abs(p - r) / max(abs(p), 1e-15) for p, r in zip(py_epsilons, rs_epsilons)]

    print(f"{'Steps':>8} {'Python ε':>14} {'Rust ε':>14} {'Rel Diff':>12}")
    print("─" * 52)
    # Print a subset
    for i in range(0, len(step_range), 5):
        print(
            f"{step_range[i]:>8} {py_epsilons[i]:>14.10f} {rs_epsilons[i]:>14.10f} {diffs[i]:>12.2e}"
        )

    print("─" * 52)
    print(f"Max relative diff across {len(step_range)} points: {max(diffs):.2e}")
    print(f"Mean relative diff: {statistics.mean(diffs):.2e}")
    print()


# ═════════════════════════════════════════════════════════════════
#  PART 6: STRESS TEST — Very large batch computation
# ═════════════════════════════════════════════════════════════════
def run_stress_test():
    """Stress test: compute 10,000 epsilon values in one call."""
    print("=" * 85)
    print("PART 6: STRESS TEST — 10,000 epsilon values in single batch call")
    print("=" * 85)
    print()

    q = 600 / 60000
    sigma = 1.0
    delta = 1e-5
    steps_list = list(range(1, 10_001))

    # Rust batch — all 10k at once
    t0 = time.perf_counter()
    rs_results = rust_compute_epsilon_batch(q, sigma, steps_list, delta)
    t1 = time.perf_counter()
    rust_time = t1 - t0

    # Python — sample 20 points to estimate total time
    sample_indices = list(range(0, 10_000, 500))  # 20 points
    t0 = time.perf_counter()
    py_subset = [python_compute_epsilon(q, sigma, steps_list[i], delta) for i in sample_indices]
    t1 = time.perf_counter()
    py_time_subset = t1 - t0

    # Extrapolated Python time for all 10,000
    py_estimated_total = py_time_subset * (10_000 / len(sample_indices))

    # Accuracy on the sampled subset
    diffs = [abs(py_subset[j] - rs_results[sample_indices[j]]) / max(abs(py_subset[j]), 1e-15)
             for j in range(len(sample_indices))]

    print(f"Rust:   {len(steps_list):,} epsilon values in {rust_time*1000:.2f} ms")
    print(f"Python: {len(sample_indices)} epsilon values in {py_time_subset*1000:.2f} ms")
    print(f"Python estimated for {len(steps_list):,}: {py_estimated_total:.2f} s")
    print(f"Speedup (estimated): {py_estimated_total / rust_time:.0f}x")
    print(f"Max relative diff on sample: {max(diffs):.2e}")
    print()


# ═════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════
def main():
    print()
    print("╔═══════════════════════════════════════════════════════════════════════════════════╗")
    print("║  dp-accelerator (Rust) vs dp_accounting (Python) — Full Benchmark Suite         ║")
    print("╚═══════════════════════════════════════════════════════════════════════════════════╝")
    print()

    accuracy_pass, max_diff = run_accuracy_tests()
    single_speedups = run_speed_single()
    run_speed_batch()
    run_scaling_orders()
    run_accuracy_sweep()
    run_stress_test()

    # ── Summary ──────────────────────────────────────────────────
    print("=" * 85)
    print("SUMMARY")
    print("=" * 85)
    print()
    print(f"  Accuracy:  {'PASS' if accuracy_pass else 'FAIL'} (max relative diff: {max_diff:.2e})")
    print(f"  Speed:     Median {statistics.median(single_speedups):.0f}x faster  "
          f"(range: {min(single_speedups):.0f}x – {max(single_speedups):.0f}x)")
    print()
    if accuracy_pass:
        print("  ✓ dp-accelerator is a correct AND fast drop-in replacement")
    else:
        print("  ⚠ dp-accelerator has accuracy differences — review above")
    print()


if __name__ == "__main__":
    main()
