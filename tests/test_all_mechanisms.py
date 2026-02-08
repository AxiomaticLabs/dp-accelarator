"""
Comprehensive accuracy tests: dp-accelerator (Rust) vs dp_accounting (Python).

Verifies that every mechanism implemented in Rust produces values matching
Google's dp_accounting reference to within tight tolerances.
"""

import math
import time
import statistics
import pytest
import numpy as np

from dp_accelerator import (
    RdpAccountant,
    DPSGDAccountant,
    NeighboringRelation,
    compute_rdp_poisson_subsampled_gaussian,
    compute_rdp_sample_wor_gaussian,
    compute_rdp_tree_aggregation,
    compute_rdp_laplace,
    compute_rdp_randomized_response,
    compute_rdp_zcdp,
    compute_rdp_repeat_and_select,
    rdp_to_epsilon,
    rdp_to_delta,
    compute_epsilon_batch,
    DEFAULT_RDP_ORDERS,
)

# ── Try importing Google dp_accounting for comparison ─────────────
try:
    import dp_accounting
    from dp_accounting.rdp import rdp_privacy_accountant as rdp_mod

    HAS_DP_ACCOUNTING = True
except ImportError:
    HAS_DP_ACCOUNTING = False

ORDERS = list(DEFAULT_RDP_ORDERS)

# ═══════════════════════════════════════════════════════════════════
#  UNIT TESTS (always run, no dp_accounting needed)
# ═══════════════════════════════════════════════════════════════════


class TestRdpAccountantBasic:
    """Basic functionality of the Rust-backed RdpAccountant."""

    def test_poisson_gaussian_finite_positive(self):
        rdp = compute_rdp_poisson_subsampled_gaussian(0.01, 1.0, ORDERS)
        assert len(rdp) == len(ORDERS)
        for r in rdp:
            assert r >= 0.0
            assert math.isfinite(r)

    def test_poisson_gaussian_q0(self):
        rdp = compute_rdp_poisson_subsampled_gaussian(0.0, 1.0, ORDERS)
        assert all(r == 0.0 for r in rdp)

    def test_poisson_gaussian_q1(self):
        rdp = compute_rdp_poisson_subsampled_gaussian(1.0, 2.0, [5.0])
        assert abs(rdp[0] - 5.0 / (2 * 4.0)) < 1e-10  # alpha/(2*sigma^2)

    def test_sample_wor_gaussian_finite(self):
        rdp = compute_rdp_sample_wor_gaussian(0.01, 1.0, [2.0, 5.0, 10.0])
        assert len(rdp) == 3
        for r in rdp:
            assert r >= 0.0
            assert math.isfinite(r)

    def test_sample_wor_q0(self):
        rdp = compute_rdp_sample_wor_gaussian(0.0, 1.0, [2.0])
        assert rdp[0] == 0.0

    def test_tree_aggregation_proportional_to_order(self):
        rdp = compute_rdp_tree_aggregation(1.0, [100], [2.0, 10.0])
        assert rdp[1] / rdp[0] == pytest.approx(10.0 / 2.0, rel=1e-10)

    def test_laplace_alpha_one(self):
        eps = 1.0
        rdp = compute_rdp_laplace(eps, [1.0])
        expected = eps + math.exp(-eps) - 1.0
        assert rdp[0] == pytest.approx(expected, abs=1e-10)

    def test_laplace_monotone_in_order(self):
        rdp = compute_rdp_laplace(1.0, [2.0, 5.0, 10.0, 50.0])
        for i in range(len(rdp) - 1):
            assert rdp[i] <= rdp[i + 1] + 1e-12

    def test_randomized_response_p1(self):
        rdp = compute_rdp_randomized_response(1.0, 4, [2.0, 5.0], False)
        assert all(r == 0.0 for r in rdp)

    def test_randomized_response_k1(self):
        rdp = compute_rdp_randomized_response(0.5, 1, [2.0, 5.0], False)
        assert all(r == 0.0 for r in rdp)

    def test_zcdp(self):
        rdp = compute_rdp_zcdp(0.1, 0.5, [2.0, 5.0])
        assert rdp[0] == pytest.approx(0.1 + 0.5 * 2.0, abs=1e-12)
        assert rdp[1] == pytest.approx(0.1 + 0.5 * 5.0, abs=1e-12)

    def test_rdp_to_epsilon_roundtrip(self):
        rdp = compute_rdp_poisson_subsampled_gaussian(0.01, 1.0, ORDERS)
        total_rdp = [r * 1000 for r in rdp]
        eps, opt = rdp_to_epsilon(ORDERS, total_rdp, 1e-5)
        assert eps > 0
        assert math.isfinite(eps)
        # Verify the optimal order is one of our orders
        assert opt in ORDERS

    def test_rdp_to_delta_roundtrip(self):
        rdp = compute_rdp_poisson_subsampled_gaussian(0.01, 1.0, ORDERS)
        total_rdp = [r * 1000 for r in rdp]
        eps, _ = rdp_to_epsilon(ORDERS, total_rdp, 1e-5)
        delta, _ = rdp_to_delta(ORDERS, total_rdp, eps)
        assert delta <= 1e-5 + 1e-10  # should be close to original delta

    def test_rdp_accountant_compose_chain(self):
        acc = RdpAccountant()
        acc.compose_poisson_subsampled_gaussian(0.01, 1.0, count=500)
        acc.compose_poisson_subsampled_gaussian(0.01, 1.0, count=500)
        eps_two = acc.get_epsilon(1e-5)

        acc2 = RdpAccountant()
        acc2.compose_poisson_subsampled_gaussian(0.01, 1.0, count=1000)
        eps_one = acc2.get_epsilon(1e-5)

        assert eps_two == pytest.approx(eps_one, rel=1e-10)

    def test_dpsgd_accountant_matches_rdp(self):
        dpsgd = DPSGDAccountant(noise_multiplier=1.0, batch_size=600, dataset_size=60000)
        eps_dpsgd = dpsgd.get_epsilon(steps=10000, delta=1e-5)

        acc = RdpAccountant(orders=dpsgd.orders)
        q = 600 / 60000
        acc.compose_poisson_subsampled_gaussian(q, 1.0, count=10000)
        eps_rdp = acc.get_epsilon(1e-5)

        assert eps_dpsgd == pytest.approx(eps_rdp, rel=1e-8)

    def test_repeat_and_select_poisson(self):
        rdp_single = compute_rdp_poisson_subsampled_gaussian(0.01, 1.0, ORDERS)
        result = compute_rdp_repeat_and_select(ORDERS, rdp_single, 10.0, float("inf"))
        assert len(result) == len(ORDERS)
        # repeat-and-select should give higher privacy cost than single run
        for r_rs, r_s in zip(result, rdp_single):
            if math.isfinite(r_rs) and r_s > 0:
                assert r_rs >= r_s - 1e-10

    def test_repeat_and_select_geometric(self):
        rdp_single = compute_rdp_poisson_subsampled_gaussian(0.01, 1.0, ORDERS)
        result = compute_rdp_repeat_and_select(ORDERS, rdp_single, 5.0, 1.0)
        assert len(result) == len(ORDERS)


# ═══════════════════════════════════════════════════════════════════
#  CROSS-VALIDATION against dp_accounting (skipped if not installed)
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_DP_ACCOUNTING, reason="dp_accounting not installed")
class TestCrossValidation:
    """Verify Rust output matches Google dp_accounting within tight tolerance."""

    def _python_rdp_poisson_gaussian(self, q, sigma, orders):
        return rdp_mod._compute_rdp_poisson_subsampled_gaussian(q, sigma, orders)

    def _python_rdp_sample_wor(self, q, sigma, orders):
        return rdp_mod._compute_rdp_sample_wor_gaussian(q, sigma, orders)

    def _python_epsilon(self, orders, rdp, delta):
        return rdp_mod.compute_epsilon(orders, rdp, delta)

    def _python_delta(self, orders, rdp, epsilon):
        return rdp_mod.compute_delta(orders, rdp, epsilon)

    # ── Poisson-subsampled Gaussian ──

    @pytest.mark.parametrize(
        "q,sigma",
        [
            (0.01, 1.0),
            (0.001, 0.5),
            (0.1, 4.0),
            (0.5, 10.0),
            (1.0, 1.0),
            (0.0, 1.0),
        ],
    )
    def test_poisson_gaussian_rdp(self, q, sigma):
        rust = compute_rdp_poisson_subsampled_gaussian(q, sigma, ORDERS)
        python = self._python_rdp_poisson_gaussian(q, sigma, ORDERS)
        for i, (r, p) in enumerate(zip(rust, python)):
            if math.isfinite(p) and p > 0:
                assert r == pytest.approx(p, rel=1e-6), (
                    f"order={ORDERS[i]}: rust={r} python={p}"
                )
            elif p == 0:
                assert r == pytest.approx(0.0, abs=1e-15)

    # ── Epsilon conversion ──

    @pytest.mark.parametrize(
        "q,sigma,steps,delta",
        [
            (0.01, 1.0, 1000, 1e-5),
            (0.01, 1.0, 10000, 1e-5),
            (0.001, 0.5, 90000, 1e-6),
            (0.1, 4.0, 20000, 1e-5),
            (0.032, 1.0, 2000, 1e-3),
        ],
    )
    def test_epsilon_conversion(self, q, sigma, steps, delta):
        rdp_rust = compute_rdp_poisson_subsampled_gaussian(q, sigma, ORDERS)
        total_rdp = [r * steps for r in rdp_rust]
        eps_rust, _ = rdp_to_epsilon(ORDERS, total_rdp, delta)

        rdp_python = self._python_rdp_poisson_gaussian(q, sigma, ORDERS)
        total_rdp_py = [float(r * steps) for r in rdp_python]
        eps_python, _ = self._python_epsilon(ORDERS, total_rdp_py, delta)

        assert eps_rust == pytest.approx(eps_python, rel=1e-4), (
            f"rust={eps_rust} python={eps_python}"
        )

    # ── Delta conversion ──

    def test_delta_conversion(self):
        q, sigma, steps = 0.01, 1.0, 1000
        rdp_rust = compute_rdp_poisson_subsampled_gaussian(q, sigma, ORDERS)
        total_rdp = [r * steps for r in rdp_rust]
        eps_rust, _ = rdp_to_epsilon(ORDERS, total_rdp, 1e-5)

        delta_rust, _ = rdp_to_delta(ORDERS, total_rdp, eps_rust)
        delta_python, _ = self._python_delta(ORDERS, total_rdp, eps_rust)

        assert delta_rust == pytest.approx(delta_python, rel=1e-4), (
            f"rust={delta_rust} python={delta_python}"
        )

    # ── Sampling without replacement ──

    @pytest.mark.parametrize(
        "q,sigma",
        [
            (0.01, 1.0),
            (0.05, 2.0),
            (0.1, 0.5),
        ],
    )
    def test_sample_wor_rdp(self, q, sigma):
        int_orders = [2.0, 3.0, 5.0, 10.0, 20.0, 50.0]
        rust = compute_rdp_sample_wor_gaussian(q, sigma, int_orders)
        python = self._python_rdp_sample_wor(q, sigma, int_orders)
        for i, (r, p) in enumerate(zip(rust, python)):
            if math.isfinite(p) and p > 0:
                assert r == pytest.approx(float(p), rel=1e-4), (
                    f"order={int_orders[i]}: rust={r} python={p}"
                )

    # ── Tree aggregation ──

    def test_tree_aggregation(self):
        sigma = 1.0
        step_counts = [100, 200]
        orders_subset = [2.0, 5.0, 10.0]
        rust = compute_rdp_tree_aggregation(sigma, step_counts, orders_subset)
        python = rdp_mod._compute_rdp_single_epoch_tree_aggregation(
            sigma, step_counts, orders_subset
        )
        for r, p in zip(rust, python):
            assert r == pytest.approx(float(p), rel=1e-10)

    # ── Laplace ──

    @pytest.mark.parametrize("pure_eps", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_laplace(self, pure_eps):
        orders_subset = [1.0, 1.5, 2.0, 5.0, 10.0, 50.0]
        rust = compute_rdp_laplace(pure_eps, orders_subset)
        python = [rdp_mod._laplace_rdp(pure_eps, a) for a in orders_subset]
        for i, (r, p) in enumerate(zip(rust, python)):
            assert r == pytest.approx(p, rel=1e-6), (
                f"order={orders_subset[i]}: rust={r} python={p}"
            )

    # ── Randomized Response ──

    @pytest.mark.parametrize(
        "p,k",
        [(0.5, 4), (0.1, 10), (0.9, 2)],
    )
    def test_randomized_response_replace_special(self, p, k):
        orders_subset = [2.0, 5.0, 10.0, 50.0]
        rust = compute_rdp_randomized_response(p, k, orders_subset, False)
        python = [
            rdp_mod._randomized_response_rdp_replace_special(p, k, a)
            for a in orders_subset
        ]
        for i, (r, pv) in enumerate(zip(rust, python)):
            assert r == pytest.approx(float(pv), rel=1e-6), (
                f"order={orders_subset[i]}: rust={r} python={pv}"
            )

    @pytest.mark.parametrize(
        "p,k",
        [(0.5, 4), (0.1, 10), (0.9, 2)],
    )
    def test_randomized_response_replace_one(self, p, k):
        orders_subset = [2.0, 5.0, 10.0, 50.0]
        rust = compute_rdp_randomized_response(p, k, orders_subset, True)
        python = [
            rdp_mod._randomized_response_rdp_replace_one(p, k, a)
            for a in orders_subset
        ]
        for i, (r, pv) in enumerate(zip(rust, python)):
            assert r == pytest.approx(float(pv), rel=1e-6), (
                f"order={orders_subset[i]}: rust={r} python={pv}"
            )

    # ── Full accountant end-to-end ──

    @pytest.mark.parametrize(
        "nm,bs,ds,steps,delta",
        [
            (1.0, 600, 60000, 1000, 1e-5),
            (1.0, 600, 60000, 10000, 1e-5),
            (0.5, 256, 1200000, 90000, 1e-6),
            (4.0, 512, 100000, 20000, 1e-5),
            (1.5, 1024, 10000000, 100000, 1e-7),
        ],
    )
    def test_full_dpsgd_epsilon(self, nm, bs, ds, steps, delta):
        """End-to-end: DPSGDAccountant vs dp_accounting.rdp.RdpAccountant."""
        # Rust
        q = bs / ds
        rust_acc = RdpAccountant(orders=ORDERS)
        rust_acc.compose_poisson_subsampled_gaussian(q, nm, count=steps)
        eps_rust = rust_acc.get_epsilon(delta)

        # Python
        py_acc = dp_accounting.rdp.RdpAccountant(
            orders=ORDERS,
            neighboring_relation=dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE,
        )
        event = dp_accounting.PoissonSampledDpEvent(
            q, dp_accounting.GaussianDpEvent(nm)
        )
        py_acc.compose(event, steps)
        eps_python = py_acc.get_epsilon(target_delta=delta)

        assert eps_rust == pytest.approx(eps_python, rel=1e-4), (
            f"config=({nm},{bs},{ds},{steps},{delta}): rust={eps_rust} python={eps_python}"
        )

    # ── Repeat-and-select ──

    def test_repeat_and_select_vs_python(self):
        orders_subset = [2.0, 5.0, 10.0, 20.0, 50.0, 128.0]
        rdp_single = list(
            rdp_mod._compute_rdp_poisson_subsampled_gaussian(0.01, 1.0, orders_subset)
        )
        rust = compute_rdp_repeat_and_select(orders_subset, rdp_single, 10.0, float("inf"))
        python = rdp_mod._compute_rdp_repeat_and_select(orders_subset, rdp_single, 10.0, float("inf"))
        for i, (r, p) in enumerate(zip(rust, python)):
            if math.isfinite(p):
                assert r == pytest.approx(float(p), rel=1e-4), (
                    f"order={orders_subset[i]}: rust={r} python={p}"
                )


# ═══════════════════════════════════════════════════════════════════
#  SPEED BENCHMARKS
# ═══════════════════════════════════════════════════════════════════


def _bench(fn, *args, warmup=3, repeats=10, **kwargs):
    for _ in range(warmup):
        result = fn(*args, **kwargs)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return result, statistics.median(times)


@pytest.mark.skipif(not HAS_DP_ACCOUNTING, reason="dp_accounting not installed")
class TestSpeedBenchmarks:
    """Verify Rust is faster than Python for every mechanism."""

    def test_poisson_gaussian_speed(self):
        q, sigma = 0.01, 1.0
        _, t_rust = _bench(compute_rdp_poisson_subsampled_gaussian, q, sigma, ORDERS)
        _, t_python = _bench(
            rdp_mod._compute_rdp_poisson_subsampled_gaussian, q, sigma, ORDERS
        )
        speedup = t_python / t_rust
        print(f"\nPoisson-Gaussian: Rust={t_rust*1e3:.3f}ms Python={t_python*1e3:.3f}ms Speedup={speedup:.0f}x")
        assert speedup > 1, f"Rust should be faster, got {speedup:.1f}x"

    def test_epsilon_conversion_speed(self):
        q, sigma, steps = 0.01, 1.0, 10000
        rdp = [r * steps for r in compute_rdp_poisson_subsampled_gaussian(q, sigma, ORDERS)]
        _, t_rust = _bench(rdp_to_epsilon, ORDERS, rdp, 1e-5)
        _, t_python = _bench(rdp_mod.compute_epsilon, ORDERS, rdp, 1e-5)
        speedup = t_python / t_rust
        print(f"\nEpsilon conversion: Rust={t_rust*1e6:.1f}us Python={t_python*1e6:.1f}us Speedup={speedup:.0f}x")
        assert speedup > 1

    def test_sample_wor_speed(self):
        q, sigma = 0.01, 1.0
        int_orders = [float(i) for i in range(2, 65)]
        _, t_rust = _bench(compute_rdp_sample_wor_gaussian, q, sigma, int_orders)
        _, t_python = _bench(rdp_mod._compute_rdp_sample_wor_gaussian, q, sigma, int_orders)
        speedup = t_python / t_rust
        print(f"\nSample-WOR Gaussian: Rust={t_rust*1e3:.3f}ms Python={t_python*1e3:.3f}ms Speedup={speedup:.0f}x")
        assert speedup > 1

    def test_laplace_speed(self):
        _, t_rust = _bench(compute_rdp_laplace, 1.0, ORDERS)
        _, t_python = _bench(
            lambda: [rdp_mod._laplace_rdp(1.0, a) for a in ORDERS]
        )
        speedup = t_python / t_rust
        print(f"\nLaplace RDP: Rust={t_rust*1e6:.1f}us Python={t_python*1e6:.1f}us Speedup={speedup:.0f}x")
        assert speedup > 1

    def test_full_dpsgd_speed(self):
        """End-to-end DP-SGD accounting: compose + get_epsilon."""
        q, sigma, steps, delta = 0.01, 1.0, 10000, 1e-5

        def rust_fn():
            acc = RdpAccountant(orders=ORDERS)
            acc.compose_poisson_subsampled_gaussian(q, sigma, count=steps)
            return acc.get_epsilon(delta)

        def python_fn():
            acc = dp_accounting.rdp.RdpAccountant(
                orders=ORDERS,
                neighboring_relation=dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE,
            )
            event = dp_accounting.PoissonSampledDpEvent(
                q, dp_accounting.GaussianDpEvent(sigma)
            )
            acc.compose(event, steps)
            return acc.get_epsilon(target_delta=delta)

        _, t_rust = _bench(rust_fn)
        _, t_python = _bench(python_fn)
        speedup = t_python / t_rust
        print(f"\nFull DP-SGD E2E: Rust={t_rust*1e3:.3f}ms Python={t_python*1e3:.3f}ms Speedup={speedup:.0f}x")
        assert speedup > 10, f"Expected >10x speedup, got {speedup:.1f}x"
