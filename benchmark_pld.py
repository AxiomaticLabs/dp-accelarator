"""Benchmark: Rust PLD backend vs Google dp_accounting PLD."""

import time
import sys

# ── Rust PLD (current) ──
from dp_accelerator._core import RustPldPmf

t0 = time.time()
pmf_rust = RustPldPmf.from_gaussian(1.0, 1.0, 1e-4, 10.0, True)
t1 = time.time()
composed_rust = pmf_rust.self_compose(100)
t2 = time.time()
eps_rust = composed_rust.get_epsilon_for_delta(1e-5)
t3 = time.time()

print("=== RUST PLD Backend ===")
print(f"  Gaussian PMF construction:  {(t1-t0)*1000:.1f} ms")
print(f"  self_compose(100):          {(t2-t1)*1000:.1f} ms")
print(f"  get_epsilon_for_delta:      {(t3-t2)*1000:.1f} ms")
print(f"  TOTAL:                      {(t3-t0)*1000:.1f} ms")
print(f"  epsilon = {eps_rust:.4f}")
print()

# ── Google dp_accounting PLD ──
sys.path.insert(0, "differential-privacy/python/dp_accounting")
try:
    from dp_accounting.pld import pld_privacy_accountant as google_pld
    from dp_accounting import dp_event as google_event

    t4 = time.time()
    acc_g = google_pld.PLDAccountant()
    acc_g.compose(google_event.GaussianDpEvent(1.0), count=100)
    t5 = time.time()
    eps_g = acc_g.get_epsilon(1e-5)
    t6 = time.time()

    print("=== Google dp_accounting PLD ===")
    print(f"  compose + get_epsilon:      {(t6-t4)*1000:.1f} ms")
    print(f"  epsilon = {eps_g:.4f}")
    print()
    print(f"=== Speedup: {(t6-t4)/(t3-t0):.1f}x ===")
    print(f"=== Accuracy diff: {abs(eps_rust - eps_g):.6f} ===")
except ImportError:
    print("(Google dp_accounting not available for comparison)")
