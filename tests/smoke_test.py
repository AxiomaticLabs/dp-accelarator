"""Quick smoke test for all new modules."""

import sys

try:
    from dp_accelerator import (
        RdpAccountant,
        GaussianDpEvent,
        PoissonSampledDpEvent,
        get_epsilon_gaussian,
        get_sigma_gaussian,
    )
    from dp_accelerator.pld import PLDAccountant

    print("All imports successful!")

    # Test gaussian mechanism
    eps = get_epsilon_gaussian(1.0, 1e-5)
    sigma = get_sigma_gaussian(1.0, 1e-5)
    print(f"get_epsilon_gaussian(1.0, 1e-5) = {eps:.6f}")
    print(f"get_sigma_gaussian(1.0, 1e-5) = {sigma:.6f}")

    # Test DpEvent-based RdpAccountant
    acc = RdpAccountant()
    acc.compose(PoissonSampledDpEvent(0.01, GaussianDpEvent(1.0)), count=1000)
    e = acc.get_epsilon(target_delta=1e-5)
    print(f"RdpAccountant DpEvent API: eps={e:.4f}")

    # Test direct API still works
    acc2 = RdpAccountant()
    acc2.compose_poisson_subsampled_gaussian(0.01, 1.0, 1000)
    e2 = acc2.get_epsilon(target_delta=1e-5)
    print(f"RdpAccountant direct API: eps={e2:.4f}")

    assert abs(e - e2) < 1e-10, f"DpEvent and direct API disagree: {e} vs {e2}"

    # Test PLDAccountant
    pld_acc = PLDAccountant()
    pld_acc.compose(GaussianDpEvent(1.0), count=100)
    e_pld = pld_acc.get_epsilon(1e-5)
    print(f"PLDAccountant: eps={e_pld:.4f}")

    print("ALL SMOKE TESTS PASSED!")
except Exception as ex:
    print(f"FAILED: {ex}", file=sys.stderr)
    import traceback

    traceback.print_exc()
    sys.exit(1)
