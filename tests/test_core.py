import pytest
import math
from dp_accelerator import DPSGDAccountant


@pytest.fixture
def accountant():
    """Setup a standard accountant for tests."""
    return DPSGDAccountant(noise_multiplier=1.0, batch_size=600, dataset_size=60000)


def test_basic_epsilon_calculation(accountant):
    """Test that we get a finite float epsilon."""
    epsilon = accountant.get_epsilon(steps=100)
    assert isinstance(epsilon, float)
    assert epsilon > 0.0
    assert epsilon < 100.0  # Sanity check


def test_epsilon_increases_with_steps(accountant):
    """More steps should leak more privacy (higher epsilon)."""
    eps_100 = accountant.get_epsilon(steps=100)
    eps_500 = accountant.get_epsilon(steps=500)
    eps_1000 = accountant.get_epsilon(steps=1000)
    assert eps_100 < eps_500 < eps_1000


def test_batch_processing_matches_single(accountant):
    """
    Verify that get_epsilon_batch returns the same results as
    calling get_epsilon individually. Proves the batch optimization is safe.
    """
    steps_list = [100, 500, 1000]
    delta = 1e-5

    # 1. Compute individually (slow way)
    single_results = [accountant.get_epsilon(s, delta) for s in steps_list]

    # 2. Compute batch (fast way)
    batch_results = accountant.get_epsilon_batch(steps_list, delta)

    # 3. Compare
    assert len(single_results) == len(batch_results)
    for s, b in zip(single_results, batch_results):
        assert math.isclose(s, b, rel_tol=1e-9), f"Mismatch: single={s}, batch={b}"


def test_higher_noise_lower_epsilon():
    """More noise should give better privacy (lower epsilon)."""
    acc_low_noise = DPSGDAccountant(
        noise_multiplier=0.5, batch_size=600, dataset_size=60000
    )
    acc_high_noise = DPSGDAccountant(
        noise_multiplier=2.0, batch_size=600, dataset_size=60000
    )

    eps_low = acc_low_noise.get_epsilon(steps=1000)
    eps_high = acc_high_noise.get_epsilon(steps=1000)
    assert eps_high < eps_low


def test_zero_steps_returns_zero(accountant):
    """Zero training steps should mean zero privacy cost."""
    eps = accountant.get_epsilon(steps=0)
    assert eps == 0.0


def test_empty_batch_list(accountant):
    """Batch with empty list should return empty list."""
    result = accountant.get_epsilon_batch([], delta=1e-5)
    assert result == []


def test_invalid_negative_noise():
    """Negative noise multiplier should raise ValueError."""
    with pytest.raises(ValueError, match="noise_multiplier"):
        DPSGDAccountant(noise_multiplier=-1.0, batch_size=10, dataset_size=100)


def test_invalid_zero_batch_size():
    """Zero batch size should raise ValueError."""
    with pytest.raises(ValueError, match="batch_size"):
        DPSGDAccountant(noise_multiplier=1.0, batch_size=0, dataset_size=100)


def test_invalid_zero_dataset_size():
    """Zero dataset size should raise ValueError."""
    with pytest.raises(ValueError, match="dataset_size"):
        DPSGDAccountant(noise_multiplier=1.0, batch_size=10, dataset_size=0)


def test_invalid_batch_larger_than_dataset():
    """Batch size larger than dataset should raise ValueError."""
    with pytest.raises(ValueError):
        DPSGDAccountant(noise_multiplier=1.0, batch_size=101, dataset_size=100)


def test_deterministic_results(accountant):
    """Same inputs should always produce same outputs."""
    eps1 = accountant.get_epsilon(steps=500, delta=1e-5)
    eps2 = accountant.get_epsilon(steps=500, delta=1e-5)
    assert eps1 == eps2
