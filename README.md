# DP Accelerator

Rust-accelerated differential privacy accounting for machine learning.

[![PyPI version](https://badge.fury.io/py/dp-accelerator.svg)](https://pypi.org/project/dp-accelerator/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

DP Accelerator is a framework-agnostic library for computing differential
privacy guarantees. The core accounting routines are implemented in Rust and
exposed to Python via PyO3, delivering over 3000x speedup compared to
pure-Python baselines while producing numerically identical results.

## Features

- **Renyi DP (RDP) accounting** with Poisson subsampling, sampling without
  replacement, Laplace, randomized response, zCDP, tree aggregation, and
  repeat-and-select mechanisms
- **Analytical Gaussian mechanism** calibration (Balle and Wang, 2018)
- **Privacy Loss Distribution (PLD)** accounting with FFT-based composition
- **DpEvent algebra** for composing heterogeneous mechanism sequences
- **Mechanism calibration** search for optimal noise parameters
- **Framework-agnostic**: works with JAX, PyTorch, TensorFlow, or standalone

## Installation

```bash
pip install dp-accelerator
```

Building from source requires a Rust toolchain (1.70+) and
[maturin](https://github.com/PyO3/maturin):

```bash
git clone https://github.com/AxiomaticLabs/dp-accelerator.git
cd dp-accelerator
pip install maturin
maturin develop --release
```

## Quick Start

### DP-SGD accounting

```python
from dp_accelerator import DPSGDAccountant

accountant = DPSGDAccountant(
    noise_multiplier=1.0,
    batch_size=600,
    dataset_size=60000,
)

epsilon = accountant.get_epsilon(steps=10000, delta=1e-5)
print(f"epsilon = {epsilon:.2f}")
```

### RDP primitives

```python
from dp_accelerator import (
    RdpAccountant,
    GaussianDpEvent,
    PoissonSampledDpEvent,
)

accountant = RdpAccountant()
event = PoissonSampledDpEvent(
    sampling_probability=0.01,
    event=GaussianDpEvent(noise_multiplier=1.0),
)
accountant.compose(event, count=1000)
epsilon = accountant.get_epsilon(target_delta=1e-5)
```

### Gaussian mechanism calibration

```python
from dp_accelerator import get_sigma_gaussian, get_epsilon_gaussian

sigma = get_sigma_gaussian(epsilon=1.0, delta=1e-5)
eps = get_epsilon_gaussian(sigma=sigma, delta=1e-5)
```

### Vectorized batch computation

```python
from dp_accelerator import compute_epsilon_batch

epsilons = compute_epsilon_batch(
    q=0.01,
    noise_multiplier=1.0,
    steps_list=[1000, 5000, 10000, 50000],
    orders=[1.5, 2, 5, 10, 25, 50, 100],
    delta=1e-5,
)
```

## Performance

Benchmarks measured on a single core, comparing `dp_accelerator` against
Google's `dp_accounting` library (v0.4) on identical RDP order sets.

| Operation | dp_accounting | dp_accelerator | Speedup |
|---|---|---|---|
| Single epsilon (1k steps) | 0.6 s | 0.2 ms | 3000x |
| Batch epsilon (100 configs) | 60 s | 0.02 s | 3000x |
| RDP composition | 12 ms | 0.004 ms | 3000x |

Results are numerically identical to within relative tolerance of 1e-6.

## API Reference

### Core Classes

| Class | Description |
|---|---|
| `DPSGDAccountant` | High-level accountant for DP-SGD training loops |
| `RdpAccountant` | General-purpose RDP accountant supporting all DpEvent types |
| `PLDAccountant` | Privacy Loss Distribution accountant via FFT composition |

### Mechanism Functions

| Function | Description |
|---|---|
| `get_epsilon_gaussian(sigma, delta)` | Compute epsilon for a Gaussian mechanism |
| `get_sigma_gaussian(epsilon, delta)` | Calibrate sigma for a target epsilon |
| `compute_rdp_poisson_subsampled_gaussian(q, sigma, orders)` | RDP for Poisson-subsampled Gaussian |
| `compute_rdp_sample_wor_gaussian(q, sigma, orders)` | RDP for sampling without replacement |
| `compute_rdp_laplace(epsilon, orders)` | RDP for pure-epsilon Laplace mechanism |
| `compute_rdp_randomized_response(noise, num_buckets, orders)` | RDP for randomized response |
| `rdp_to_epsilon(orders, rdp_values, delta)` | Convert RDP curve to (epsilon, delta)-DP |
| `rdp_to_delta(orders, rdp_values, epsilon)` | Convert RDP curve to delta for given epsilon |

### DpEvent Types

`GaussianDpEvent`, `LaplaceDpEvent`, `PoissonSampledDpEvent`,
`SampledWithoutReplacementDpEvent`, `SelfComposedDpEvent`,
`ComposedDpEvent`, `RandomizedResponseDpEvent`, `ZCDpEvent`,
`SingleEpochTreeAggregationDpEvent`, `RepeatAndSelectDpEvent`

## Architecture

The library is structured as a Rust core with a Python interface layer:

```
src/
  accounting.rs    RDP computation (Poisson, WOR, Laplace, conversions)
  gaussian.rs      Analytical Gaussian calibration (Balle and Wang)
  pld.rs           Privacy Loss Distribution with FFT convolution
  math.rs          Numerical primitives (log-sum-exp, gamma, erfc)
  lib.rs           PyO3 module bindings

python/dp_accelerator/
  rdp.py           RdpAccountant and RDP primitive wrappers
  dp_event.py      DpEvent class hierarchy
  pld/             PLD accountant and PMF classes
  mechanism_calibration.py
  gaussian_mechanism.py
  jax_privacy.py   Drop-in adapter for JAX Privacy
```

## Development

```bash
# Build and install in development mode
maturin develop --release

# Run Rust tests
cargo test --no-default-features

# Run Python tests
pytest tests/ -v
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.