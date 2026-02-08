# DP Accelerator

Universal High-Performance Differential Privacy Accounting Engine

[![PyPI version](https://badge.fury.io/py/dp-accelerator.svg)](https://pypi.org/project/dp-accelerator/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A framework-agnostic Rust-accelerated library for computing differential privacy guarantees with **3000x+ speedup** over pure Python implementations.

## Features

- 🚀 **3000x faster** than pure Python DP accounting
- 🔧 **Framework-agnostic**: Works with JAX, PyTorch, TensorFlow
- 🦀 **Rust-powered**: Zero-cost abstractions with memory safety
- 📦 **Easy installation**: `pip install dp-accelerator`
- 🎯 **Drop-in replacement**: Compatible APIs for existing libraries

## Quick Start

```python
from dp_accelerator import DPSGDAccountant

# Initialize accountant
accountant = DPSGDAccountant(
    noise_multiplier=1.0,
    batch_size=600,
    dataset_size=60000
)

# Compute privacy guarantee
epsilon = accountant.get_epsilon(steps=10000, delta=1e-5)
print(f"Privacy guarantee: ε = {epsilon:.2f}")
```

## Framework Adapters

### JAX Privacy
```python
from dp_accelerator.jax_adapter import compute_dpsgd_epsilon

epsilon = compute_dpsgd_epsilon(
    noise_multiplier=1.0,
    batch_size=600,
    dataset_size=60000,
    num_steps=10000,
    delta=1e-5
)
```

## Performance

| Implementation | Time | Speedup |
|----------------|------|---------|
| Pure Python | 0.613s | 1x |
| **DP Accelerator** | **0.0002s** | **3000x** |

## Installation

```bash
pip install dp-accelerator
```

## Development

```bash
git clone https://github.com/yourusername/dp-accelerator
cd dp-accelerator
maturin develop
```

## Contributing

Contributions welcome! Please see our [contributing guide](CONTRIBUTING.md).

## License

Apache License 2.0