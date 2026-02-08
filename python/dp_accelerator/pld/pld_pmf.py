"""PMF (Probability Mass Function) for Privacy Loss Distributions.

All heavy computation (FFT convolution, self-composition, epsilon/delta
queries) is delegated to the Rust ``RustPldPmf`` backend for maximum
performance.  The Python classes here are thin wrappers that maintain
API compatibility.
"""

from __future__ import annotations

import abc
import math
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from dp_accelerator._core import RustPldPmf

_MAX_PMF_SPARSE_SIZE = 1000


class PLDPmf(abc.ABC):
    """Abstract base class for PLD probability mass functions."""

    @abc.abstractmethod
    def get_delta_for_epsilon(
        self, epsilon: Union[float, Sequence[float]]
    ) -> Union[float, np.ndarray]:
        """Compute the hockey stick divergence at the given epsilon(s)."""
        ...

    @abc.abstractmethod
    def get_epsilon_for_delta(self, delta: float) -> float:
        """Find smallest epsilon with hockey-stick divergence <= delta."""
        ...

    @abc.abstractmethod
    def compose(self, other: "PLDPmf") -> "PLDPmf":
        """Compose (convolve) this PMF with another."""
        ...

    @abc.abstractmethod
    def self_compose(self, count: int) -> "PLDPmf":
        """Self-compose count times (power-of-two FFT)."""
        ...

    @property
    @abc.abstractmethod
    def size(self) -> int:
        ...


class DensePLDPmf(PLDPmf):
    """Dense PMF backed by a Rust ``RustPldPmf``.

    All arithmetic (FFT convolution, truncation, epsilon/delta queries)
    runs in optimised Rust — Python only orchestrates.
    """

    def __init__(
        self,
        probs: np.ndarray,
        lower_loss: int,
        discretization_interval: float,
        infinity_mass: float = 0.0,
        pessimistic_estimate: bool = True,
    ):
        if isinstance(probs, np.ndarray):
            probs_list = probs.tolist()
        else:
            probs_list = list(probs)
        self._rust = RustPldPmf(
            probs_list, int(lower_loss),
            float(discretization_interval), float(infinity_mass),
        )
        self._pessimistic = pessimistic_estimate

    @classmethod
    def _from_rust(cls, rust_pmf: RustPldPmf,
                   pessimistic: bool = True) -> "DensePLDPmf":
        """Wrap an existing Rust PMF without data copy."""
        obj = cls.__new__(cls)
        obj._rust = rust_pmf
        obj._pessimistic = pessimistic
        return obj

    @classmethod
    def from_gaussian(cls, sigma: float, sensitivity: float, di: float,
                      tail_bound: float = 10.0,
                      is_add: bool = True) -> "DensePLDPmf":
        """Build Gaussian PMF via Rust connect-the-dots."""
        return cls._from_rust(
            RustPldPmf.from_gaussian(sigma, sensitivity, di, tail_bound, is_add)
        )

    @classmethod
    def from_laplace(cls, parameter: float, sensitivity: float, di: float,
                     tail_bound: float = 10.0) -> "DensePLDPmf":
        """Build Laplace PMF via Rust."""
        return cls._from_rust(
            RustPldPmf.from_laplace(parameter, sensitivity, di, tail_bound)
        )

    # ── Properties ───────────────────────────────────────────────

    @property
    def size(self) -> int:
        return self._rust.size

    @property
    def _discretization_interval(self) -> float:
        return self._rust.di

    @property
    def _infinity_mass(self) -> float:
        return self._rust.infinity_mass_val

    @property
    def _lower_loss(self) -> int:
        return self._rust.lower_loss

    # ── Queries (delegated to Rust) ──────────────────────────────

    def get_delta_for_epsilon(
        self, epsilon: Union[float, Sequence[float]]
    ) -> Union[float, np.ndarray]:
        scalar = isinstance(epsilon, (int, float))
        if scalar:
            return self._rust.get_delta_for_epsilon(float(epsilon))
        eps_list = [float(e) for e in np.atleast_1d(epsilon)]
        return np.array(self._rust.get_delta_for_epsilon_list(eps_list))

    def get_epsilon_for_delta(self, delta: float) -> float:
        return self._rust.get_epsilon_for_delta(float(delta))

    # ── Composition (delegated to Rust FFT) ──────────────────────

    def compose(self, other: PLDPmf) -> "DensePLDPmf":
        if isinstance(other, SparsePLDPmf):
            other = other._to_dense()
        if not isinstance(other, DensePLDPmf):
            raise TypeError("Can only compose DensePLDPmf with DensePLDPmf")
        return DensePLDPmf._from_rust(
            self._rust.compose(other._rust), self._pessimistic
        )

    def self_compose(self, count: int) -> "DensePLDPmf":
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")
        return DensePLDPmf._from_rust(
            self._rust.self_compose(count), self._pessimistic
        )


class SparsePLDPmf(PLDPmf):
    """Sparse dict-backed PMF for distributions with a small support."""

    def __init__(
        self,
        loss_probs: Mapping[int, float],
        discretization_interval: float,
        infinity_mass: float = 0.0,
        pessimistic_estimate: bool = True,
    ):
        self._loss_probs = dict(loss_probs)
        self._discretization_interval = discretization_interval
        self._infinity_mass = infinity_mass
        self._pessimistic = pessimistic_estimate

    @property
    def size(self) -> int:
        return len(self._loss_probs)

    def _to_dense(self) -> DensePLDPmf:
        if not self._loss_probs:
            return DensePLDPmf(
                np.array([1.0]),
                0,
                self._discretization_interval,
                self._infinity_mass,
                self._pessimistic,
            )
        indices = sorted(self._loss_probs.keys())
        lower = indices[0]
        upper = indices[-1]
        n = upper - lower + 1
        probs = np.zeros(n)
        for idx, prob in self._loss_probs.items():
            probs[idx - lower] = prob
        return DensePLDPmf(
            probs, lower, self._discretization_interval,
            self._infinity_mass, self._pessimistic,
        )

    def get_delta_for_epsilon(
        self, epsilon: Union[float, Sequence[float]]
    ) -> Union[float, np.ndarray]:
        return self._to_dense().get_delta_for_epsilon(epsilon)

    def get_epsilon_for_delta(self, delta: float) -> float:
        return self._to_dense().get_epsilon_for_delta(delta)

    def compose(self, other: PLDPmf) -> PLDPmf:
        # Convert to dense for composition
        dense_self = self._to_dense()
        if isinstance(other, SparsePLDPmf):
            other = other._to_dense()
        return dense_self.compose(other)

    def self_compose(self, count: int) -> PLDPmf:
        return self._to_dense().self_compose(count)


def create_pmf(
    rounded_pmf: Mapping[int, float],
    discretization_interval: float,
    infinity_mass: float,
    pessimistic_estimate: bool = True,
) -> PLDPmf:
    """Create a PLDPmf from a rounded probability mass function.

    Chooses DensePLDPmf or SparsePLDPmf based on support size.
    """
    if not rounded_pmf:
        return SparsePLDPmf({}, discretization_interval, infinity_mass, pessimistic_estimate)

    indices = sorted(rounded_pmf.keys())
    span = indices[-1] - indices[0] + 1

    if span <= _MAX_PMF_SPARSE_SIZE or len(rounded_pmf) < span * 0.5:
        return SparsePLDPmf(
            rounded_pmf, discretization_interval, infinity_mass, pessimistic_estimate
        )

    lower = indices[0]
    probs = np.zeros(span)
    for idx, prob in rounded_pmf.items():
        probs[idx - lower] = prob
    return DensePLDPmf(
        probs, lower, discretization_interval, infinity_mass, pessimistic_estimate
    )
