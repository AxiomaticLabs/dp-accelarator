"""Standard DpEvent classes for differential privacy accounting.

A DpEvent represents the (hyper)parameters of a differentially private query,
amplification mechanism, or composition. API-compatible with
``dp_accounting.dp_event``.
"""

from __future__ import annotations

from typing import List, Sequence, Union


class DpEvent:
    """Base class for all differential-privacy events."""

    def __eq__(self, other):
        return type(self) is type(other) and self.__dict__ == other.__dict__

    def __repr__(self):
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{type(self).__name__}({attrs})"


class NoOpDpEvent(DpEvent):
    """Operation with no privacy impact."""


class NonPrivateDpEvent(DpEvent):
    """Non-private operation (infinite epsilon)."""


class UnsupportedDpEvent(DpEvent):
    """Unsupported / unknown operation."""


class GaussianDpEvent(DpEvent):
    """Application of the Gaussian mechanism.

    Attributes:
        noise_multiplier: Ratio of noise std to sensitivity (sigma / C).
    """

    def __init__(self, noise_multiplier: float):
        self.noise_multiplier = float(noise_multiplier)


class LaplaceDpEvent(DpEvent):
    """Application of the Laplace mechanism.

    Attributes:
        noise_multiplier: Ratio of Laplace scale to sensitivity (b / C).
    """

    def __init__(self, noise_multiplier: float):
        self.noise_multiplier = float(noise_multiplier)


class DiscreteLaplaceDpEvent(DpEvent):
    """Application of the Discrete Laplace mechanism.

    Attributes:
        noise_parameter: The noise parameter of DLap(a).
        sensitivity: L1 sensitivity.
    """

    def __init__(self, noise_parameter: float, sensitivity: int = 1):
        self.noise_parameter = float(noise_parameter)
        self.sensitivity = int(sensitivity)


class RandomizedResponseDpEvent(DpEvent):
    """Randomized response over k buckets with noise parameter p.

    Attributes:
        noise_parameter: Probability of outputting a random bucket.
        num_buckets: Number of buckets.
    """

    def __init__(self, noise_parameter: float, num_buckets: int):
        self.noise_parameter = float(noise_parameter)
        self.num_buckets = int(num_buckets)


class ZCDpEvent(DpEvent):
    """Mechanism satisfying (xi, rho)-zCDP.

    Satisfies (alpha, xi + rho * alpha)-RDP for all alpha > 0.

    Attributes:
        rho: Multiplicative constant.
        xi: Additive constant (default 0).
    """

    def __init__(self, rho: float, xi: float = 0.0):
        self.rho = float(rho)
        self.xi = float(xi)


class SelfComposedDpEvent(DpEvent):
    """Repeated application of a mechanism.

    Attributes:
        event: The underlying DpEvent.
        count: Number of repetitions.
    """

    def __init__(self, event: DpEvent, count: int):
        self.event = event
        self.count = int(count)


class ComposedDpEvent(DpEvent):
    """Sequential composition of multiple mechanisms.

    Attributes:
        events: List of composed DpEvents.
    """

    def __init__(self, events: List[DpEvent]):
        self.events = list(events)


class PoissonSampledDpEvent(DpEvent):
    """Poisson subsampling followed by a mechanism.

    Attributes:
        sampling_probability: Probability each record is included.
        event: The mechanism applied to the sample.
    """

    def __init__(self, sampling_probability: float, event: DpEvent):
        self.sampling_probability = float(sampling_probability)
        self.event = event


class SampledWithReplacementDpEvent(DpEvent):
    """Sampling with replacement.

    Attributes:
        source_dataset_size: Size of the source dataset.
        sample_size: Number of (possibly repeated) records drawn.
        event: The mechanism applied to the sample.
    """

    def __init__(self, source_dataset_size: int, sample_size: int, event: DpEvent):
        self.source_dataset_size = int(source_dataset_size)
        self.sample_size = int(sample_size)
        self.event = event


class SampledWithoutReplacementDpEvent(DpEvent):
    """Sampling without replacement.

    Attributes:
        source_dataset_size: Size of the source dataset.
        sample_size: Number of unique records drawn.
        event: The mechanism applied to the sample.
    """

    def __init__(self, source_dataset_size: int, sample_size: int, event: DpEvent):
        self.source_dataset_size = int(source_dataset_size)
        self.sample_size = int(sample_size)
        self.event = event


class SingleEpochTreeAggregationDpEvent(DpEvent):
    """Single-epoch tree aggregation.

    See "Practical and Private (Deep) Learning without Sampling or Shuffling"
    https://arxiv.org/abs/2103.00039.

    Attributes:
        noise_multiplier: Ratio of noise per node to sensitivity.
        step_counts: Number of steps in each tree (int or list of int).
    """

    def __init__(self, noise_multiplier: float, step_counts: Union[int, List[int]]):
        self.noise_multiplier = float(noise_multiplier)
        self.step_counts = step_counts


class RepeatAndSelectDpEvent(DpEvent):
    """Repeatedly running a mechanism and selecting the best output.

    The total number of runs follows a distribution parameterized by mean and
    shape: Poisson (shape=inf), Geometric (shape=1), Logarithmic (shape=0),
    or Truncated Negative Binomial (0 < shape < inf).

    See https://arxiv.org/abs/2110.03620.

    Attributes:
        event: The underlying mechanism.
        mean: Mean number of repetitions.
        shape: Distribution shape parameter.
    """

    def __init__(self, event: DpEvent, mean: float, shape: float):
        self.event = event
        self.mean = float(mean)
        self.shape = float(shape)


class MixtureOfGaussiansDpEvent(DpEvent):
    """Mixture of Gaussians mechanism.

    See https://arxiv.org/abs/2310.15526.

    Attributes:
        standard_deviation: Noise standard deviation.
        sensitivities: Support of sensitivity random variable.
        sampling_probs: Probabilities for each sensitivity.
    """

    def __init__(
        self,
        standard_deviation: float,
        sensitivities: Sequence[float],
        sampling_probs: Sequence[float],
    ):
        self.standard_deviation = float(standard_deviation)
        self.sensitivities = list(sensitivities)
        self.sampling_probs = list(sampling_probs)


class TruncatedSubsampledGaussianDpEvent(DpEvent):
    """Gaussian mechanism with truncated Poisson sampling.

    See https://arxiv.org/abs/2508.15089.

    Attributes:
        dataset_size: Size of the dataset.
        sampling_probability: Probability of sampling each record.
        truncated_batch_size: Max records in a batch.
        noise_multiplier: Gaussian noise multiplier.
    """

    def __init__(
        self,
        dataset_size: int,
        sampling_probability: float,
        truncated_batch_size: int,
        noise_multiplier: float,
    ):
        self.dataset_size = int(dataset_size)
        self.sampling_probability = float(sampling_probability)
        self.truncated_batch_size = int(truncated_batch_size)
        self.noise_multiplier = float(noise_multiplier)
