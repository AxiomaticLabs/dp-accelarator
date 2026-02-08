"""Privacy accountant based on Privacy Loss Distributions.

API-compatible with ``dp_accounting.pld.PLDAccountant``.
"""

from __future__ import annotations

import math
from typing import Optional

from dp_accelerator.dp_event import (
    ComposedDpEvent,
    DiscreteLaplaceDpEvent,
    DpEvent,
    GaussianDpEvent,
    LaplaceDpEvent,
    MixtureOfGaussiansDpEvent,
    NoOpDpEvent,
    NonPrivateDpEvent,
    PoissonSampledDpEvent,
    RandomizedResponseDpEvent,
    SelfComposedDpEvent,
    TruncatedSubsampledGaussianDpEvent,
)
from dp_accelerator.pld.privacy_loss_distribution import PrivacyLossDistribution
from dp_accelerator.privacy_accountant import (
    NeighboringRelation,
    PrivacyAccountant,
)

PLD = PrivacyLossDistribution


class PLDAccountant(PrivacyAccountant):
    """Privacy accountant using Privacy Loss Distributions.

    Provides tighter privacy bounds than RDP for some regimes, at the cost
    of higher computational overhead.

    Example::

        accountant = PLDAccountant()
        accountant.compose(GaussianDpEvent(1.0), count=100)
        eps = accountant.get_epsilon(target_delta=1e-5)
    """

    def __init__(
        self,
        neighboring_relation: NeighboringRelation = NeighboringRelation.ADD_OR_REMOVE_ONE,
        value_discretization_interval: float = 1e-4,
    ):
        super().__init__(neighboring_relation)
        self._contains_non_dp_event = False
        self._pld = PLD.identity(
            value_discretization_interval=value_discretization_interval
        )
        self._value_discretization_interval = value_discretization_interval

    def _maybe_compose(
        self, event: DpEvent, count: int, do_compose: bool
    ) -> Optional[PrivacyAccountant.CompositionErrorDetails]:
        """Process a DpEvent for composition."""
        CompositionError = PrivacyAccountant.CompositionErrorDetails

        if isinstance(event, NoOpDpEvent):
            return None

        elif isinstance(event, NonPrivateDpEvent):
            if do_compose:
                self._contains_non_dp_event = True
            return None

        elif isinstance(event, SelfComposedDpEvent):
            return self._maybe_compose(event.event, event.count * count, do_compose)

        elif isinstance(event, ComposedDpEvent):
            for e in event.events:
                result = self._maybe_compose(e, count, do_compose)
                if result is not None:
                    return result
            return None

        elif isinstance(event, GaussianDpEvent):
            if do_compose:
                if event.noise_multiplier == 0:
                    self._contains_non_dp_event = True
                else:
                    pld = PLD.from_gaussian_mechanism(
                        standard_deviation=event.noise_multiplier,
                        value_discretization_interval=self._value_discretization_interval,
                        neighboring_relation=self._neighboring_relation,
                    )
                    if count > 1:
                        pld = pld.self_compose(count)
                    self._pld = self._pld.compose(pld)
            return None

        elif isinstance(event, LaplaceDpEvent):
            if self._neighboring_relation not in (
                NeighboringRelation.ADD_OR_REMOVE_ONE,
                NeighboringRelation.REPLACE_SPECIAL,
            ):
                return CompositionError(
                    invalid_event=event,
                    error_message=(
                        "neighboring_relation must be ADD_OR_REMOVE_ONE or "
                        f"REPLACE_SPECIAL for LaplaceDpEvent. Found {self._neighboring_relation}."
                    ),
                )
            if do_compose:
                if event.noise_multiplier == 0:
                    self._contains_non_dp_event = True
                else:
                    pld = PLD.from_laplace_mechanism(
                        parameter=event.noise_multiplier,
                        value_discretization_interval=self._value_discretization_interval,
                    ).self_compose(count)
                    self._pld = self._pld.compose(pld)
            return None

        elif isinstance(event, DiscreteLaplaceDpEvent):
            if self._neighboring_relation not in (
                NeighboringRelation.ADD_OR_REMOVE_ONE,
                NeighboringRelation.REPLACE_SPECIAL,
            ):
                return CompositionError(
                    invalid_event=event,
                    error_message=(
                        "neighboring_relation must be ADD_OR_REMOVE_ONE or "
                        f"REPLACE_SPECIAL for DiscreteLaplaceDpEvent. Found {self._neighboring_relation}."
                    ),
                )
            if do_compose:
                if event.noise_parameter == 0:
                    self._contains_non_dp_event = True
                else:
                    pld = PLD.from_discrete_laplace_mechanism(
                        parameter=event.noise_parameter,
                        sensitivity=event.sensitivity,
                        value_discretization_interval=self._value_discretization_interval,
                    ).self_compose(count)
                    self._pld = self._pld.compose(pld)
            return None

        elif isinstance(event, RandomizedResponseDpEvent):
            if self._neighboring_relation not in (
                NeighboringRelation.REPLACE_ONE,
                NeighboringRelation.REPLACE_SPECIAL,
            ):
                return CompositionError(
                    invalid_event=event,
                    error_message=(
                        "neighboring_relation must be REPLACE_ONE or REPLACE_SPECIAL "
                        f"for RandomizedResponseDpEvent. Found {self._neighboring_relation}."
                    ),
                )
            if do_compose:
                if event.num_buckets == 1:
                    pass
                elif event.noise_parameter == 0:
                    self._contains_non_dp_event = True
                else:
                    pld = PLD.from_randomized_response(
                        noise_parameter=event.noise_parameter,
                        num_buckets=event.num_buckets,
                        value_discretization_interval=self._value_discretization_interval,
                        neighboring_relation=self._neighboring_relation,
                    )
                    self._pld = self._pld.compose(pld)
            return None

        elif isinstance(event, MixtureOfGaussiansDpEvent):
            if self._neighboring_relation not in (
                NeighboringRelation.ADD_OR_REMOVE_ONE,
                NeighboringRelation.REPLACE_SPECIAL,
            ):
                return CompositionError(
                    invalid_event=event,
                    error_message=(
                        "neighboring_relation must be ADD_OR_REMOVE_ONE or "
                        f"REPLACE_SPECIAL for MixtureOfGaussiansDpEvent. Found {self._neighboring_relation}."
                    ),
                )
            if do_compose:
                if len(event.sensitivities) == 1 and event.sensitivities[0] == 0.0:
                    pass
                elif event.standard_deviation == 0:
                    self._contains_non_dp_event = True
                else:
                    pld = PLD.from_mixture_gaussian_mechanism(
                        standard_deviation=event.standard_deviation,
                        sensitivities=event.sensitivities,
                        sampling_probs=event.sampling_probs,
                        value_discretization_interval=self._value_discretization_interval,
                    ).self_compose(count)
                    self._pld = self._pld.compose(pld)
            return None

        elif isinstance(event, PoissonSampledDpEvent):
            if isinstance(event.event, GaussianDpEvent):
                if do_compose:
                    if event.sampling_probability == 0:
                        pass
                    elif event.event.noise_multiplier == 0:
                        self._contains_non_dp_event = True
                    else:
                        pld = PLD.from_gaussian_mechanism(
                            standard_deviation=event.event.noise_multiplier,
                            value_discretization_interval=self._value_discretization_interval,
                            sampling_prob=event.sampling_probability,
                            neighboring_relation=self._neighboring_relation,
                        ).self_compose(count)
                        self._pld = self._pld.compose(pld)
                return None
            elif isinstance(event.event, LaplaceDpEvent):
                if self._neighboring_relation not in (
                    NeighboringRelation.ADD_OR_REMOVE_ONE,
                    NeighboringRelation.REPLACE_SPECIAL,
                ):
                    return CompositionError(
                        invalid_event=event,
                        error_message=(
                            "neighboring_relation must be ADD_OR_REMOVE_ONE or "
                            "REPLACE_SPECIAL for PoissonSampled LaplaceDpEvent."
                        ),
                    )
                if do_compose:
                    if event.sampling_probability == 0:
                        pass
                    elif event.event.noise_multiplier == 0:
                        self._contains_non_dp_event = True
                    else:
                        pld = PLD.from_laplace_mechanism(
                            parameter=event.event.noise_multiplier,
                            value_discretization_interval=self._value_discretization_interval,
                            sampling_prob=event.sampling_probability,
                        ).self_compose(count)
                        self._pld = self._pld.compose(pld)
                return None
            else:
                return CompositionError(
                    invalid_event=event,
                    error_message=(
                        "Subevent of PoissonSampledEvent must be GaussianDpEvent "
                        f"or LaplaceDpEvent. Found {event.event}."
                    ),
                )

        elif isinstance(event, TruncatedSubsampledGaussianDpEvent):
            if do_compose:
                if (
                    event.sampling_probability == 0
                    or event.truncated_batch_size == 0
                    or event.dataset_size == 0
                ):
                    pass
                elif event.noise_multiplier == 0:
                    self._contains_non_dp_event = True
                else:
                    pld = PLD.from_truncated_subsampled_gaussian_mechanism(
                        dataset_size=event.dataset_size,
                        sampling_probability=event.sampling_probability,
                        truncated_batch_size=event.truncated_batch_size,
                        noise_multiplier=event.noise_multiplier,
                        value_discretization_interval=self._value_discretization_interval,
                        neighboring_relation=self._neighboring_relation,
                    ).self_compose(count)
                    self._pld = self._pld.compose(pld)
            return None

        else:
            return CompositionError(
                invalid_event=event, error_message="Unsupported event."
            )

    def get_epsilon(self, target_delta: float) -> float:
        if self._contains_non_dp_event:
            return math.inf
        return self._pld.get_epsilon_for_delta(target_delta)

    def get_delta(self, target_epsilon: float) -> float:
        if self._contains_non_dp_event:
            return 1.0
        result = self._pld.get_delta_for_epsilon(target_epsilon)
        return float(result)
