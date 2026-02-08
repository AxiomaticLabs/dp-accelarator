"""Privacy accountant base class and NeighboringRelation enum.

API-compatible with ``dp_accounting.privacy_accountant``.
"""

from __future__ import annotations

import abc
import enum
import math
from typing import Optional

from dp_accelerator.dp_event import DpEvent, NoOpDpEvent
from dp_accelerator.dp_event_builder import DpEventBuilder


class NeighboringRelation(enum.Enum):
    """Defines the adjacency relation between neighbouring datasets."""

    ADD_OR_REMOVE_ONE = 1
    REPLACE_ONE = 2
    REPLACE_SPECIAL = 3


class UnsupportedEventError(Exception):
    """Raised when compose() is called with an unsupported event type."""


class PrivacyAccountant(abc.ABC):
    """Abstract base class for privacy accountants.

    Matches the ``dp_accounting.privacy_accountant.PrivacyAccountant`` API.
    """

    class CompositionErrorDetails:
        """Describes why a composition failed."""

        __slots__ = ("invalid_event", "error_message")

        def __init__(
            self,
            invalid_event: Optional[DpEvent] = None,
            error_message: Optional[str] = None,
        ):
            self.invalid_event = invalid_event
            self.error_message = error_message

    def __init__(self, neighboring_relation: NeighboringRelation):
        self._neighboring_relation = neighboring_relation
        self._ledger = DpEventBuilder()

    @property
    def neighboring_relation(self) -> NeighboringRelation:
        return self._neighboring_relation

    def supports(self, event: DpEvent) -> bool:
        """Check whether the accountant can process the given event."""
        return self._maybe_compose(event, 0, False) is None

    @abc.abstractmethod
    def _maybe_compose(
        self, event: DpEvent, count: int, do_compose: bool
    ) -> Optional[CompositionErrorDetails]:
        """Traverse event and perform composition if do_compose is True."""
        ...

    def compose(self, event: DpEvent, count: int = 1) -> "PrivacyAccountant":
        """Compose an event into the accountant.

        Args:
            event: A DpEvent instance.
            count: Number of times to compose.

        Returns:
            self (for chaining).

        Raises:
            UnsupportedEventError: If the event type is not supported.
        """
        if not isinstance(event, DpEvent):
            raise TypeError(f"`event` must be `DpEvent`. Found {type(event)}.")
        err = self._maybe_compose(event, count, False)
        if err is not None:
            raise UnsupportedEventError(
                f"Unsupported event: {event}. Error: [{err.error_message}] "
                f"caused by subevent {err.invalid_event}."
            )
        self._ledger.compose(event, count)
        self._maybe_compose(event, count, True)
        return self

    @property
    def ledger(self) -> DpEvent:
        """Return the composed DpEvent processed so far."""
        return self._ledger.build()

    @abc.abstractmethod
    def get_epsilon(self, target_delta: float) -> float:
        """Get current epsilon for the given delta."""
        ...

    def get_delta(self, target_epsilon: float) -> float:
        """Get current delta for the given epsilon."""
        raise NotImplementedError()
