"""Builder class for composing DpEvents into a ComposedDpEvent.

API-compatible with ``dp_accounting.dp_event_builder``.
"""

from __future__ import annotations

from dp_accelerator.dp_event import (
    ComposedDpEvent,
    DpEvent,
    NoOpDpEvent,
    SelfComposedDpEvent,
)


class DpEventBuilder:
    """Constructs a DpEvent representing the composition of a series of events.

    Example::

        builder = DpEventBuilder()
        builder.compose(GaussianDpEvent(1.0), count=100)
        builder.compose(LaplaceDpEvent(0.5))
        event = builder.build()
    """

    def __init__(self):
        self._event_counts: list[tuple[DpEvent, int]] = []
        self._composed_event: DpEvent | None = None

    def compose(self, event: DpEvent, count: int = 1) -> None:
        """Compose a new event into the builder.

        Args:
            event: The DpEvent to compose.
            count: Number of times to compose this event.
        """
        if not isinstance(event, DpEvent):
            raise TypeError(
                f"`event` must be a subclass of `DpEvent`. Found {type(event)}."
            )
        if not isinstance(count, int):
            raise TypeError(f"`count` must be an integer. Found {type(count)}.")
        if count < 1:
            raise ValueError(f"`count` must be positive. Found {count}.")

        if isinstance(event, NoOpDpEvent):
            return
        elif isinstance(event, SelfComposedDpEvent):
            self.compose(event.event, count * event.count)
        else:
            if self._event_counts and self._event_counts[-1][0] == event:
                old_event, old_count = self._event_counts[-1]
                self._event_counts[-1] = (old_event, old_count + count)
            else:
                self._event_counts.append((event, count))
            self._composed_event = None

    def build(self) -> DpEvent:
        """Build and return the composed DpEvent."""
        if not self._composed_event:
            events = []
            for event, count in self._event_counts:
                if count == 1:
                    events.append(event)
                else:
                    events.append(SelfComposedDpEvent(event, count))
            if not events:
                self._composed_event = NoOpDpEvent()
            elif len(events) == 1:
                self._composed_event = events[0]
            else:
                self._composed_event = ComposedDpEvent(events)
        return self._composed_event
