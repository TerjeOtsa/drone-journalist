"""
Lightweight publish-subscribe event bus.
All modules post and subscribe to events through a single global instance.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable, List

from interfaces.schemas import SystemEvent

log = logging.getLogger(__name__)

# Type alias for clarity
EventHandler = Callable[[SystemEvent], None]


class EventBus:
    """Simple synchronous event bus (single-threaded companion computer)."""

    def __init__(self) -> None:
        self._subscribers: dict[str, List[EventHandler]] = defaultdict(list)
        self._history: List[SystemEvent] = []
        self._max_history = 500

    # ── public API ───────────────────────────────────────────────────────

    def subscribe(self, event_name: str, handler: EventHandler) -> None:
        """Register *handler* to be called whenever *event_name* is published."""
        self._subscribers[event_name].append(handler)

    def publish(self, event: SystemEvent) -> None:
        """Dispatch *event* to subscribers and append it to the history ring."""
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        for handler in self._subscribers.get(event.event, []):
            try:
                handler(event)
            except Exception:
                log.exception("Event handler failed for %s", event.event)

    def last(self, event_name: str) -> SystemEvent | None:
        """Return the most recent event with the given name, or ``None``."""
        for e in reversed(self._history):
            if e.event == event_name:
                return e
        return None

    def clear(self) -> None:
        """Remove all subscribers and clear the event history."""
        self._subscribers.clear()
        self._history.clear()


# Singleton used across the system
bus = EventBus()
