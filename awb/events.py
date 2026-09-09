from __future__ import annotations

import heapq
from collections.abc import Callable

from .types import WorldEvent


class EventQueue:
    def __init__(self) -> None:
        self._heap: list[WorldEvent] = []
        self._sequence = 0

    def push(
        self,
        timestamp_ms: int,
        event_type: str,
        *,
        source_id: int | None = None,
        target_id: int | None = None,
        location_id: int | None = None,
        payload: dict | None = None,
    ) -> WorldEvent:
        event = WorldEvent(
            timestamp_ms=timestamp_ms,
            sequence=self._sequence,
            event_type=event_type,
            source_id=source_id,
            target_id=target_id,
            location_id=location_id,
            payload=payload or {},
        )
        self._sequence += 1
        heapq.heappush(self._heap, event)
        return event

    def pop(self) -> WorldEvent:
        return heapq.heappop(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[[WorldEvent], None]]] = {}

    def subscribe(self, event_type: str, handler: Callable[[WorldEvent], None]) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def dispatch(self, event: WorldEvent) -> None:
        for handler in self._handlers.get(event.event_type, ()):
            handler(event)
