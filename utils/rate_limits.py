from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AsyncSlidingWindowRateLimiter:
    limit: float
    window_seconds: float
    name: str
    enabled: bool = True
    _events: deque[tuple[float, float]] = field(default_factory=deque)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def acquire(self, amount: float = 1.0) -> None:
        if not self.enabled or self.limit <= 0 or self.window_seconds <= 0:
            return

        amount = max(0.0, min(float(amount), self.limit))
        while True:
            async with self._lock:
                now = time.monotonic()
                self._prune(now)
                used = sum(value for _, value in self._events)
                if used + amount <= self.limit:
                    self._events.append((now, amount))
                    return

                oldest_time = self._events[0][0] if self._events else now
                wait_seconds = max(0.25, self.window_seconds - (now - oldest_time) + 0.1)

            logger.info(
                "Rate limit wait for %s: %.1fs (requested=%.0f, limit=%.0f/%ss)",
                self.name,
                wait_seconds,
                amount,
                self.limit,
                int(self.window_seconds),
            )
            await asyncio.sleep(wait_seconds)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._events and self._events[0][0] <= cutoff:
            self._events.popleft()
