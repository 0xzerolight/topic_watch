"""Process-global mutable state for the web layer.

Centralizes the in-memory state that used to live as module globals in
the original web routes module: the in-progress check tracker, the
dashboard stats cache, and the feed-validation rate limiter. Mutations
are guarded with ``asyncio.Lock`` where concurrent access is possible.
"""

import asyncio
import time

# --- In-progress check tracker ---


class CheckingState:
    """Async-safe state tracker for in-progress topic checks."""

    def __init__(self) -> None:
        self._topics: set[int] = set()
        self._start_times: dict[int, float] = {}
        self._checking_all: bool = False
        self._lock = asyncio.Lock()

    async def start_check(self, topic_id: int) -> bool:
        """Mark topic as being checked. Returns False if already checking."""
        async with self._lock:
            if topic_id in self._topics:
                return False
            self._topics.add(topic_id)
            self._start_times[topic_id] = time.monotonic()
            return True

    async def finish_check(self, topic_id: int) -> None:
        """Mark topic check as finished."""
        async with self._lock:
            self._topics.discard(topic_id)
            self._start_times.pop(topic_id, None)

    async def is_checking(self, topic_id: int) -> bool:
        """Return True if topic is currently being checked."""
        async with self._lock:
            return topic_id in self._topics

    async def start_check_all(self) -> bool:
        """Mark check-all as running. Returns False if already running."""
        async with self._lock:
            if self._checking_all:
                return False
            self._checking_all = True
            return True

    async def finish_check_all(self) -> None:
        """Mark check-all as finished."""
        async with self._lock:
            self._checking_all = False

    async def is_checking_all(self) -> bool:
        """Return True if a check-all is currently running."""
        async with self._lock:
            return self._checking_all

    async def clear_stale(self, timeout_seconds: float) -> list[int]:
        """Remove topic entries older than timeout_seconds. Returns cleared IDs."""
        now = time.monotonic()
        async with self._lock:
            stale = [tid for tid, start in self._start_times.items() if now - start > timeout_seconds]
            for tid in stale:
                self._topics.discard(tid)
                self._start_times.pop(tid, None)
        return stale


_checking_state = CheckingState()


# --- Dashboard stats cache (single-worker safe) ---

_stats_cache: dict = {"data": None, "expires": 0.0}
_STATS_CACHE_TTL = 60  # seconds


# --- Feed-validation rate limiter ---

_rate_limit_store: dict[str, list[float]] = {}
_RATE_LIMIT_MAX = 10
_RATE_LIMIT_WINDOW = 60  # seconds
_RATE_LIMIT_MAX_IPS = 10000  # hard cap on tracked IPs to bound memory


def _check_rate_limit(ip: str) -> bool:
    """Check if IP is within rate limit. Returns True if allowed.

    Evicts entries whose timestamps have all fallen outside the window so the
    store cannot grow without bound (one entry per IP would otherwise leak).
    """
    now = time.time()
    timestamps = _rate_limit_store.get(ip, [])
    active = [t for t in timestamps if now - t < _RATE_LIMIT_WINDOW]
    if len(active) >= _RATE_LIMIT_MAX:
        _rate_limit_store[ip] = active
        return False
    active.append(now)
    _rate_limit_store[ip] = active
    # Evict stale IPs (all timestamps outside the window) to bound memory.
    if len(_rate_limit_store) > _RATE_LIMIT_MAX_IPS:
        stale = [k for k, v in _rate_limit_store.items() if not v or now - v[-1] >= _RATE_LIMIT_WINDOW]
        for k in stale:
            del _rate_limit_store[k]
    return True
