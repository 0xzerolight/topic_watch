"""Process-global mutable state for the web layer.

Centralizes the in-memory state that used to live as module globals in
the original web routes module: the in-progress check tracker and the
feed-validation rate limiter. Mutations are guarded with ``asyncio.Lock``
where concurrent access is possible.
"""

import asyncio
import secrets
import time

# --- In-progress check tracker ---


def _new_owner_token() -> str:
    """Mint an unguessable, never-repeated owner token for one acquisition."""
    return secrets.token_hex(8)


class CheckingState:
    """Async-safe ownership tracker for in-progress topic checks.

    Each acquisition gets its own token and a release only takes effect when it
    presents the token that is currently held. Releases used to match on topic id
    alone, which made ``clear_stale`` an ABA hole: a legitimate check running past
    the callers' 600-second eviction threshold stayed live, a second owner took
    the freed slot, and the first owner's release then evicted the *second*
    owner's guard — letting a third check in alongside it (AUG-264). With tokens,
    a release from an evicted owner is simply a no-op, so eviction can never hand
    a live slot away.
    """

    def __init__(self) -> None:
        self._topics: dict[int, str] = {}
        self._start_times: dict[int, float] = {}
        self._checking_all: str | None = None
        self._lock = asyncio.Lock()

    async def start_check(self, topic_id: int) -> str | None:
        """Claim the per-topic slot. Returns the owner token, or None if busy."""
        async with self._lock:
            if topic_id in self._topics:
                return None
            token = _new_owner_token()
            self._topics[topic_id] = token
            self._start_times[topic_id] = time.monotonic()
            return token

    async def finish_check(self, topic_id: int, token: str) -> None:
        """Release the per-topic slot. No-op unless ``token`` still owns it."""
        async with self._lock:
            if self._topics.get(topic_id) != token:
                return
            del self._topics[topic_id]
            self._start_times.pop(topic_id, None)

    async def is_checking(self, topic_id: int) -> bool:
        """Return True if topic is currently being checked."""
        async with self._lock:
            return topic_id in self._topics

    def start_check_all(self) -> str | None:
        """Claim the whole-cycle gate. Returns the owner token, or None if busy.

        Synchronous: this is a single test-and-set over one attribute with no
        await in between, so the event loop cannot interleave another claim.
        """
        if self._checking_all is not None:
            return None
        token = _new_owner_token()
        self._checking_all = token
        return token

    def finish_check_all(self, token: str) -> None:
        """Release the whole-cycle gate. No-op unless ``token`` still owns it."""
        if self._checking_all == token:
            self._checking_all = None

    async def is_checking_all(self) -> bool:
        """Return True if a check-all is currently running."""
        return self._checking_all is not None

    async def clear_stale(self, timeout_seconds: float) -> list[int]:
        """Remove topic entries older than timeout_seconds. Returns cleared IDs.

        Safe now that releases are token-checked: evicting a still-running owner
        costs that owner nothing but its slot, and its later release cannot touch
        whoever took the slot next.
        """
        now = time.monotonic()
        async with self._lock:
            stale = [tid for tid, start in self._start_times.items() if now - start > timeout_seconds]
            for tid in stale:
                self._topics.pop(tid, None)
                self._start_times.pop(tid, None)
        return stale


_checking_state = CheckingState()


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
