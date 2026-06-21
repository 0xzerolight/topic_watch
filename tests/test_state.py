"""Tests for process-global web state (OVH-126 dashboard stats cache guard)."""

import asyncio

from app.web.state import DashboardStatsCache


class TestDashboardStatsCache:
    async def test_populates_then_caches(self) -> None:
        cache = DashboardStatsCache(ttl=60.0)
        calls = 0

        def loader() -> str:
            nonlocal calls
            calls += 1
            return f"stats-{calls}"

        first = await cache.get_or_populate(loader)
        second = await cache.get_or_populate(loader)
        assert first == "stats-1"
        assert second == "stats-1"  # cached, not recomputed
        assert calls == 1

    async def test_recomputes_after_expiry(self) -> None:
        cache = DashboardStatsCache(ttl=0.0)  # always stale
        calls = 0

        def loader() -> str:
            nonlocal calls
            calls += 1
            return f"stats-{calls}"

        await cache.get_or_populate(loader)
        # ttl=0 means already-expired; a re-read recomputes.
        await asyncio.sleep(0.001)
        await cache.get_or_populate(loader)
        assert calls == 2

    async def test_concurrent_populate_runs_loader_once(self) -> None:
        """The check-then-set is atomic under the lock, even if the loader
        awaits — only one coroutine recomputes a stale/empty cache."""
        cache = DashboardStatsCache(ttl=60.0)
        calls = 0
        gate = asyncio.Event()

        async def slow_loader() -> str:
            nonlocal calls
            calls += 1
            await gate.wait()  # await inside the populate path
            return "stats"

        async def populate() -> str:
            return await cache.get_or_populate(slow_loader)

        task_a = asyncio.create_task(populate())
        task_b = asyncio.create_task(populate())
        await asyncio.sleep(0.01)  # let both reach the lock
        gate.set()
        results = await asyncio.gather(task_a, task_b)

        assert calls == 1  # loader invoked exactly once despite the await
        assert results == ["stats", "stats"]

    async def test_supports_sync_loader(self) -> None:
        cache = DashboardStatsCache(ttl=60.0)
        result = await cache.get_or_populate(lambda: "value")
        assert result == "value"

    async def test_reset_clears_cache(self) -> None:
        cache = DashboardStatsCache(ttl=60.0)
        calls = 0

        def loader() -> str:
            nonlocal calls
            calls += 1
            return "x"

        await cache.get_or_populate(loader)
        cache.reset()
        await cache.get_or_populate(loader)
        assert calls == 2
