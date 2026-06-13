"""Aggregate web router.

Combines the per-domain routers into a single ``router`` mounted by
``app.main``. Include order matters: routers with static topic paths
(``/topics/search``, ``/topics/new``) must register before the dynamic
``/topics/{topic_id}`` route so the static paths win.
"""

from fastapi import APIRouter

from app.web.routers import dashboard, feed_health, opml, settings, topics

router = APIRouter()

# dashboard first: registers static "/topics/search" ahead of "/topics/{topic_id}".
router.include_router(dashboard.router)
router.include_router(feed_health.router)
router.include_router(settings.router)
router.include_router(opml.router)
# topics last: "/topics/new" precedes "/topics/{topic_id}" within this router.
router.include_router(topics.router)

__all__ = ["router"]
