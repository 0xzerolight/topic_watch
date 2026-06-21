"""Aggregate web router.

Combines the per-domain routers into a single ``router`` mounted by
``app.main``. Include order matters: routers with static topic paths
(``/topics/search``, ``/topics/new``) must register before the dynamic
``/topics/{topic_id}`` route so the static paths win.
"""

from fastapi import APIRouter

from app.web.routers import dashboard, exports, feed_health, opml, settings, topics

router = APIRouter()

# dashboard first: registers static "/topics/search" ahead of "/topics/{topic_id}".
router.include_router(dashboard.router)
router.include_router(feed_health.router)
router.include_router(settings.router)
router.include_router(opml.router)
# exports before topics: its "/topics/{topic_id}/export/*" paths are distinct from
# the dynamic "/topics/{topic_id}", but registering ahead keeps the static export
# paths unambiguous regardless of future route additions (OVH-155).
router.include_router(exports.router)
# topics last: "/topics/new" precedes "/topics/{topic_id}" within this router.
router.include_router(topics.router)

__all__ = ["router"]
