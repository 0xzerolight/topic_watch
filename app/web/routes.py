"""Backwards-compatible re-export of the aggregate web router.

The route handlers were split into the ``app.web.routers`` package. This
module is kept so existing imports of ``from app.web.routes import router``
continue to work.
"""

from app.web.routers import router

__all__ = ["router"]
