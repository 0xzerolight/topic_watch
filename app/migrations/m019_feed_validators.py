"""Add conditional-GET validators (ETag / Last-Modified) to feed_health.

Stores the opaque HTTP validator strings echoed back on the next fetch as
If-None-Match / If-Modified-Since, so an unchanged feed can answer 304 with no
body re-parse. NULL until a feed first returns a validator. Never parsed as a
datetime — Last-Modified is an opaque HTTP-date string sent back verbatim.
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE feed_health ADD COLUMN etag TEXT")
    conn.execute("ALTER TABLE feed_health ADD COLUMN last_modified TEXT")
