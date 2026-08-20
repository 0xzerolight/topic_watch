"""Resolve the file a sqlite3 connection is backed by.

The pipeline takes a ``db_path`` rather than a connection, so helper functions
that already receive the test's ``db_conn`` can derive the matching path without
threading a second fixture through every call.
"""

import sqlite3
from pathlib import Path


def conn_db_path(conn: sqlite3.Connection) -> Path:
    """Return the on-disk path backing ``conn``."""
    for _seq, name, file in conn.execute("PRAGMA database_list").fetchall():
        if name == "main" and file:
            return Path(file)
    raise AssertionError("connection is not backed by a file")
