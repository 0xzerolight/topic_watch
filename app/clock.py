"""The process's monotonic clock, as one seam.

Liveness — how long an owner has held a slot, how much of a cooldown is left,
how old an entry in a rate-limit window is — is measured monotonically, never by
differencing two wall-clock readings. A host clock step (an NTP correction, a
resumed laptop, an operator setting the date) must not expire a cooldown early
or extend it for hours, and it must not make a rate-limit window forget or keep
requests it should not (wave-A clock policy). Durable due-times stay wall clock,
spelled by ``models.to_db_utc``, because they outlive the process.

Import the module, not the function — ``from app import clock`` then
``clock.monotonic_now()`` — so a test controls every caller by patching
``app.clock.monotonic_now`` once, instead of one patch per importing module.
"""

import time


def monotonic_now() -> float:
    """Seconds since an arbitrary fixed point, guaranteed never to go backwards."""
    return time.monotonic()
