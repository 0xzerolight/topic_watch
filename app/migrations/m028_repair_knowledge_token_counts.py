"""Migration 028: repair impossible ``knowledge_states.token_count`` values.

``token_count`` is denormalized: every live path recomputes it with the
model-aware counter right after it changes ``summary_text``. Migration 021 broke
that pairing — it rewrote summaries to scrub leaked ``[STUB]`` reliability notes
and left the counts alone, so a topic that has not had a knowledge update since
still renders its old, larger number against the token budget (TW-AUD-014). m021
is append-only history and is not edited; this is the repair pass.

What can be repaired without a tokenizer is what is *provably* wrong, and only
that. A token is at least one character, so ``token_count > length(summary_text)``
cannot be true of the stored text under any tokenizer, and an empty summary is
zero tokens under all of them. Those two rows are corrected here.

A count that is merely proportionally stale — the note scrubbed out was 5% of a
still-substantial summary — is not detectable after the fact: nothing records the
pre-scrub length, and re-deriving one would need the configured model, which
migrations do not have. Those rows keep a slightly high count until the next
knowledge write, which recounts from scratch. New revisions record the model that
counted them (migration 029) so history at least stops comparing counts across
different units.

``knowledge_revisions`` is deliberately untouched: it is append-only history, and
its m025-backfilled rows are marked unknown-provenance by their NULL ``model``
rather than rewritten.
"""

import sqlite3

# SQLite's one-argument ``trim`` strips spaces and nothing else, so a summary m021
# reduced to a bare newline would not read as empty. Name the whitespace
# explicitly.
_BLANK = "trim(summary_text, char(32) || char(9) || char(10) || char(13)) = ''"


def up(conn: sqlite3.Connection) -> None:
    # Empty text is zero tokens under every tokenizer.
    conn.execute(
        f"UPDATE knowledge_states SET token_count = 0 WHERE token_count != 0 AND (summary_text IS NULL OR {_BLANK})"  # noqa: S608 - no interpolated values
    )
    # One token cannot be shorter than one character, so anything above the
    # character count is impossible. Fall back to the same char/4 estimate
    # ``count_tokens`` uses when a tokenizer is unavailable, floored at 1 so a
    # non-empty summary never reports zero.
    conn.execute(
        f"UPDATE knowledge_states SET token_count = max(length(summary_text) / 4, 1) "  # noqa: S608 - no values
        f"WHERE summary_text IS NOT NULL AND NOT ({_BLANK}) "
        f"AND token_count > length(summary_text)"
    )
