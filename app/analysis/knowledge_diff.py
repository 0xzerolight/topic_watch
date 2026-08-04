"""Segment-level diffing of knowledge-state revisions.

Knowledge summaries are LLM-written markdown: labelled paragraphs and bullet
lists of short factual sentences. A raw line diff reports a whole reflowed
paragraph as changed; a character diff produces unreadable confetti. Splitting
on line *and* sentence boundaries and diffing those segments matches how the
model actually edits the state — it adds, drops, or rewrites individual facts.

Pure and dependency-free: no DB, web, or LLM imports. Sited under
``app/analysis/`` alongside ``restatement.py``, the existing precedent for a
pure string algorithm kept out of ``llm.py``.
"""

import difflib
import re
from dataclasses import dataclass

# 4th copy of this regex; see app/analysis/citations.py:137. Not shared — the
# migration's copy is a frozen snapshot and citations' drives the egress scrub.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Above this many segments on either side, skip the matcher and show a snapshot.
# ``SequenceMatcher`` is O(n*m) and ``autojunk=False`` removes its safety valve:
# measured on repetitive input, 830 segments costs ~0.10 s, 1500 ~0.32 s, and
# 3000 ~1.28 s. The cap keeps an unauthenticated GET from buying seconds of CPU.
MAX_DIFF_SEGMENTS = 1500


@dataclass(frozen=True)
class DiffSegment:
    """One comparable unit of a knowledge summary and its diff verdict.

    ``kind`` is ``"equal"``, ``"insert"`` (present only in the newer revision),
    or ``"delete"`` (present only in the older one).
    """

    kind: str
    text: str


def split_segments(text: str) -> list[str]:
    """Split a knowledge summary into comparable segments.

    Splits on line boundaries first (preserving markdown structure — a heading
    and a bullet are never merged), then on sentence boundaries within each
    line. Blank lines are dropped and every segment is stripped.
    """
    segments: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        segments.extend(part for part in _SENTENCE_SPLIT.split(stripped) if part)
    return segments


def diff_segments(old: str, new: str) -> list[DiffSegment]:
    """Diff two knowledge summaries into an ordered list of segments.

    A ``replace`` opcode is emitted as its deletions followed by its insertions,
    so the output reads top-to-bottom in document order with the old wording
    immediately above the new.

    ``old=""`` (the oldest retained revision) yields an all-``insert`` list — the
    full snapshot. Input beyond ``MAX_DIFF_SEGMENTS`` degrades to the same
    all-insert snapshot rather than running the quadratic matcher.

    ``autojunk=False`` is deliberate: the heuristic treats any segment appearing
    in more than 1% of a >200-item sequence as junk, which on a bullet list of
    similar short facts silently drops real matches. Callers in a request path
    must still run this via ``asyncio.to_thread``.
    """
    old_segments = split_segments(old)
    new_segments = split_segments(new)
    if max(len(old_segments), len(new_segments)) > MAX_DIFF_SEGMENTS:
        return [DiffSegment("insert", text) for text in new_segments]

    matcher = difflib.SequenceMatcher(None, old_segments, new_segments, autojunk=False)
    out: list[DiffSegment] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            out.extend(DiffSegment("equal", text) for text in old_segments[i1:i2])
            continue
        out.extend(DiffSegment("delete", text) for text in old_segments[i1:i2])
        out.extend(DiffSegment("insert", text) for text in new_segments[j1:j2])
    return out
