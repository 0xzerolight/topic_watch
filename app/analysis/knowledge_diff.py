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
import unicodedata
from dataclasses import dataclass

# 4th copy of this regex; see app/analysis/citations.py:137. Not shared — the
# migration's copy is a frozen snapshot and citations' drives the egress scrub.
#
# Two alternatives, because sentence terminators do not all behave alike
# (AUG-170): an ASCII terminator ends a sentence only when whitespace follows
# ("v1.2" must not split), while the CJK fullwidth terminators are unambiguous
# and are normally written with no space after them. The second branch therefore
# matches zero-width, which ``re.split`` has honoured since 3.7.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])\s*")

# A line ending in two or more spaces is a markdown hard break — it renders as
# <br>, so gaining or losing it changes the rendered state (AUG-254).
_HARD_BREAK = re.compile(r"[ \t]{2,}$")

# Above this many segments on either side, skip the matcher and show a snapshot.
# ``SequenceMatcher`` is O(n*m) and ``autojunk=False`` removes its safety valve:
# measured on repetitive input, 830 segments costs ~0.10 s, 1500 ~0.32 s, and
# 3000 ~1.28 s. The cap keeps an unauthenticated GET from buying seconds of CPU.
MAX_DIFF_SEGMENTS = 1500

# What a rendered revision body actually is. Only ``diff`` is a comparison of two
# adjacent revisions; the rest are bounded snapshots of a single one, and the
# added/removed counts and token delta mean nothing for them (AUG-222).
MODE_DIFF = "diff"
"""A real segment diff against the immediately preceding revision."""
MODE_SNAPSHOT = "snapshot"
"""No previous text was supplied. The router refines this into the reason —
``MODE_OLDEST``, ``MODE_REINITIALIZE`` or ``MODE_UNKNOWN_SOURCE``."""
MODE_OVERSIZE = "oversize"
"""Input beyond ``MAX_DIFF_SEGMENTS``: the matcher was skipped."""
MODE_OLDEST = "oldest"
"""The oldest revision still retained, so nothing precedes it."""
MODE_REINITIALIZE = "reinitialize"
"""An ``init`` revision that has retained history behind it — research was re-run
and this is the new baseline, not an edit of the old one."""
MODE_UNKNOWN_SOURCE = "unknown_source"
"""The stored ``source`` is not one this version knows, so its place in the
lineage is unknown and an adjacent diff would be a guess (AUG-155)."""

SNAPSHOT_MODES = frozenset({MODE_SNAPSHOT, MODE_OVERSIZE, MODE_OLDEST, MODE_REINITIALIZE, MODE_UNKNOWN_SOURCE})


@dataclass(frozen=True)
class DiffSegment:
    """One comparable unit of a knowledge summary and its diff verdict.

    ``kind`` is ``"equal"``, ``"insert"`` (present only in the newer revision),
    or ``"delete"`` (present only in the older one). A snapshot mode emits every
    segment as ``"equal"``: nothing was compared, so nothing was added.
    """

    kind: str
    text: str


@dataclass(frozen=True)
class Segment:
    """A summary fragment split into what the reader sees and what is compared.

    ``text`` keeps the markdown the model wrote, including the leading
    indentation that decides a bullet's nesting level. ``key`` is what the
    matcher compares: NFC-normalized (so canonically equivalent text is equal
    rather than a false rewrite) and carrying the structural facts that
    ``text`` alone would lose once stripped — indent width, hard break, and
    whether a blank line precedes it (AUG-170, AUG-254).
    """

    text: str
    key: str


@dataclass(frozen=True)
class DiffResult:
    """The rendered body of one revision, plus what kind of body it is.

    ``mode`` is load-bearing: a caller must not present ``inserted``/``deleted``
    counts or a token delta for anything in :data:`SNAPSHOT_MODES`, because no
    comparison happened.
    """

    mode: str
    segments: list[DiffSegment]


def _key(*, indent: str, body: str, hard_break: bool, after_blank: bool) -> str:
    """Build the comparison key for one segment.

    ``\\x00`` joins the parts so no body text can forge a different structure.
    Tabs expand to 4 columns so a tab-indented and a space-indented child of the
    same bullet compare equal.
    """
    return "\x00".join(
        (
            "blank" if after_blank else "",
            str(len(indent.expandtabs(4))),
            unicodedata.normalize("NFC", body),
            "br" if hard_break else "",
        )
    )


def split_segments(text: str) -> list[Segment]:
    """Split a knowledge summary into comparable segments.

    Splits on line boundaries first (preserving markdown structure — a heading
    and a bullet are never merged), then on sentence boundaries within each
    line.

    Blank lines produce no segment of their own but are not discarded: they mark
    the following segment as starting a new paragraph, so moving a sentence
    across a paragraph boundary is a change rather than a no-op. Indentation and
    hard-break markers are likewise kept out of the display text and folded into
    the comparison key.
    """
    segments: list[Segment] = []
    after_blank = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            after_blank = True
            continue
        indent = line[: len(line) - len(line.lstrip())]
        hard_break = bool(_HARD_BREAK.search(line.rstrip("\n")))
        parts = [part for part in _SENTENCE_SPLIT.split(stripped) if part]
        for index, part in enumerate(parts):
            first = index == 0
            last = index == len(parts) - 1
            segments.append(
                Segment(
                    text=(indent + part) if first else part,
                    key=_key(
                        indent=indent if first else "",
                        body=part,
                        hard_break=hard_break and last,
                        after_blank=after_blank and first,
                    ),
                )
            )
        after_blank = False
    return segments


def _snapshot(mode: str, segments: list[Segment]) -> DiffResult:
    """Render one revision's own segments, with no comparison implied."""
    return DiffResult(mode, [DiffSegment("equal", segment.text) for segment in segments])


def diff_segments(old: str, new: str) -> DiffResult:
    """Diff two knowledge summaries into an ordered list of segments.

    A ``replace`` opcode is emitted as its deletions followed by its insertions,
    so the output reads top-to-bottom in document order with the old wording
    immediately above the new.

    ``old=""`` (the oldest retained revision, a re-initialization, an
    unrecognised lineage) yields ``MODE_SNAPSHOT`` — every segment ``equal``,
    because there was nothing to compare against and calling the whole body an
    insertion invented a count (AUG-222). Input beyond ``MAX_DIFF_SEGMENTS``
    degrades to the same snapshot under ``MODE_OVERSIZE`` rather than running
    the quadratic matcher.

    ``autojunk=False`` is deliberate: the heuristic treats any segment appearing
    in more than 1% of a >200-item sequence as junk, which on a bullet list of
    similar short facts silently drops real matches. Callers in a request path
    must still run this via ``asyncio.to_thread``.
    """
    old_segments = split_segments(old)
    new_segments = split_segments(new)
    if max(len(old_segments), len(new_segments)) > MAX_DIFF_SEGMENTS:
        return _snapshot(MODE_OVERSIZE, new_segments)
    if not old_segments:
        return _snapshot(MODE_SNAPSHOT, new_segments)

    matcher = difflib.SequenceMatcher(
        None,
        [segment.key for segment in old_segments],
        [segment.key for segment in new_segments],
        autojunk=False,
    )
    out: list[DiffSegment] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            out.extend(DiffSegment("equal", segment.text) for segment in old_segments[i1:i2])
            continue
        out.extend(DiffSegment("delete", segment.text) for segment in old_segments[i1:i2])
        out.extend(DiffSegment("insert", segment.text) for segment in new_segments[j1:j2])
    return DiffResult(MODE_DIFF, out)
