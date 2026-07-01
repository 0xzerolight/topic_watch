"""Migration 021: Scrub leaked [STUB]/[NO CONTENT] reliability notes from
persisted knowledge summaries.

Older summaries in ``knowledge_states.summary_text`` contain internal reliability
bookkeeping the LLM narrated back into the state ("Note on Data Quality: Articles
[1], [2] are marked [STUB] with minimal content."). Egress scrubbing now removes
this going forward (``app.analysis.citations.strip_reliability_notes``), but rows
written before that change are still polluted and keep getting re-fed into every
future prompt. This one-time pass cleans them.

The scrub logic is INLINED (a frozen copy of ``strip_reliability_notes`` at the
time this migration was written) rather than imported: migrations are append-only
and must stay reproducible even if the live function later changes. The scrub is
idempotent, so re-running it is a no-op.
"""

import re
import sqlite3

_RELIABILITY_TAG = re.compile(r"\[\s*(?:STUB|NO\s+CONTENT)\b[^\]]*\]", re.IGNORECASE)
_INDEX_TOKEN = re.compile(r"\[\s*\d+\s*\]")
_STRONG_SIGNAL = re.compile(
    r"\[\s*(?:STUB|NO\s+CONTENT)\b[^\]]*\]|\bstubs?\b|drawn\s+(?:primarily\s+)?from\s+stub",
    re.IGNORECASE,
)
_WEAK_SIGNAL = re.compile(
    r"\b(?:marked|minimal content|most substantive|low reliability|incomplete|no content)\b",
    re.IGNORECASE,
)
_NOTE_LABEL = re.compile(r"^\s*note\b[^:]{0,40}:", re.IGNORECASE)
_ARTICLE_SUBJECT = re.compile(r"^\s*articles?\s*\[\s*\d+\s*\]", re.IGNORECASE)
_QUALITY_HEADING = re.compile(
    r"^\s*\**\s*(?:note\s+on\s+data\s+quality|data\s+quality|"
    r"note\s+on\s+source\s+quality|note\s+on\s+data\s+sources)\b",
    re.IGNORECASE,
)
_SECTION_START = re.compile(r"^\s*(?:\*\*.+?\*\*|#{1,6}\s|[A-Z][A-Za-z0-9 /()&'-]{1,48}:)")
_HEADING = re.compile(r"^\s*(?:\*\*.+?\*\*|#{1,6}\s)")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_BULLET_PREFIX = re.compile(r"^(\s*(?:[-*]\s+)?)(.*)$", re.DOTALL)
_PAREN = re.compile(r"\(([^()]*)\)")


def _is_reliability_paren(inner: str) -> bool:
    if _STRONG_SIGNAL.search(inner):
        return True
    return bool(_WEAK_SIGNAL.search(inner) and (_INDEX_TOKEN.search(inner) or _RELIABILITY_TAG.search(inner)))


def _is_note_sentence(sentence: str) -> bool:
    if _NOTE_LABEL.match(sentence):
        return True
    if _ARTICLE_SUBJECT.match(sentence) and (_STRONG_SIGNAL.search(sentence) or _WEAK_SIGNAL.search(sentence)):
        return True
    if re.search(r"drawn\s+(?:primarily\s+)?from\s+stub", sentence, re.IGNORECASE):
        return True
    return bool(re.search(r"\bstubs?\b", sentence, re.IGNORECASE) and _WEAK_SIGNAL.search(sentence))


def _scrub_line(line: str) -> str | None:
    if not line.strip() or _HEADING.match(line):
        return line
    body = _PAREN.sub(lambda m: "" if _is_reliability_paren(m.group(1)) else m.group(0), line)
    prefix, content = _BULLET_PREFIX.match(body).groups()  # type: ignore[union-attr]
    kept = [s for s in _SENTENCE_SPLIT.split(content) if s.strip() and not _is_note_sentence(s)]
    new_content = " ".join(kept)
    new_content = _RELIABILITY_TAG.sub("", new_content)
    new_content = re.sub(r"\bmarked\b\s*(?=$|[.,;:)])", "", new_content, flags=re.IGNORECASE)
    new_content = new_content.strip()
    return prefix + new_content if new_content else None


def _drop_empty_sections(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _SECTION_START.match(line) and line.strip():
            has_inline = bool(_SECTION_START.sub("", line, count=1).strip())
            j = i + 1
            has_body = False
            while j < len(lines) and not (_SECTION_START.match(lines[j]) and lines[j].strip()):
                if lines[j].strip():
                    has_body = True
                j += 1
            if not has_inline and not has_body:
                i += 1
                continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _scrub(text: str) -> str:
    if not text:
        return text
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if _QUALITY_HEADING.match(lines[i]):
            i += 1
            while i < len(lines) and lines[i].strip() and not _SECTION_START.match(lines[i]):
                i += 1
            continue
        scrubbed = _scrub_line(lines[i])
        if scrubbed is not None:
            out.append(scrubbed)
        i += 1
    result = "\n".join(out)
    result = _drop_empty_sections(result)
    result = re.sub(r"[ \t]{2,}", " ", result)
    result = re.sub(r" +([,.;:)])", r"\1", result)
    result = re.sub(r"\( +", "(", result)
    result = re.sub(r"[ \t]+(\n|$)", r"\1", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip("\n")


def up(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT topic_id, summary_text FROM knowledge_states").fetchall()
    for topic_id, summary_text in rows:
        if not summary_text:
            continue
        scrubbed = _scrub(summary_text)
        if scrubbed != summary_text:
            conn.execute(
                "UPDATE knowledge_states SET summary_text = :summary WHERE topic_id = :topic_id",
                {"summary": scrubbed, "topic_id": topic_id},
            )
