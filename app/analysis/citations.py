"""Strip ephemeral article-index citations from LLM output.

The novelty/knowledge prompts number input articles ``[1]``, ``[2]`` … per call
(see :func:`app.analysis.prompts._format_articles`). The model echoes those
indices back into its summaries as ``(Article [1])`` / ``(Articles [3], [5])``.
The numbering is per run, so a persisted ``Article [4]`` points at a different
article on the next check — incoherent noise. Real provenance already lives in
``NoveltyResult.source_urls`` and in the named attribution the model writes
("per D&C Media's report").

Scope: citations are removed ONLY where they sit inside parentheses as
attribution glue — the form they take in every fact field. A citation that is the
grammatical subject of a sentence ("Article [7] restates …", only ever in the
free-prose ``reasoning`` field) is left untouched: deleting it would yield
subjectless grammar.

The match is anchored on a bracketed DIGIT preceded by the word "article", so it
never touches ``[STUB]``/``[NO CONTENT]`` tags, named ``[source]`` qualifiers, or
markdown ``[text](url)`` links.

``strip_reliability_notes`` is the companion scrub for the OTHER leak: internal
``[STUB]``/``[NO CONTENT]`` reliability bookkeeping the model narrates back into
the summary ("Note on Data Quality: Articles [1], [2] are marked [STUB] with
minimal content."). ``strip_index_citations`` deliberately leaves that
subject-position prose alone, so it used to survive, get persisted into the stored
knowledge state, and be re-fed into every later prompt. The two run in sequence at
the LLM egress boundary (index-citations first, then reliability notes).
"""

import re

# A citation token-run: an optional attribution connector, the word "article(s)",
# and one or more bracketed integer indices joined by ",", "and", "through", "to".
# The index list is an ATOMIC group: without it the engine backtracks the run to a
# single index to satisfy a following separator, orphaning the rest of the list
# ("(Articles [3], [5])" -> "([5])").
_CONNECTOR = r"(?:per|see|according to|reported in|reported by)\s+"
_INDEX = r"\[\s*\d+\s*\]"
_CITE = rf"(?:{_CONNECTOR})?articles?\s*{_INDEX}(?>(?:\s*(?:,|and|through|to)\s*{_INDEX})*)"

# Trailing position — a separator precedes the cite ("June 10; Article [1]" -> "June 10").
_CITE_TRAILING = re.compile(rf"\s*[;,]\s*{_CITE}", re.IGNORECASE)
# Leading position — a separator follows the cite
# ("Article [7], reported by Bloomberg" -> "reported by Bloomberg").
_CITE_LEADING = re.compile(rf"{_CITE}\s*[;,]\s*", re.IGNORECASE)
# No adjacent separator — whole-paren, or space-joined ("… per Article [10]").
_CITE_BARE = re.compile(rf"\s*{_CITE}", re.IGNORECASE)

# Flat (non-nested) parenthetical group; citations never nest parens in practice.
_PAREN = re.compile(r"\(([^()]*)\)")


def _clean_paren_inner(inner: str) -> str:
    """Remove every citation run from one parenthetical's inner text."""
    out = _CITE_TRAILING.sub("", inner)
    out = _CITE_LEADING.sub("", out)
    out = _CITE_BARE.sub("", out)
    return out


def strip_index_citations(text: str) -> str:
    """Remove parenthetical article-index citations, leaving prose cites intact.

    A parenthetical reduced to nothing (or only stray separators/space) is dropped
    whole; one that still holds named attribution keeps it. Out-of-paren citations
    are returned untouched.
    """
    if not text or "[" not in text:
        return text

    def _repl(match: re.Match[str]) -> str:
        cleaned = _clean_paren_inner(match.group(1))
        if not cleaned.strip(" ,;"):
            return ""
        return f"({cleaned.strip()})"

    result = _PAREN.sub(_repl, text)
    # Normalize residue without collapsing newlines: squeeze horizontal-space runs,
    # drop a space before punctuation, tidy a space orphaned after a cut paren, and
    # trim horizontal whitespace left at a line end where a trailing cite was cut.
    result = re.sub(r"[ \t]{2,}", " ", result)
    result = re.sub(r" +([,.;:)])", r"\1", result)
    result = re.sub(r"\( +", "(", result)
    result = re.sub(r"[ \t]+(\n|$)", r"\1", result)
    return result


# --- Reliability-note scrubbing (path C: keep the [STUB] heuristic, kill its leak) ---
#
# The prompts tag thin articles ``[STUB — …]`` / ``[NO CONTENT]`` so the LLM
# down-weights them (OVH-158). Despite ``_RULE_NO_INDEX_CITATIONS``, the model
# still narrates that internal bookkeeping into the user-facing summary. The three
# leak shapes seen in the DB:
#   1. a labelled aside block ("Note on Data Quality: Articles [1], [2] are …")
#   2. a standalone note sentence ("Article [3] provides the most substantive …")
#   3. a reliability aside embedded in a real fact
#      ("Fable 5 released June 10 (source articles marked [STUB]; incomplete)")
# Shapes 1-2 are dropped whole; shape 3 keeps the fact and strips only the aside.

# Bracketed internal tag ("[STUB — minimal content, low reliability]", "[NO CONTENT]").
_RELIABILITY_TAG = re.compile(r"\[\s*(?:STUB|NO\s+CONTENT)\b[^\]]*\]", re.IGNORECASE)
# A bracketed integer index; after strip_index_citations the survivors are all
# subject-position, i.e. sitting inside a leaked note.
_INDEX_TOKEN = re.compile(r"\[\s*\d+\s*\]")

# "Strong" signals: internal jargon that ~never appears in real monitored-news
# prose, so their presence alone marks a leaked note.
_STRONG_SIGNAL = re.compile(
    r"\[\s*(?:STUB|NO\s+CONTENT)\b[^\]]*\]|\bstubs?\b|drawn\s+(?:primarily\s+)?from\s+stub",
    re.IGNORECASE,
)
# "Weak" signals: plausible in real prose ("the deal is incomplete"), so they only
# mark a note when paired with an index token, a [STUB] tag, or a "Note" label.
_WEAK_SIGNAL = re.compile(
    r"\b(?:marked|minimal content|most substantive|low reliability|incomplete|no content)\b",
    re.IGNORECASE,
)
_NOTE_LABEL = re.compile(r"^\s*note\b[^:]{0,40}:", re.IGNORECASE)
# A sentence whose subject is an article-index enumeration ("Articles [1], [2] …").
_ARTICLE_SUBJECT = re.compile(r"^\s*articles?\s*\[\s*\d+\s*\]", re.IGNORECASE)

# A whole quality-note block heading ("Note on Data Quality:", bold or plain).
_QUALITY_HEADING = re.compile(
    r"^\s*\**\s*(?:note\s+on\s+data\s+quality|data\s+quality|"
    r"note\s+on\s+source\s+quality|note\s+on\s+data\s+sources)\b",
    re.IGNORECASE,
)
# Start of a new structured section: bold **Heading:**, markdown #, or a short
# "Label:" line. Only used to bound a dropped block (terminating early keeps
# content — the safe direction).
_SECTION_START = re.compile(r"^\s*(?:\*\*.+?\*\*|#{1,6}\s|[A-Z][A-Za-z0-9 /()&'-]{1,48}:)")
# A heading line we must never sentence-split (bold or markdown only; plain
# "Label:" lines are safe to run through the sentence pass — they never match a
# note signature).
_HEADING = re.compile(r"^\s*(?:\*\*.+?\*\*|#{1,6}\s)")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_BULLET_PREFIX = re.compile(r"^(\s*(?:[-*]\s+)?)(.*)$", re.DOTALL)


def _is_reliability_paren(inner: str) -> bool:
    """True when a parenthetical's inner text is a reliability aside."""
    if _STRONG_SIGNAL.search(inner):
        return True
    return bool(_WEAK_SIGNAL.search(inner) and (_INDEX_TOKEN.search(inner) or _RELIABILITY_TAG.search(inner)))


def _is_note_sentence(sentence: str) -> bool:
    """True when a whole sentence is standalone reliability narration."""
    if _NOTE_LABEL.match(sentence):
        return True
    if _ARTICLE_SUBJECT.match(sentence) and (_STRONG_SIGNAL.search(sentence) or _WEAK_SIGNAL.search(sentence)):
        return True
    if re.search(r"drawn\s+(?:primarily\s+)?from\s+stub", sentence, re.IGNORECASE):
        return True
    return bool(re.search(r"\bstubs?\b", sentence, re.IGNORECASE) and _WEAK_SIGNAL.search(sentence))


def _scrub_line(line: str) -> str | None:
    """Scrub one line; return None to drop it entirely."""
    if not line.strip() or _HEADING.match(line):
        return line
    # Shape 3: strip reliability parentheticals, keep the host sentence.
    body = _PAREN.sub(lambda m: "" if _is_reliability_paren(m.group(1)) else m.group(0), line)
    # Shapes 1-2: drop standalone note sentences, preserving any bullet marker.
    prefix, content = _BULLET_PREFIX.match(body).groups()  # type: ignore[union-attr]
    kept = [s for s in _SENTENCE_SPLIT.split(content) if s.strip() and not _is_note_sentence(s)]
    new_content = " ".join(kept)
    # Token mop-up for a stray tag left inside an otherwise-kept sentence.
    new_content = _RELIABILITY_TAG.sub("", new_content)
    new_content = re.sub(r"\bmarked\b\s*(?=$|[.,;:)])", "", new_content, flags=re.IGNORECASE)
    new_content = new_content.strip()
    return prefix + new_content if new_content else None


def _drop_empty_sections(text: str) -> str:
    """Drop a section heading left with no body by the scrub (e.g. an emptied
    Reported/Claimed block)."""
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


def strip_reliability_notes(text: str) -> str:
    """Remove leaked ``[STUB]``/``[NO CONTENT]`` reliability bookkeeping.

    Run AFTER :func:`strip_index_citations` so its parenthetical index cites are
    already gone. Whole quality-note blocks and standalone note sentences are
    dropped; reliability asides inside a real fact are stripped in place.
    """
    if not text:
        return text
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if _QUALITY_HEADING.match(lines[i]):
            # Drop the note heading and its body until a blank line, the next
            # section, or EOF.
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
    # Normalize residue (mirror strip_index_citations) and collapse gaps left by
    # dropped blocks, without collapsing intentional single newlines.
    result = re.sub(r"[ \t]{2,}", " ", result)
    result = re.sub(r" +([,.;:)])", r"\1", result)
    result = re.sub(r"\( +", "(", result)
    result = re.sub(r"[ \t]+(\n|$)", r"\1", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip("\n")
