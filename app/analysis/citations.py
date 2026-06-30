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
free-prose ``reasoning`` field, and a few ``Note: Article [N] is marked [STUB]``
asides) is left untouched: deleting it would yield subjectless grammar. Those are
suppressed at the source by the ``_RULE_NO_INDEX_CITATIONS`` prompt rule instead.

The match is anchored on a bracketed DIGIT preceded by the word "article", so it
never touches ``[STUB]``/``[NO CONTENT]`` tags, named ``[source]`` qualifiers, or
markdown ``[text](url)`` links.
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
