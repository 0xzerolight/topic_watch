"""key_facts restatement filtering (OVH-178).

Pure string algorithm extracted out of ``llm.py`` so the LLM I/O module is not
mixed with this ~90-line phrase-matching logic. ``llm.py`` re-exports the public
names for back-compat.

A key_fact is dropped only when it is a CLEAR restatement of the existing
knowledge summary: either its normalized text appears verbatim inside the
normalized summary, or a long *contiguous* word-sequence (n-gram) of the fact
appears verbatim in the summary. Phrase-level matching (not bag-of-words set
overlap) is required so a short genuinely-new fact whose words merely scatter
across a long summary is never silently dropped. Conservative by design.
"""

import re

# Restatement requires the longest fact word-sequence shared contiguously with
# the summary to cover at least this fraction of the fact's content words...
_RESTATEMENT_PHRASE_OVERLAP_THRESHOLD = 0.8
# ...and the fact must have at least this many content words. Shorter facts are
# never auto-dropped, since coincidental phrase matches are too easy on them.
_RESTATEMENT_MIN_FACT_WORDS = 4
_WORD_RE = re.compile(r"\w+")


def _normalize_for_match(text: str) -> str:
    """Lowercase and collapse whitespace for substring comparison."""
    return " ".join(text.lower().split())


def _content_words(text: str) -> list[str]:
    """Extract ordered, lowercased word tokens (multiplicity preserved)."""
    return [m.group(0) for m in _WORD_RE.finditer(text.lower())]


def _longest_contiguous_run(fact_words: list[str], summary_words: list[str]) -> int:
    """Length of the longest contiguous run of ``fact_words`` that appears, in
    order and adjacent, anywhere within ``summary_words``.

    Classic DP for longest common substring over token sequences.
    """
    if not fact_words or not summary_words:
        return 0
    prev = [0] * (len(summary_words) + 1)
    best = 0
    for fw in fact_words:
        curr = [0] * (len(summary_words) + 1)
        for j, sw in enumerate(summary_words, start=1):
            if fw == sw:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best:
                    best = curr[j]
        prev = curr
    return best


def _is_restatement(fact: str, knowledge_summary: str) -> bool:
    """True if ``fact`` clearly restates content already in the summary.

    Two conservative, phrase-level signals:
    * normalized-substring: the fact's normalized text appears verbatim within
      the normalized knowledge summary; or
    * contiguous-phrase: the fact has ≥ ``_RESTATEMENT_MIN_FACT_WORDS`` content
      words AND its longest contiguous word-sequence shared with the summary
      covers ≥ ``_RESTATEMENT_PHRASE_OVERLAP_THRESHOLD`` of the fact's words.

    Set-overlap is deliberately NOT used: scattered, non-contiguous word matches
    must never drop a short, genuinely-new fact.

    An empty summary or empty fact is never a restatement.
    """
    if not knowledge_summary.strip() or not fact.strip():
        return False

    norm_fact = _normalize_for_match(fact)
    norm_summary = _normalize_for_match(knowledge_summary)
    if norm_fact and norm_fact in norm_summary:
        return True

    fact_words = _content_words(fact)
    if len(fact_words) < _RESTATEMENT_MIN_FACT_WORDS:
        return False
    summary_words = _content_words(knowledge_summary)
    run = _longest_contiguous_run(fact_words, summary_words)
    return run / len(fact_words) >= _RESTATEMENT_PHRASE_OVERLAP_THRESHOLD


def filter_restated_key_facts(key_facts: list[str], knowledge_summary: str) -> list[str]:
    """Drop key_facts that clearly restate the current knowledge summary.

    Kept conservative: only removes clear restatements. If every fact is filtered
    the caller keeps ``has_new_info`` as-is with an empty ``key_facts`` (the
    summary still conveys the novelty).
    """
    return [fact for fact in key_facts if not _is_restatement(fact, knowledge_summary)]
