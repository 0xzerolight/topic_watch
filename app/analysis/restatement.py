"""key_facts restatement filtering (OVH-178).

Pure string algorithm extracted out of ``llm.py`` so the LLM I/O module is not
mixed with this phrase-matching logic. ``llm.py`` re-exports the public names for
back-compat.

A key_fact is dropped only when it is a CLEAR restatement of the existing
knowledge summary: a long *contiguous* word-sequence (n-gram) of the fact appears,
in order and adjacent, inside the summary. Matching is on WORD TOKENS, never on
raw characters, so a fact can only match at word boundaries — character
containment used to drop "He won." as a restatement of "She won." (AUG-165).
Phrase-level matching (not bag-of-words set overlap) is likewise required so a
short genuinely-new fact whose words merely scatter across a long summary is never
silently dropped. Conservative by design.
"""

import re
from collections import defaultdict
from math import ceil

# Restatement requires the longest fact word-sequence shared contiguously with
# the summary to cover at least this fraction of the fact's content words...
_RESTATEMENT_PHRASE_OVERLAP_THRESHOLD = 0.8
# ...and the fact must have at least this many content words. Shorter facts are
# never auto-dropped, since coincidental phrase matches are too easy on them.
_RESTATEMENT_MIN_FACT_WORDS = 4
_WORD_RE = re.compile(r"\w+")
# Hard ceiling on window comparisons per fact. Anchoring on the summary's word
# index makes real text cheap (a few hundred probes for a maximum-size summary),
# but text with extreme token repetition — every window anchored on a word that
# occurs thousands of times — would still cost enough to stall the shared event
# loop, which is the failure AUG-033 is about. On exhaustion the fact is KEPT:
# the filter's whole purpose is to drop only CLEAR restatements, so running out
# of budget must never be a reason to hide a fact.
_MAX_RUN_PROBES = 5_000


def _normalize_for_match(text: str) -> str:
    """Lowercase and collapse whitespace for comparison."""
    return " ".join(text.lower().split())


def _content_words(text: str) -> list[str]:
    """Extract ordered, lowercased word tokens (multiplicity preserved)."""
    return [m.group(0) for m in _WORD_RE.finditer(text.lower())]


def _word_positions(words: list[str]) -> dict[str, list[int]]:
    """Index word -> ascending positions, so a run can be probed by anchor word."""
    positions: dict[str, list[int]] = defaultdict(list)
    for index, word in enumerate(words):
        positions[word].append(index)
    return positions


def _required_run(fact_words: list[str]) -> int:
    """How many contiguous words a fact must share with the summary to be dropped."""
    return ceil(_RESTATEMENT_PHRASE_OVERLAP_THRESHOLD * len(fact_words))


def _has_shared_run(
    fact_words: list[str],
    summary_words: list[str],
    summary_positions: dict[str, list[int]],
    required: int,
) -> bool:
    """True if any ``required``-long window of the fact occurs inside the summary.

    Equivalent to "longest common contiguous run >= required", but anchored on the
    summary's word index instead of the O(fact x summary) dynamic-programming
    matrix the previous implementation ran per fact (AUG-033). Each window is only
    compared where its rarest word actually occurs, which on real text is a handful
    of places, so a verbose completion against a maximum-size summary no longer
    costs tens of millions of comparisons on the event loop.
    """
    if required <= 0 or required > len(fact_words) or required > len(summary_words):
        return False
    probes = 0
    for start in range(len(fact_words) - required + 1):
        window = fact_words[start : start + required]
        # Anchor on the window's RAREST word: a word absent from the summary rules
        # the window out for free, and a rare one leaves a couple of positions to
        # check instead of every occurrence of a common leading word.
        offset = min(range(required), key=lambda i: len(summary_positions.get(window[i], ())))
        for position in summary_positions.get(window[offset], ()):
            begin = position - offset
            if begin < 0:
                continue
            if begin + required > len(summary_words):
                break
            probes += 1
            if probes > _MAX_RUN_PROBES:
                return False
            if summary_words[begin : begin + required] == window:
                return True
    return False


def _is_restatement(fact: str, knowledge_summary: str) -> bool:
    """True if ``fact`` clearly restates content already in the summary.

    One conservative, phrase-level signal: the fact has at least
    ``_RESTATEMENT_MIN_FACT_WORDS`` content words AND its longest contiguous
    word-sequence shared with the summary covers at least
    ``_RESTATEMENT_PHRASE_OVERLAP_THRESHOLD`` of the fact's words. A fact repeated
    verbatim is simply the ratio-1.0 case of that rule.

    Set-overlap is deliberately NOT used: scattered, non-contiguous word matches
    must never drop a short, genuinely-new fact. Character containment is not used
    either — it matched across word boundaries, so a fact differing only in its
    subject ("He won." vs "She won.") counted as already known (AUG-165).

    An empty summary or empty fact is never a restatement.
    """
    if not knowledge_summary.strip() or not fact.strip():
        return False
    summary_words = _content_words(knowledge_summary)
    return _is_restatement_of(fact, summary_words, _word_positions(summary_words))


def _is_restatement_of(fact: str, summary_words: list[str], summary_positions: dict[str, list[int]]) -> bool:
    """``_is_restatement`` against an already-tokenized, already-indexed summary."""
    fact_words = _content_words(fact)
    if len(fact_words) < _RESTATEMENT_MIN_FACT_WORDS:
        return False
    return _has_shared_run(fact_words, summary_words, summary_positions, _required_run(fact_words))


def filter_restated_key_facts(key_facts: list[str], knowledge_summary: str) -> list[str]:
    """Drop key_facts that clearly restate the current knowledge summary.

    Kept conservative: only removes clear restatements. If every fact is filtered
    the caller keeps ``has_new_info`` as-is with an empty ``key_facts`` (the
    summary still conveys the novelty).

    The summary is tokenized and indexed ONCE for the whole batch rather than per
    fact (AUG-033).
    """
    if not knowledge_summary.strip():
        return list(key_facts)
    summary_words = _content_words(knowledge_summary)
    summary_positions = _word_positions(summary_words)
    return [fact for fact in key_facts if not _is_restatement_of(fact, summary_words, summary_positions)]
