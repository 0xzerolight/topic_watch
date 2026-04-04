"""Prompt templates for LLM-based novelty detection and knowledge management.

Each builder function returns a list of chat messages (system + user)
ready to be passed to the LLM via instructor/litellm.
"""

from app.models import Article, Topic

_PROMPT_ARTICLE_MAX_CHARS = 1500

# --- Novelty detection ---

_NOVELTY_SYSTEM = """\
You are a novelty detector for a news monitoring system. Your job is to compare \
new articles against an existing knowledge state and determine if the articles \
contain genuinely new information.

=== CRITICAL RULES ===
1. Base your analysis ONLY on what is written in the provided articles and \
knowledge state. Do NOT use your training data to fill gaps or resolve ambiguity.
2. Be CONSERVATIVE. The user only wants to be notified about meaningful updates — \
not reworded versions of known facts, not speculation that repeats existing rumors, \
not "roundup" articles that summarize old news.
3. SCOPE to the topic description. The description defines what the user cares about. \
Only flag information as new if it directly relates to those specific aspects. Ignore \
facts about the broader topic that fall outside the described scope.

=== MARK has_new_info=true ONLY WHEN ===
- Concrete new facts (specific dates, names, numbers, decisions) not in the \
knowledge state, clearly stated in an article (not inferred)
- Official announcements or confirmations of previously unconfirmed information
- Meaningful status changes (e.g., "delayed" to "released", "rumored" to "confirmed")
- Corrections to facts in the knowledge state (with evidence from articles)

=== MARK has_new_info=false WHEN ===
- Articles restate information already captured in the knowledge state
- The "new" information is just rewording, rephrasing, or editorial spin
- Articles contain speculation or opinion without new supporting evidence
- The article covers a related but different subject
- Articles are thin stubs with only headlines and no substantive content
- Information is approximate, estimated, or hedged ("might", "could", "expected to", \
"analysts predict", "sources suggest") without a concrete verifiable fact
- Rumors or unverified claims from unnamed sources

=== EDGE CASES ===
- If the knowledge state is empty or says "No existing knowledge state": treat \
all well-sourced factual content as new (but still ignore stubs and speculation).
- If articles contradict the knowledge state: this IS new information — flag it \
with the specific contradiction.
- If you are unsure whether something is genuinely new: set has_new_info=false. \
False negatives (missing an update) are much less harmful than false positives \
(spamming the user with non-updates).

=== OUTPUT ===
In your reasoning field, briefly explain what you compared and why you reached \
your conclusion. If has_new_info is true, list ONLY the specific new facts in \
key_facts (not restatements of known info) and the source article URLs in source_urls. \
Set confidence using this scale:
- 0.9-1.0: Official/primary source, unambiguous new fact, directly answers the topic description
- 0.7-0.8: Credible secondary source, clear new fact, directly relevant to description
- 0.5-0.6: Single source or partially ambiguous, but concrete and relevant
- 0.3-0.4: Weak sourcing, tangentially relevant, or not clearly new
- 0.1-0.2: Speculative, rumored, or only marginally related to description
Do NOT default to 0.7-0.8. Calibrate deliberately using the criteria above.
Set relevance to indicate how directly the new information addresses the topic \
description (0.0 = tangentially related, 1.0 = exactly what the user asked about)."""

_NOVELTY_USER = """\
Topic: {topic_name}
Description: {topic_description}

Current Knowledge State:
{knowledge_summary}

New Articles:
{articles}"""

# --- Knowledge initialization ---

_KNOWLEDGE_INIT_SYSTEM = """\
You are an information extraction system. Your job is to read the provided \
articles about a topic and extract a structured summary of facts \
**relevant to the topic description**. The description tells you what the user \
wants to monitor. Focus extraction on those aspects. Omit tangential facts even \
if they appear in the articles.

=== CRITICAL RULES ===
1. Use ONLY information that is explicitly stated in the provided articles. \
Do NOT add facts, dates, names, numbers, or context from your own training data.
2. If the articles do not contain enough relevant information about the topic, \
set sufficient_data to false and explain what is missing.
3. Every fact in your summary must be traceable to at least one provided article. \
If only one article mentions a fact, note it as "single-source."
4. Clearly distinguish between confirmed facts and claims/rumors reported in \
the articles. Use qualifiers: "according to [source]", "reportedly", "rumored."
5. If articles contain contradictory information, include BOTH versions and note \
the contradiction.
6. Do NOT speculate, infer, or fill gaps. If the articles don't mention something, \
leave it out entirely.
7. Articles marked [STUB] or [NO CONTENT] have unreliable or missing text — weigh \
them lower and rely primarily on their titles.

=== OUTPUT FORMAT ===
Write a structured summary using only categories that have supporting evidence. \
Do NOT include empty categories. Possible categories (use only as needed):
- **Confirmed Facts:** Specific, sourced facts from the articles
- **Reported/Claimed:** Information attributed to specific sources but not \
independently confirmed
- **Contradictions:** Where articles disagree (include both versions)
- **Timeline:** Only dates/events explicitly mentioned in the articles

Keep the summary under {max_tokens} tokens. Be concise — fact density over prose."""

_KNOWLEDGE_INIT_USER = """\
Topic: {topic_name}
Description: {topic_description}

Articles to analyze:
{articles}"""

# --- Knowledge update ---

_KNOWLEDGE_UPDATE_SYSTEM = """\
You are updating an existing knowledge state by incorporating newly verified \
information. Your job is to merge the new findings into the existing summary.

=== CRITICAL RULES ===
1. Only add the specific new facts listed in "New Findings." Do NOT introduce \
additional information from your training data.
2. If new information contradicts existing facts in the knowledge state, keep \
BOTH versions and note the contradiction with dates/sources where available.
3. Preserve all existing facts unless they are directly superseded by the new \
information (e.g., "release date: TBD" updated to "release date: March 2026").
4. Maintain the same structured format. Only use categories that have content.
5. If approaching the token limit, compress older facts by combining related \
points — but do not delete them entirely unless they are fully superseded.
6. Drop facts from the knowledge state that are not relevant to the topic description. \
The summary should stay focused on what the user asked to monitor.

Stay under {max_tokens} tokens. Set sufficient_data=false only if the new \
findings are too vague or contradictory to incorporate meaningfully."""

_KNOWLEDGE_UPDATE_USER = """\
Topic: {topic_name}
Description: {topic_description}

Current Knowledge State:
{current_summary}

New Findings to Incorporate:
Summary: {novelty_summary}
Key Facts:
{key_facts}"""


def _content_quality_tag(content: str | None) -> str:
    """Classify article content quality for the LLM."""
    if not content:
        return "[NO CONTENT]"
    if len(content) < 200:
        return "[STUB — minimal content, low reliability]"
    return ""


def _format_articles(articles: list[Article], max_content_chars: int = _PROMPT_ARTICLE_MAX_CHARS) -> str:
    """Format articles as a numbered list with quality indicators."""
    parts: list[str] = []
    for i, article in enumerate(articles, 1):
        content = article.raw_content or ""
        tag = _content_quality_tag(content)
        if not content:
            content = "(no content available)"
        elif len(content) > max_content_chars:
            # Truncate at last sentence boundary within budget, fall back to word boundary
            truncated = content[:max_content_chars]
            last_period = truncated.rfind(". ")
            if last_period > max_content_chars // 2:
                content = truncated[: last_period + 1]
            else:
                last_space = truncated.rfind(" ")
                content = truncated[:last_space] + "..." if last_space > 0 else truncated + "..."
        source = article.source_feed or "unknown"
        header = f"[{i}] {article.title}\n    URL: {article.url}\n    Source: {source}"
        if tag:
            header += f"\n    {tag}"
        parts.append(f"{header}\n    Content: {content}")
    return "\n\n".join(parts)


def build_novelty_messages(articles: list[Article], knowledge_summary: str, topic: Topic) -> list[dict]:
    """Build chat messages for novelty detection."""
    effective_summary = knowledge_summary or "No existing knowledge state."
    return [
        {"role": "system", "content": _NOVELTY_SYSTEM},
        {
            "role": "user",
            "content": _NOVELTY_USER.format(
                topic_name=topic.name,
                topic_description=topic.description,
                knowledge_summary=effective_summary,
                articles=_format_articles(articles),
            ),
        },
    ]


def build_knowledge_init_messages(articles: list[Article], topic: Topic, max_tokens: int) -> list[dict]:
    """Build chat messages for initial knowledge state generation."""
    return [
        {
            "role": "system",
            "content": _KNOWLEDGE_INIT_SYSTEM.format(max_tokens=max_tokens),
        },
        {
            "role": "user",
            "content": _KNOWLEDGE_INIT_USER.format(
                topic_name=topic.name,
                topic_description=topic.description,
                articles=_format_articles(articles),
            ),
        },
    ]


def build_knowledge_update_messages(
    current_summary: str,
    novelty_summary: str,
    key_facts: list[str],
    topic: Topic,
    max_tokens: int,
) -> list[dict]:
    """Build chat messages for knowledge state update."""
    facts_formatted = "\n".join(f"- {fact}" for fact in key_facts) if key_facts else "- (none)"
    return [
        {
            "role": "system",
            "content": _KNOWLEDGE_UPDATE_SYSTEM.format(max_tokens=max_tokens),
        },
        {
            "role": "user",
            "content": _KNOWLEDGE_UPDATE_USER.format(
                topic_name=topic.name,
                topic_description=topic.description,
                current_summary=current_summary,
                novelty_summary=novelty_summary or "(no summary)",
                key_facts=facts_formatted,
            ),
        },
    ]
