"""Prompt templates for LLM-based novelty detection and knowledge management.

Each builder function returns a list of chat messages (system + user)
ready to be passed to the LLM via instructor/litellm.
"""

from app.models import Article, Topic

_PROMPT_ARTICLE_MAX_CHARS = 1000

# --- Novelty detection ---

_NOVELTY_SYSTEM = """\
You are a news novelty detector. Your job is to determine whether new articles \
contain genuinely new information about a topic that is NOT already covered in \
the existing knowledge state.

You must be CONSERVATIVE. The user only wants to be notified about meaningful \
updates — not reworded versions of known facts, not speculation that repeats \
existing rumors, not "roundup" articles that summarize old news.

Mark has_new_info=true ONLY when you find:
- Concrete new facts (dates, names, numbers) not in the knowledge state
- Official announcements or confirmations of previously unconfirmed info
- Meaningful status changes (e.g., "delayed" → "released", "rumored" → "confirmed")

Mark has_new_info=false when:
- Articles restate information already in the knowledge state
- The "new" information is just rewording or minor rephrasing
- Articles are speculation without new supporting evidence
- The article is about a different but similar topic

Set confidence to how certain you are about your decision (0.0-1.0).
If has_new_info is true, list the specific new facts in key_facts and the \
source article URLs in source_urls. Provide a brief summary of the new information."""

_NOVELTY_USER = """\
Topic: {topic_name}
Description: {topic_description}

Current Knowledge State:
{knowledge_summary}

New Articles:
{articles}"""

# --- Knowledge initialization ---

_KNOWLEDGE_INIT_SYSTEM = """\
You are building an initial knowledge state for a topic the user wants to \
monitor. Analyze the provided articles and create a comprehensive summary \
of everything currently known.

Format the summary as a structured list of facts, grouped by category:
- **Status:** [current overall status]
- **Key Facts:** [bullet points of confirmed information]
- **Recent Developments:** [what happened most recently]
- **Upcoming/Expected:** [known future dates, events, plans]
- **Unconfirmed:** [rumors or unverified claims, clearly labeled]

Be factual and concise. Do not include article-level detail — synthesize \
across all articles into a single unified summary. Stay under {max_tokens} tokens."""

_KNOWLEDGE_INIT_USER = """\
Topic: {topic_name}
Description: {topic_description}

Articles to analyze:
{articles}"""

# --- Knowledge update ---

_KNOWLEDGE_UPDATE_SYSTEM = """\
You are updating an existing knowledge state with newly discovered information. \
Incorporate the new findings while keeping the summary under {max_tokens} tokens.

Rules:
- Add new facts to the appropriate category
- If new info contradicts existing info, update it and note the change
- If approaching the token limit, compress older/less relevant facts
- Preserve the same structured format
- Do NOT remove information just because it's old — only compress when needed for space"""

_KNOWLEDGE_UPDATE_USER = """\
Topic: {topic_name}
Description: {topic_description}

Current Knowledge State:
{current_summary}

New Findings to Incorporate:
Summary: {novelty_summary}
Key Facts:
{key_facts}"""


def _format_articles(articles: list[Article], max_content_chars: int = _PROMPT_ARTICLE_MAX_CHARS) -> str:
    """Format articles as a numbered list for inclusion in prompts."""
    parts: list[str] = []
    for i, article in enumerate(articles, 1):
        content = article.raw_content or "(no content available)"
        if len(content) > max_content_chars:
            content = content[:max_content_chars] + "..."
        parts.append(f"[{i}] {article.title}\n    URL: {article.url}\n    Content: {content}")
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
