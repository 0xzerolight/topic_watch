"""Prepare-then-apply knowledge writes, for tests that used to call one function.

``initialize_knowledge`` / ``update_knowledge`` were split into a connection-free
``prepare_*`` LLM phase and an ``apply_knowledge_update`` write that runs inside
the caller's transaction, so the pipeline can hold no connection across the LLM
await. Tests that care about the resulting stored state — rather than about where
the phase boundary falls — drive both halves through these helpers instead of
restating the pair at every call site.
"""

import sqlite3
from dataclasses import dataclass

from app.analysis.knowledge import (
    KnowledgeUpdatePlan,
    apply_knowledge_update,
    prepare_initial_knowledge,
    prepare_knowledge_update,
)
from app.analysis.llm import NoveltyResult, TokenUsage
from app.config import Settings
from app.crud import get_knowledge_state
from app.models import Article, KnowledgeRevisionSource, KnowledgeState, Topic


@dataclass
class WriteResult:
    """What a prepare+apply round produced: the stored state, cost, sufficiency."""

    state: KnowledgeState
    usage: TokenUsage
    sufficient_data: bool


def apply_plan(
    conn: sqlite3.Connection,
    topic: Topic,
    plan: KnowledgeUpdatePlan,
    source: KnowledgeRevisionSource,
    settings: Settings,
    change_note: str | None = None,
) -> bool:
    """Apply a plan against the state's current version and commit."""
    assert topic.id is not None
    current = get_knowledge_state(conn, topic.id)
    applied = apply_knowledge_update(
        conn,
        topic.id,
        plan,
        expected_version=current.version if current else 0,
        source=source,
        settings=settings,
        change_note=change_note,
    )
    conn.commit()
    return applied


def _stored(conn: sqlite3.Connection, topic: Topic) -> KnowledgeState:
    assert topic.id is not None
    state = get_knowledge_state(conn, topic.id)
    assert state is not None
    return state


async def init_knowledge(
    topic: Topic,
    articles: list[Article],
    conn: sqlite3.Connection,
    settings: Settings,
) -> WriteResult:
    """Build and store an initial knowledge state.

    An insufficient-data verdict is still stored: the explanatory summary is the
    baseline the next check builds on.
    """
    plan = await prepare_initial_knowledge(topic, articles, settings)
    apply_plan(conn, topic, plan, KnowledgeRevisionSource.INIT, settings)
    return WriteResult(state=_stored(conn, topic), usage=plan.usage, sufficient_data=plan.sufficient_data)


async def update_knowledge(
    topic: Topic,
    novelty: NoveltyResult,
    conn: sqlite3.Connection,
    settings: Settings,
) -> WriteResult:
    """Merge novelty into the stored state.

    An insufficient-data verdict preserves the existing state, mirroring the
    pipeline: the plan is prepared but never applied.
    """
    assert topic.id is not None
    current = get_knowledge_state(conn, topic.id)
    if current is None:
        raise ValueError(f"No knowledge state found for topic '{topic.name}' (id={topic.id})")
    plan = await prepare_knowledge_update(topic, novelty, current.summary_text, settings)
    if plan.sufficient_data:
        # The pipeline stamps the novelty summary as the revision's change note.
        apply_plan(conn, topic, plan, KnowledgeRevisionSource.UPDATE, settings, change_note=novelty.summary)
    return WriteResult(state=_stored(conn, topic), usage=plan.usage, sufficient_data=plan.sufficient_data)
