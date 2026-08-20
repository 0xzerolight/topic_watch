"""LLM wrapper for novelty detection and knowledge state generation.

Uses Instructor + LiteLLM for structured output with automatic
validation retry. All LLM calls go through this module.
"""

import asyncio
import json
import logging
import random
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, TypeVar, cast

import instructor
import litellm
from instructor import Mode
from instructor.core import AsyncValidationError as InstructorAsyncValidationError
from instructor.core import InstructorRetryException
from instructor.core import ResponseParsingError as InstructorResponseParsingError
from pydantic import BaseModel, Field, ValidationInfo, model_validator
from pydantic import ValidationError as PydanticValidationError
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt

from app.analysis.citations import strip_index_citations, strip_reliability_notes
from app.analysis.prompts import (
    build_knowledge_compress_messages,
    build_knowledge_init_messages,
    build_knowledge_update_messages,
    build_novelty_messages,
)

# Back-compat re-exports: the restatement-filter algorithm moved to
# app/analysis/restatement.py (OVH-178); keep these importable from here.
from app.analysis.restatement import (
    _is_restatement as _is_restatement,
)
from app.analysis.restatement import (
    _longest_contiguous_run as _longest_contiguous_run,
)
from app.analysis.restatement import (
    _normalize_for_match as _normalize_for_match,
)
from app.analysis.restatement import (
    filter_restated_key_facts,
)
from app.config import Settings
from app.models import Article, Topic

logger = logging.getLogger(__name__)


# --- Token usage ---


@dataclass(frozen=True)
class TokenUsage:
    """Per-call LLM token consumption, extracted from the raw completion.

    Both fields default to 0 when usage is unavailable (some providers omit it,
    or the call short-circuited to a safe default before any LLM round-trip).
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0


def _extract_usage(completion: Any) -> TokenUsage:
    """Pull prompt/completion token counts off a raw litellm completion.

    Returns ``TokenUsage(0, 0)`` if the completion has no usable usage block
    (missing attribute, None, or non-integer values) so callers never crash on
    provider-specific shapes.
    """
    usage = getattr(completion, "usage", None)
    if usage is None:
        return TokenUsage()

    def _coerce(value: Any) -> int:
        try:
            return int(value) if value is not None else 0
        except (TypeError, ValueError):
            return 0

    # litellm.Usage supports both attribute and mapping access depending on provider.
    prompt = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    if prompt is None and isinstance(usage, dict):
        prompt = usage.get("prompt_tokens")
    if completion_tokens is None and isinstance(usage, dict):
        completion_tokens = usage.get("completion_tokens")
    return TokenUsage(prompt_tokens=_coerce(prompt), completion_tokens=_coerce(completion_tokens))


# --- Response models (structured output) ---


# Validation-context flag marking a decode as LIVE provider output rather than a
# stored blob being re-read. The two have opposite needs — see NoveltyResult —
# and the context is the seam that lets one class serve both.
_LIVE_OUTPUT_CONTEXT = {"live_llm_output": True}
_SCORE_FIELDS = ("relevance", "importance")


class NoveltyResult(BaseModel):
    """LLM response for novelty detection, decoded under one of two contracts.

    STORED (default): every scoring field has a default, so an ``llm_response``
    blob written before that field existed still re-parses — the force-notify
    handler calls ``model_validate_json`` on exactly those blobs.

    LIVE (``context=_LIVE_OUTPUT_CONTEXT``, used by ``analyze_articles``): a
    POSITIVE result must carry its own ``relevance`` and ``importance``. Those
    defaults are load-bearing downstream — ``relevance=0.0`` is below every usable
    threshold and ``importance=3`` is below a threshold of 4 or 5 — so a provider
    that simply omits the field would silently mute a genuine update with nothing
    in the logs (AUG-159 / TW-AUD-009). Omission is told apart from a real zero via
    ``model_fields_set``: an explicit ``relevance: 0.0`` is the model's own
    judgement and passes.

    A live violation raises ``ValidationError``, which instructor re-prompts with
    the message (the model usually complies, so the update is DELIVERED rather
    than suppressed); if it never complies, ``analyze_articles`` maps it through
    the settled safe-false path with ``error`` populated, so the checker records
    ``analysis_failed`` instead of a silent skip. The requirement is scoped to
    ``has_new_info=true`` because nothing is gated on either score otherwise.

    ``prompt_tokens`` / ``completion_tokens`` are NOT filled by the LLM — they
    default to 0 and are populated from the raw completion's usage after the
    call (0 on the safe-default error path or when the provider omits usage).
    """

    reasoning: str = Field(default="", description="Brief chain-of-thought: what you compared, why you decided.")
    has_new_info: bool
    # Consumed by the knowledge-update prompt's "New Findings to Incorporate"
    # block, so the model must populate it whenever there is new info (OVH-026).
    summary: str | None = Field(
        default=None,
        description=(
            "A one-to-two sentence neutral summary of the new development. "
            "Required when has_new_info is true; null only when has_new_info is false."
        ),
    )
    key_facts: list[str] = []
    source_urls: list[str] = []
    confidence: float = Field(ge=0.0, le=1.0)
    # Default (not required) is deliberate: see the class docstring — stored blobs
    # predate this field. 0.0 is the value the checker's relevance gate reads as
    # "off-topic", which is why a LIVE omission must never reach it (AUG-159).
    relevance: float = Field(
        ge=0.0,
        le=1.0,
        default=0.0,
        description="How relevant the new information is to the topic description (0=off-topic, 1=exactly what user asked)",
    )
    # Default (not required) is deliberate: stored llm_response blobs predate this
    # field and are re-parsed via model_validate_json in the force-notify handler —
    # a required field would break the Notify re-send button for pre-existing
    # checks. 3 is the neutral midpoint so an old blob doesn't hard-suppress.
    importance: int = Field(
        ge=1,
        le=5,
        default=3,
        description="How significant the new development is for someone monitoring this topic (1=trivial, 5=major)",
    )
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Set ONLY on the fail-safe error path (LLM call failed). Lets the caller
    # distinguish a genuine analysis failure from a clean "nothing new" result
    # without making analyze_articles raise (settled decision #3). None on
    # every successful call, including a legitimate has_new_info=False. The
    # description instructs the model not to populate it; analyze_articles also
    # force-resets it on the success path (belt-and-suspenders).
    error: str | None = Field(
        default=None,
        description="Internal error channel; the model must always leave this null.",
    )

    @model_validator(mode="after")
    def _require_scores_on_live_positive(self, info: ValidationInfo) -> "NoveltyResult":
        if not (info.context or {}).get("live_llm_output") or not self.has_new_info:
            return self
        missing = [name for name in _SCORE_FIELDS if name not in self.model_fields_set]
        if missing:
            raise ValueError(
                f"{' and '.join(missing)} must be set explicitly when has_new_info is true "
                "(they decide whether the user is notified); do not omit them."
            )
        return self


class KnowledgeStateUpdate(BaseModel):
    """LLM response for knowledge state init/update.

    ``prompt_tokens`` / ``completion_tokens`` are populated from the raw
    completion's usage after the call (not filled by the LLM); they default to
    0 when the provider omits usage.
    """

    sufficient_data: bool = Field(
        description=(
            "False ONLY when the articles are entirely off-topic (unrelated to the description) "
            "or establish no current state at all relevant to the description. "
            "A negative or not-yet-occurred current state (e.g. 'X has not returned', "
            "'the ban remains in place') IS sufficient — set this to true in that case."
        )
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="How confident you are in the accuracy of this summary based on source articles."
    )
    updated_summary: str = Field(
        description="The knowledge summary. If sufficient_data is false, explain what information was missing."
    )
    token_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class CompressedKnowledge(BaseModel):
    """LLM response for knowledge-state compression.

    ``prompt_tokens`` / ``completion_tokens`` are populated from the raw
    completion's usage after the call (not filled by the LLM); they default to
    0 when the provider omits usage. They let the compression round-trip's cost
    flow into the per-check token totals instead of vanishing (OVH-129).
    """

    compressed_summary: str = Field(
        description="The condensed knowledge summary: same facts, less verbosity, within the token budget."
    )
    token_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


# --- Helpers ---

# Structured-output response model, so ``_create_structured`` stays generic while
# each call site keeps its concrete model type (mypy passes without new ignores).
T = TypeVar("T", bound=BaseModel)


def _summarize_exc(exc: BaseException, *, limit: int = 200) -> str:
    """One-line, length-bounded summary of an exception for stored error fields."""
    summary = f"{type(exc).__name__}: {exc}".replace("\n", " ").strip()
    return summary[:limit]


# Parse failures instructor is *supposed* to retry: the model returned something
# that does not fit the response schema, so re-prompting it with the validation
# error genuinely can fix the next attempt. Everything else that can come out of
# a call is a TRANSPORT outcome and belongs to ``_call_with_transport_retry``,
# which is the only layer that waits between attempts (AUG-325). These are the
# same types instructor's own retry loop classifies as re-askable.
_RETRYABLE_PARSE_ERRORS: tuple[type[BaseException], ...] = (
    PydanticValidationError,
    json.JSONDecodeError,
    InstructorAsyncValidationError,
    InstructorResponseParsingError,
)

# HTTP statuses no retry can fix: the identical request fails identically. A 400
# may still MODE-HOP first (see ``_fallback_mode``) — that changes the request,
# which is why it is not simply "permanent" — but it is never re-sent unchanged.
_PERMANENT_STATUS_CODES = frozenset({400, 401, 403, 404, 405, 413, 422})
# Statuses worth re-sending the same request for, after a wait.
_TRANSIENT_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def _instructor_retries(max_retries: int) -> AsyncRetrying:
    """Build instructor's per-call retry policy: validation re-prompts ONLY.

    Instructor's ``max_retries`` governs structured-output *validation* retries
    (re-prompting when the response fails Pydantic validation), and the classifier
    is POSITIVE — only ``_RETRYABLE_PARSE_ERRORS`` are retried here (AUG-325).
    A negative "retry everything except these types" list let transport failures
    into this loop, where they were re-fired ``max_retries`` times with ZERO delay
    and no ``Retry-After`` handling: a throttled provider got hammered (OVH-008), a
    transient 5xx got no backoff, and a permanent error was replayed and then
    buried inside an opaque ``RetryError[...]`` (issue #53). Every provider
    exception now leaves instructor after one attempt, with the real error on
    ``__cause__``, and ``_call_with_transport_retry`` decides its fate.
    """
    return AsyncRetrying(
        stop=stop_after_attempt(max_retries + 1),
        retry=retry_if_exception_type(_RETRYABLE_PARSE_ERRORS),
    )


def _iter_error_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield ``exc`` and every exception reachable from it, once each.

    Instructor's v2 retry stack re-wraps a provider error as
    ``InstructorRetryException``: it stringifies the error into ``args``, may
    record it in ``failed_attempts``, and chains the real one onto ``__cause__``.
    Callers therefore have to look past the wrapper to classify what happened —
    all three places are walked here (cycle-guarded) so each classifier below is
    a one-line filter over this iterator.
    """
    seen: set[int] = set()
    queue: list[BaseException] = [exc]
    while queue:
        cur = queue.pop(0)
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))
        yield cur
        if isinstance(cur, InstructorRetryException):
            queue.extend(arg for arg in cur.args if isinstance(arg, BaseException))
            for attempt in cur.failed_attempts or []:
                attempt_exc = getattr(attempt, "exception", None)
                if isinstance(attempt_exc, BaseException):
                    queue.append(attempt_exc)
        for nxt in (cur.__cause__, cur.__context__):
            if nxt is not None:
                queue.append(nxt)


def _unwrap_rate_limit(exc: BaseException) -> litellm.RateLimitError | None:
    """Return the underlying ``RateLimitError`` if ``exc`` represents a 429."""
    return next((e for e in _iter_error_chain(exc) if isinstance(e, litellm.RateLimitError)), None)


def _status_code(exc: BaseException) -> int | None:
    """HTTP status carried by ``exc`` or anything in its chain, else None.

    Classification is by STATUS, not by exception type: a gateway or adapter that
    raises a generic ``litellm.APIError(status_code=400)`` instead of the specific
    ``BadRequestError`` subclass described the same permanent failure, and used to
    be treated as retryable purely because of its Python type (AUG-325).
    """
    for candidate in _iter_error_chain(exc):
        code = getattr(candidate, "status_code", None)
        if isinstance(code, int):
            return code
        if isinstance(code, str) and code.isdigit():
            return int(code)
    return None


def _unwrap_bad_request(exc: BaseException) -> BaseException | None:
    """Return the 400 in ``exc``'s chain, else None.

    A bare 400 is returned as-is; otherwise the wrapper chain is walked, because
    instructor re-wraps an unretried 400 as ``InstructorRetryException`` with the
    real error on ``__cause__`` — a bare 400 never propagates from
    ``create_with_completion``. Returns ``None`` for anything that is not a 400
    (e.g. a rate-limit wrapper), so those keep flowing untouched to their own
    handler.
    """
    return next(
        (e for e in _iter_error_chain(exc) if isinstance(e, litellm.BadRequestError) or _has_status(e, 400)),
        None,
    )


def _has_status(exc: BaseException, code: int) -> bool:
    status = getattr(exc, "status_code", None)
    return status == code or (isinstance(status, str) and status.isdigit() and int(status) == code)


def _is_transient(exc: BaseException) -> bool:
    """True when re-sending the SAME request after a wait could plausibly work.

    Transient: 429/408/5xx and network-level failures (no response at all).
    Permanent: every other classified status, and anything unclassifiable — an
    unknown failure is not replayed, so a bug in this module cannot turn one
    check into a retry storm.
    """
    if any(isinstance(e, litellm.APIConnectionError | litellm.Timeout) for e in _iter_error_chain(exc)):
        return True
    status = _status_code(exc)
    if status is None:
        return False
    return status in _TRANSIENT_STATUS_CODES or (status >= 500 and status not in _PERMANENT_STATUS_CODES)


# Ordered structured-output fallback chain. TOOLS sends a forced named
# ``tool_choice`` (rejected by e.g. DeepSeek thinking mode); JSON sends
# ``response_format={"type":"json_object"}``; MD_JSON uses plain prompting +
# markdown-JSON parsing and needs no special provider support — the guaranteed
# terminal (absent from the map, so ``_fallback_mode`` returns None there).
_MODE_FALLBACK: dict[instructor.Mode, instructor.Mode] = {Mode.TOOLS: Mode.JSON, Mode.JSON: Mode.MD_JSON}


def _fallback_mode(mode: instructor.Mode, exc: BaseException) -> instructor.Mode | None:
    """Next structured-output mode to try after ``exc``, or ``None`` to give up.

    The criterion is STRUCTURAL: fall back only on a 400 a mode switch could
    plausibly fix (e.g. a forced ``tool_choice`` rejection), never on a
    mode-INVARIANT error — one caused by a request property that
    ``_create_structured`` sends identically in every mode, so switching cannot
    change the outcome. Two invariants are excluded:

    - ``ContextWindowExceededError``: prompt size is the same in every mode.
    - a ``max_tokens`` 400: ``_bounded_max_tokens`` passes the same ceiling in all
      three modes, so this guard is about that mode-invariance, NOT about trusting
      provider phrasing in general — every other 400 falls back by design, so
      MD_JSON stays the terminal no matter how an unknown gateway words its
      rejection.

    Anything that is not an unwrapped 400 (e.g. a 429 wrapper) returns ``None`` so
    it propagates to its own handler untouched. A 400 is recognized by STATUS as
    well as by type, so a gateway raising a generic ``APIError(status_code=400)``
    for a rejected structured-output mode still mode-hops instead of being
    replayed in the same mode (AUG-325).
    """
    bad_request = _unwrap_bad_request(exc)
    if bad_request is None or isinstance(bad_request, litellm.ContextWindowExceededError):
        return None
    if "max_tokens" in str(bad_request).lower():
        return None
    return _MODE_FALLBACK.get(mode)


# One instructor client per structured-output mode. The mode is baked into the
# client at ``from_litellm`` time (it decides tool_choice vs response_format vs
# markdown prompting), so the TOOLS -> JSON -> MD_JSON fallback needs a distinct
# client per mode. Each wraps the stateless ``litellm.acompletion`` (model, key,
# base_url passed per call), so one cached client per mode is reused across calls.
_clients: dict[instructor.Mode, instructor.AsyncInstructor] = {}


def _get_client(settings: Settings, mode: instructor.Mode = instructor.Mode.TOOLS) -> instructor.AsyncInstructor:
    """Return a cached async instructor-patched litellm client for ``mode``.

    Built lazily per mode and memoized in ``_clients``. ``settings`` is accepted
    for call-site symmetry and to keep ``_get_client`` the patch seam used by
    single-mode tests. Default ``Mode.TOOLS`` matches instructor's own default.
    """
    if mode not in _clients:
        _clients[mode] = cast(instructor.AsyncInstructor, instructor.from_litellm(litellm.acompletion, mode=mode))
    return _clients[mode]


def _effective_base_url(settings: Settings) -> str | None:
    """Return the configured LLM base_url, or None when unset.

    An explicitly-set base_url is honored for every provider (OVH-104 reversal):
    it points litellm at an OpenAI-compatible gateway (e.g. OpenCode Go), a
    LiteLLM proxy, or a self-hosted server (Ollama). Kept as a thin seam so the
    four call sites and their tests have one place to patch.
    """
    return settings.llm.base_url or None


# Internal output-token ceiling. When a call omits ``max_tokens``, litellm's
# Anthropic path injects the model's FULL max output (64000 for claude-haiku-4-5),
# which Anthropic rejects for non-streaming requests — the underlying issue #53
# 400. Capping output well below that avoids it and bounds cost/latency. Not a user
# setting: it's a safety bound, and the structured outputs here are small. 8192 is
# comfortably above the default knowledge-summary budget (2000) yet safely below
# any provider's non-streaming ceiling.
_OUTPUT_TOKEN_CAP = 8192


def _bounded_max_tokens(settings: Settings) -> int:
    """Explicit ``max_tokens`` for LLM calls: a hard, per-model output ceiling.

    Returns ``_OUTPUT_TOKEN_CAP``, clamped down to the model's own max output so a
    model whose ceiling is below the cap (e.g. 4096-output models) is not
    over-asked — a flat 8192 would 400 there. This is a HARD ceiling: it never
    rises above ``_OUTPUT_TOKEN_CAP`` even for a large ``knowledge_state_max_tokens``.
    Letting it exceed the cap would re-expose issue #53 (an oversized non-streaming
    ``max_tokens`` rejected with a 400) for the exact 64k-output Anthropic models
    where the bug originated. A summary budget above the cap is therefore bounded
    to the cap; ``knowledge_state_max_tokens`` remains the *prompt* budget the model
    is told to stay under, which is the effective control for summary length.

    ``litellm.get_max_tokens`` returns max *output* tokens (not the context
    window) and RAISES for unmapped/gateway model strings; the ``or`` handles its
    ``None`` return and the ``except`` handles the raise — either way we fall back
    to the cap rather than crashing every call for gateway users.
    """
    try:
        model_max = litellm.get_max_tokens(settings.llm.model) or _OUTPUT_TOKEN_CAP
    except Exception:
        model_max = _OUTPUT_TOKEN_CAP
    return min(_OUTPUT_TOKEN_CAP, model_max)


# Models whose tokenizer has already failed once. The char/4 fallback diverges
# from a real model tokenizer (OVH-136), so budget decisions made on it run on a
# wrong unit; surface that as a WARNING. Cached per model so a broken tokenizer
# does not flood the log with one line per count_tokens call — the operator sees
# the divergence once per model and can correct the model id / tokenizer asset.
_token_fallback_warned: set[str] = set()


def count_tokens(text: str, model: str) -> int:
    """Count tokens using litellm's model-aware tokenizer.

    Falls back to ``len(text) // 4`` if the tokenizer fails. Because that
    char-based estimate systematically diverges from the model tokenizer
    (non-English/structured text especially), the first fallback for a given
    model is logged at WARNING so budget enforcement running on the wrong unit is
    observable (OVH-136); subsequent fallbacks for the same model stay quiet.
    """
    try:
        return litellm.token_counter(model=model, text=text)  # type: ignore[no-any-return]
    except Exception:
        if model not in _token_fallback_warned:
            _token_fallback_warned.add(model)
            logger.warning(
                "Token counting failed for model %s; using char/4 fallback — token-budget "
                "decisions for this model are approximate until the tokenizer is available",
                model,
                exc_info=True,
            )
        else:
            logger.debug("Token counting failed for model %s, using fallback", model)
        return len(text) // 4


# Bounds on the transport backoff (AUG-160). Uncapped ``base_delay *
# multiplier**attempt`` reached 98,415s on the tenth sleep at the supported
# maximum ``llm_max_retries=10`` — one throttled check held the scheduler's
# single job instance for ~41 hours, skipping every later minute tick. Both are
# module constants, not settings: they are liveness bounds on a shared runtime,
# not a per-deployment preference.
#
# The total budget is the load-bearing one — it is what makes the worst case
# finite regardless of ``llm_max_retries`` — and 120s sits well under
# ``interval.MIN_INTERVAL_MINUTES`` (10 minutes), so even the shortest check
# interval cannot be overrun by backoff alone.
_RETRY_MAX_DELAY_SECONDS = 60.0
_RETRY_TOTAL_BUDGET_SECONDS = 120.0
# Jitter spreads the retries of concurrently-checked topics that hit the same
# provider limit in the same tick, so they do not re-fire in lockstep.
_RETRY_JITTER_FRACTION = 0.25


def _backoff_delay(attempt: int, base_delay: float, backoff_multiplier: float) -> float:
    """Capped exponential delay with bounded upward jitter."""
    delay = min(base_delay * (backoff_multiplier**attempt), _RETRY_MAX_DELAY_SECONDS)
    # Not a security decision — retry spacing only (hence the S311 exemption).
    return delay + random.uniform(0, delay * _RETRY_JITTER_FRACTION)  # noqa: S311


async def _call_with_transport_retry(
    call_func: Any,
    max_retries: int = 3,
    base_delay: float = 5.0,
    backoff_multiplier: float = 3.0,
) -> Any:
    """Wrap an async LLM call with the ONE delayed retry policy for transport failures.

    Retries only what re-sending the same request can fix — 429, 408, 5xx and
    network failures (``_is_transient``) — waiting ``_backoff_delay`` between
    attempts. Permanent failures and structured-output rejections are re-raised
    immediately for their own handlers (``_fallback_mode``, the callers' safe-false
    / raise split).

    Two bounds hold the worst case (AUG-160): each delay is capped, and the whole
    sequence stops at ``_RETRY_TOTAL_BUDGET_SECONDS`` measured on the event loop's
    MONOTONIC clock, whichever comes first. ``max_retries`` is therefore an upper
    bound on attempts, not a promise of them.
    """
    deadline = time.monotonic() + _RETRY_TOTAL_BUDGET_SECONDS
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return await call_func()
        except Exception as exc:
            if not _is_transient(exc):
                raise
            last_exc = exc
            if attempt >= max_retries:
                break
            delay = _backoff_delay(attempt, base_delay, backoff_multiplier)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning("Retry budget of %.0fs exhausted; giving up", _RETRY_TOTAL_BUDGET_SECONDS)
                break
            delay = min(delay, remaining)
            if _unwrap_rate_limit(exc) is not None:
                logger.warning("Rate limit hit (attempt %d/%d), retrying in %.0fs", attempt + 1, max_retries, delay)
            else:
                logger.warning(
                    "Transient LLM error (attempt %d/%d), retrying in %.0fs: %s",
                    attempt + 1,
                    max_retries,
                    delay,
                    _summarize_exc(exc),
                )
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


async def _create_structured(
    settings: Settings,
    *,
    response_model: type[T],
    build_messages: Callable[[], list[dict[str, Any]]],
    timeout: int,
    validation_context: dict[str, Any] | None = None,
) -> tuple[T, Any]:
    """Run one structured-output call, falling back TOOLS -> JSON -> MD_JSON.

    On a structured-output-fixable 400 (see ``_fallback_mode``) the call is retried
    in the next mode; a mode-invariant error re-raises immediately, and MD_JSON is
    the terminal. ``build_messages`` is a factory invoked FRESH per attempt: this
    preserves the rebuild-per-attempt invariant and avoids instructor's in-place
    message mutation leaking a doubled schema block across mode hops.

    Stateless by design — no per-model memory, so a false-positive fallback costs
    one call and never downgrades the process.

    ``validation_context`` reaches the response model's validators as pydantic's
    validation context, and instructor re-prompts on whatever they reject — that
    is how the live-output contract (see ``NoveltyResult``) is enforced against
    the provider rather than merely detected after the fact.
    """
    mode: instructor.Mode = instructor.Mode.TOOLS
    while True:
        client = _get_client(settings, mode)
        try:
            result = await client.chat.completions.create_with_completion(
                model=settings.llm.model,
                response_model=response_model,
                messages=build_messages(),  # type: ignore[arg-type]
                context=validation_context,
                max_retries=_instructor_retries(settings.llm_max_retries),
                api_key=settings.llm.api_key,
                api_base=_effective_base_url(settings),
                timeout=timeout,
                temperature=settings.llm_temperature,
                max_tokens=_bounded_max_tokens(settings),
            )
        except Exception as exc:
            next_mode = _fallback_mode(mode, exc)
            if next_mode is None:
                raise
            logger.warning(
                "Provider rejected %s structured-output mode for model %s; retrying with %s",
                mode.value,
                settings.llm.model,
                next_mode.value,
            )
            mode = next_mode
        else:
            return cast(tuple[T, Any], result)


# --- key_facts restatement filtering ---
#
# The phrase-matching algorithm lives in app/analysis/restatement.py (OVH-178);
# these aliases keep the historical ``app.analysis.llm`` import path working for
# call sites and tests. ``analyze_articles`` calls ``_filter_restated_key_facts``.
_filter_restated_key_facts = filter_restated_key_facts


# --- source_urls subset guard (prompt-injection output validation) ---


def _filter_source_urls(source_urls: list[str], articles: list[Article]) -> list[str]:
    """Keep only LLM-returned source_urls that match an input article URL.

    A successful-but-manipulated completion can emit an attacker-chosen
    source_url (e.g. a phishing link injected via feed text) that still passes
    schema validation and would otherwise flow into notifications/webhooks
    (OVH-058). Cross-checking against the input set drops any smuggled URL while
    preserving order and de-duplicating. Comparison is on the exact URL string,
    matching how the URLs were presented to the model.
    """
    allowed = {article.url for article in articles}
    seen: set[str] = set()
    kept: list[str] = []
    for url in source_urls:
        if url in allowed and url not in seen:
            kept.append(url)
            seen.add(url)
    return kept


# --- Public API ---


async def analyze_articles(
    articles: list[Article],
    knowledge_summary: str,
    topic: Topic,
    settings: Settings,
) -> NoveltyResult:
    """Analyze articles for novelty against the current knowledge state.

    Returns a safe default (has_new_info=False) on any LLM error
    to prevent spurious notifications. On success, ``prompt_tokens`` /
    ``completion_tokens`` are populated from the raw completion's usage, and
    ``key_facts`` that merely restate the knowledge summary are dropped.
    """

    try:
        result, completion = await _call_with_transport_retry(
            lambda: _create_structured(
                settings,
                response_model=NoveltyResult,
                build_messages=lambda: build_novelty_messages(articles, knowledge_summary, topic),
                timeout=settings.llm_analysis_timeout,
                # Decode under the LIVE contract: a positive result that omits its
                # own relevance/importance is rejected and re-prompted, instead of
                # inheriting stored-blob defaults that mute it (AUG-159).
                validation_context=_LIVE_OUTPUT_CONTEXT,
            ),
            max_retries=settings.llm_max_retries,
        )
    except Exception as exc:
        logger.warning("LLM analysis failed for topic '%s'", topic.name, exc_info=True)
        return NoveltyResult(has_new_info=False, confidence=0.0, error=_summarize_exc(exc))

    novelty: NoveltyResult = result
    # ``error`` is in the LLM's structured-output schema, so a model can populate
    # it on a clean run. Force it None here so ONLY the except-branch above ever
    # sets it; otherwise the checker mis-stamps a healthy run as analysis_failed.
    novelty.error = None
    # ``relevance``/``importance`` carry stored-blob defaults (see NoveltyResult),
    # and ``NoveltyResponse`` only *requires* them on a positive result. A provider
    # that omits them on negatives is still worth one line per check: the same
    # habit is what would mute a later positive, and the default scores are what
    # the checker's gates read.
    omitted = [name for name in _SCORE_FIELDS if name not in novelty.model_fields_set]
    if omitted:
        logger.warning(
            "LLM omitted %s for topic '%s'; using the stored-blob default(s) (relevance=%.2f, importance=%d). "
            "A threshold above the default will suppress notifications with this model.",
            " and ".join(f"'{name}'" for name in omitted),
            topic.name,
            novelty.relevance,
            novelty.importance,
        )
    usage = _extract_usage(completion)
    novelty.prompt_tokens = usage.prompt_tokens
    novelty.completion_tokens = usage.completion_tokens
    novelty.key_facts = _filter_restated_key_facts(novelty.key_facts, knowledge_summary)
    # Strip ephemeral article-index citations ("(Article [1])") then leaked
    # [STUB]/[NO CONTENT] reliability notes from the fact fields before they reach
    # the knowledge-update merge, notifications, and webhooks. Order matters:
    # index-citations first, so their parentheticals are gone before the reliability
    # pass classifies sentences. Not reasoning — its cites are subject-position prose
    # that would mangle if stripped.
    if novelty.summary:
        novelty.summary = strip_reliability_notes(strip_index_citations(novelty.summary))
    novelty.key_facts = [strip_reliability_notes(strip_index_citations(fact)) for fact in novelty.key_facts]
    # Drop any source_url not in the input set so an injected completion cannot
    # smuggle an attacker-chosen URL into notifications/webhooks (OVH-058).
    novelty.source_urls = _filter_source_urls(novelty.source_urls, articles)
    return novelty


async def generate_initial_knowledge(
    articles: list[Article],
    topic: Topic,
    settings: Settings,
) -> KnowledgeStateUpdate:
    """Generate an initial knowledge state from articles.

    Raises on failure — knowledge initialization is critical.
    """

    raw_result, completion = await _call_with_transport_retry(
        lambda: _create_structured(
            settings,
            response_model=KnowledgeStateUpdate,
            build_messages=lambda: build_knowledge_init_messages(articles, topic, settings.knowledge_state_max_tokens),
            timeout=settings.llm_knowledge_timeout,
        ),
        max_retries=settings.llm_max_retries,
    )
    result: KnowledgeStateUpdate = raw_result
    # Strip article-index citations then leaked reliability notes before counting
    # tokens so the freed budget is real.
    result.updated_summary = strip_reliability_notes(strip_index_citations(result.updated_summary))
    result.token_count = count_tokens(result.updated_summary, settings.llm.model)
    usage = _extract_usage(completion)
    result.prompt_tokens = usage.prompt_tokens
    result.completion_tokens = usage.completion_tokens
    return result


async def compress_knowledge_summary(
    current_summary: str,
    topic: Topic,
    settings: Settings,
) -> CompressedKnowledge:
    """Compress an over-budget knowledge summary while preserving its facts.

    Raises on failure — the caller decides how to degrade (e.g. fall back to
    lossy truncation). The returned ``token_count`` is recomputed authoritatively,
    and ``prompt_tokens`` / ``completion_tokens`` are populated from the raw
    completion's usage so this round-trip's cost is not lost (OVH-129).
    """

    raw_result, completion = await _call_with_transport_retry(
        lambda: _create_structured(
            settings,
            response_model=CompressedKnowledge,
            build_messages=lambda: build_knowledge_compress_messages(
                current_summary=current_summary,
                topic=topic,
                max_tokens=settings.knowledge_state_max_tokens,
            ),
            timeout=settings.llm_knowledge_timeout,
        ),
        max_retries=settings.llm_max_retries,
    )
    result: CompressedKnowledge = raw_result
    # Strip article-index citations then leaked reliability notes before counting
    # tokens so the freed budget is real.
    result.compressed_summary = strip_reliability_notes(strip_index_citations(result.compressed_summary))
    result.token_count = count_tokens(result.compressed_summary, settings.llm.model)
    usage = _extract_usage(completion)
    result.prompt_tokens = usage.prompt_tokens
    result.completion_tokens = usage.completion_tokens
    return result


async def generate_knowledge_update(
    current_summary: str,
    novelty_result: NoveltyResult,
    topic: Topic,
    settings: Settings,
) -> KnowledgeStateUpdate:
    """Update the knowledge state with new findings.

    Raises on failure — knowledge updates are critical.
    """

    raw_result, completion = await _call_with_transport_retry(
        lambda: _create_structured(
            settings,
            response_model=KnowledgeStateUpdate,
            build_messages=lambda: build_knowledge_update_messages(
                current_summary=current_summary,
                novelty_summary=novelty_result.summary or "",
                key_facts=novelty_result.key_facts,
                topic=topic,
                max_tokens=settings.knowledge_state_max_tokens,
            ),
            timeout=settings.llm_knowledge_timeout,
        ),
        max_retries=settings.llm_max_retries,
    )
    result: KnowledgeStateUpdate = raw_result
    # Strip article-index citations (the update LLM grafts them onto clean input by
    # mimicking the existing cited style) then leaked reliability notes before
    # counting tokens so the budget is real.
    result.updated_summary = strip_reliability_notes(strip_index_citations(result.updated_summary))
    result.token_count = count_tokens(result.updated_summary, settings.llm.model)
    usage = _extract_usage(completion)
    result.prompt_tokens = usage.prompt_tokens
    result.completion_tokens = usage.completion_tokens
    return result
