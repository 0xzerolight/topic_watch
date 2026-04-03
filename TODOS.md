# TODOs

Deferred work items from eng review and design discussions.

## GDELT as Third News Provider

**What:** Add GDELT (Global Database of Events, Language, and Tone) as a third news provider.

**Why:** Diversifies beyond Bing+Google. GDELT covers global events with no rate limiting, providing a fallback that doesn't depend on commercial search engines.

**Pros:** Better geographic coverage, no rate limiting, tests the Protocol's extensibility beyond RSS.

**Cons:** Returns JSON (not RSS), so the `NewsProvider` Protocol needs a `fetch_entries()` method or a JSON-to-FeedEntry adapter. Non-trivial change to the provider interface.

**Context:** Design doc Open Question #1. The current `build_feed_url()` + `fetch_feed()` pattern only works for RSS providers. GDELT would need either a new Protocol method or an adapter that converts JSON responses to `FeedEntry` objects.

**Depends on:** Multi-provider PR (provider registry + router).

## Per-Topic Provider Configuration

**What:** Add user-facing config (`config.yml`) to set provider priority order and enable/disable providers globally or per topic.

**Why:** Power users on restricted networks (China, Russia) may need to disable Google entirely. Per-topic pinning lets users optimize provider choice for specific topic domains.

**Pros:** Maximum flexibility for self-hosted users on restricted networks.

**Cons:** Adds config surface area. Violates "user doesn't think about it" for the common case. Only valuable with 3+ providers.

**Context:** Design doc Approach C, rejected for the initial multi-provider PR. Auto-routing handles the common case. Only worth building when the provider ecosystem grows.

**Depends on:** Multi-provider PR + at least one more provider (e.g. GDELT).
