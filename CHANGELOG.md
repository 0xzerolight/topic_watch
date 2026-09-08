# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).
Pre-1.3.0 releases predate that policy: 1.2.1, 1.2.3 and 1.2.5 shipped
user-visible features as PATCH bumps. Versions before 1.3.0 are not a
reliable signal of change scope; historical tags are not renumbered.

## [Unreleased]

### Added

- Check Now and bulk checks are recorded before the page responds; a check a crash or restart interrupted runs on the next scheduler cycle (after the 10-minute claim lease when the crash hit mid-check) instead of being lost
- Durable per-target delivery intents for notifications and webhooks. The intent is written inside the check's own transaction, before any send, so a crash between deciding to notify and the message leaving no longer looks like no notification was ever intended. Eligibility lives inside an atomic claim and every apply is fenced by the winning claim token, so a wall-clock jump in either direction cannot re-arm a live claim or let a second drainer send the same message twice. Delivered intents are retained as the ledger "Last notified" reads, and finished rows are pruned after 30 days
- Test LLM configuration on the Settings page. A save writes the model, key and base URL with no live check, so a typo or a dead endpoint stayed active configuration until a scheduled check failed. The new action probes the values currently in the form — unsaved — through the same client, schema, mode fallback and timeout that live analysis uses; saving stays offline and explicit
- Check History rows expand to the finding the check actually stored: summary, key facts, source URLs and scores, loaded on demand and ownership-checked. Raw model reasoning is deliberately excluded
- Every feed fetch reports a typed outcome — entries, not-modified, empty, failed or aborted — replacing an inference from the length of the entry list that conflated four different things. A 304 is its own outcome, so an unchanged feed stops triggering the AUTO fallback and keeps its validators; an empty 200 and a 304 both reset a provider's failure streak; a feed that publishes entries and yields none usable is a failure rather than quiet news; a blocked or unresolvable URL records a failed fetch and enters backoff like any other broken feed. Conditional requests are replay-safe per topic, so a shared feed's 304 cannot leave a newly added topic permanently empty
- `/health` reports per-job scheduler outcomes — last success, last error, consecutive failures, missed runs — and a monitoring freshness verdict alongside liveness. One process takes an advisory lock beside the database before starting a scheduler, so extra web workers can no longer each run their own minute tick, and the tick applies edited jitter and misfire grace to the live jobs the Settings page has always reported as active
- Knowledge revisions carry lineage and provenance. A new migration records the model and a topic-scope basis hash per revision, so the token delta renders only when both sides name the same model and a topic whose scope was edited after its baseline says so. The diff reports an explicit mode: a snapshot — the oldest retained revision, a re-initialization, an unrecognised source or input past the cost cap — no longer renders as if every segment were an addition, and segment comparison keeps markdown indentation, hard breaks and paragraph boundaries, so nesting a bullet is a change rather than a no-op
- Feed Health shows who owns each source. Rows name the topics using them, are labelled Orphaned when none do, and Exa's shared search endpoint renders as a typed search source instead of a clickable feed link. The Repair column links to the owning topic's edit page, or to Settings for Exa

### Changed

- Dependencies relocked (litellm 1.99.0, instructor 1.16.0, pydantic 2.13.5, apprise 1.13.0, openai 2.54.0, ruff 0.16.6, mypy 2.3.1 and 20 more). litellm pulls boto3/botocore/s3transfer/jmespath in as unconditional runtime requirements, so they are now in `requirements.txt`. Supersedes the weekly group PRs #74 and #77, neither of which could install. Dependabot no longer proposes openai 3.x either: the cap comes from instructor, so the whole 3.x line is uninstallable here, and the ignore is range-scoped like the jiter/rich/importlib-metadata ones so 2.x bumps still flow
- The Docker image builds from pinned inputs only: hatchling is hash-locked in `requirements-build.txt`, gosu comes from its signed release with a per-architecture checksum, and pip is no longer upgraded at build time. No `apt-get` remains in the build. The image also ships license notices for the front-end assets it vendors, alongside the project's own license text. `make lock` / `make lock-upgrade` maintain the new lock; Dependabot tracks it with the other requirements files
- The topic page leads with its actions. Check Now, Edit and Enable/Disable moved into the masthead action row, so they render above the fold regardless of how long the check history is; reinitialize, export and delete moved to a Maintenance block below. The novelty instruction and its three notification gates are grouped into one Novelty Policy card instead of being scattered across the at-a-glance grid
- One story is one article. Article identity is keyed to the representation actually fetched and carries the source's revision signals, and dedup compares an entry against both that key and the story it belongs to, so a changed Google wrapper, a tracking variant, a resolved redirect and the other provider's copy of the same story stop becoming separate articles — while a genuine correction still passes. Stored hashes stay valid; a pre-upgrade article can be seen once more
- Feed backoff is a manual-source rule again, and its cap is a cap on elapsed time. The Feed Health page ran the formula over every row, so AUTO and Exa sources claimed a retry time the runtime never honors, and a failure stamped during a forward clock jump could keep a feed unfetched days past the 24-hour cap. Sub-hour delays no longer all read as "~1h"
- Settings are written atomically through the schema's own codec: a temp file and one `os.replace`, fsync of file and directory, 0600 on a new file. The manual whitelist is gone — the writer patches the model's dump onto the existing document, so unrecognized keys survive and cleared optionals are deleted instead of resurrected. Fields the environment owns are never written to `config.yml` at all and render read-only in the forms; provenance is tracked per field by the presence of its `TOPIC_WATCH_*` variable rather than by two special cases for a truthy API key, so the Secure Cookies toggle every Docker install owns stops accepting an edit and reporting a success that changed nothing
- Config and database roots resolve through one state-root helper. An existing `data/` directory still wins, so checkouts, worktrees and the container bind mount are unchanged; `TOPIC_WATCH_CONFIG_PATH` pins the root; a writable source checkout resolves to `data/` before it exists; a wheel install with no `data/` beside the package falls back to a user-level directory and never to `site-packages`
- Every LLM request is bounded against the model's context window. Article bodies were capped per item and the knowledge state had its own budget, but nothing sized knowledge plus articles plus schema overhead plus requested output together — and an overrun is not retryable. The gateway reserves output and schema tokens and degrades in the order that costs least: shrink article bodies, drop trailing articles, then trim the knowledge state being read. Notification payloads are bounded to each channel's title and body limits the same way, so an oversized alert is no longer rejected and replayed unchanged until abandoned. The structured-output mode that worked is remembered per model, base URL and response model for a bounded TTL instead of restarting from TOOLS every call, and is cached only once it has actually answered. Retry sleeps are bounded too: at `llm_max_retries=10` the delays totalled 147,620 seconds and held the single-instance scheduler job for roughly 41 hours. Transport failures are classified by status and routed through one delayed policy, instructor retries parse failures only, and prompts are no longer rendered through its templating, where a stray `{{` in a feed title failed the check before any call was made
- No database connection is held across a fetch, LLM call or send. The pipeline snapshots, fetches, analyzes and prepares outside any transaction, then commits the knowledge write, its revision, the article disposition, the check result and the delivery intents together, fenced by the topic generation and the knowledge version
- Installs and updates run a pinned image digest instead of the movable `latest` tag: `install.sh` and `install.ps1` pin the exact digest they pulled and health-checked, and `update.sh` captures the previous digest first and restarts it when the post-update health check fails. CI gates what it claimed to gate too — the `pip-audit` scan is blocking, a pushed tag builds only after the full suite passes on that exact commit with the tag matching the packaged version, `make ci` runs the same format and evals checks the hosted lint job does plus a lockfile parity gate, and the test job has a timeout and uploads a per-matrix receipt
- Async results announce themselves to screen readers: the feed-validation result, the notification test result and the browser-notification status are live regions, a topic reaching a terminal status is announced, and the feed-mode fieldset, the OPML file input and the token-budget meter have accessible names. Quiet text in the Nord, Dracula, Solarized and Tokyo Night themes now meets WCAG AA on both the page and card surfaces, and an inactive row is marked with a border instead of a 55% opacity fade that dimmed its own still-operable controls
- The eval harness records replay lineage: artifacts carry a schema version, a run id, a replay parent and the app version they were captured under, every structured-output attempt is recorded including the ones that raise, run artifacts are written 0600 inside a path-confined directory, and an errored run exits non-zero instead of reporting a coincidental match

### Fixed

- Articles stop multiplying and stop vanishing. A correction republished at a story's own URL, a moving `updated` stamp, a publisher whose clock runs fast, a feed rotating a campaign parameter, a Google headline carrying its `- Publisher` suffix and a search provider's prefetched text each used to mint a fresh article every check or drop a real revision. Bodies that cannot be read are no longer stored as evidence, entries describing one article collapse to their best copy before the cap is applied, a URL listed twice in a topic's feeds is fetched and counted once, and undated or impossibly dated entries are ranked at retrieval time rather than at year 1 — so a wholly dateless feed sharing the article cap with a busy dated one stops delivering nothing at all
- A provider cooldown actually suppresses requests. Both source helpers handed back an unhealthy provider as best effort, so during a shared outage every due topic still attempted both providers and every failure pushed both deadlines forward again. Once a cooldown elapses exactly one caller takes a half-open probe; health records survive recovery, so a recycled epoch cannot wipe a newer cooldown; and Google's resolver shares one rate-limit guard and 429 cooldown across concurrent checks instead of allocating a private one per call. Feed validation reports the fetch's own outcome rather than the length of the entry list, which made a blocked URL, a timeout, a 404 and an unparseable body all read as "Valid RSS feed with 0 entries"; and one saturated resolver no longer bankrupts unrelated feeds, since slot starvation is charged to the check instead of costing every concurrent fetch a failure streak
- Notifications are delivered once and only once, and their outcome is recorded. An exception escaping a send left the intent claimed with no outcome, so every later drain re-sent it; a single `[` in an SMTP password stranded every intent it produced; one stalled target reported failure for every target in the message; a repeated URL sent every alert twice; and a placeholder check matching on substrings refused real targets such as `ntfy://api_token-alerts` forever. Each target carries its own deadline now, a timeout is an unknown outcome rather than a failure, and a target the resolver simply could not answer for is retried instead of being destroyed on its first attempt
- A webhook target that resolves to a private or reserved address is abandoned after one attempt instead of after the whole retry budget; a target the resolver could not answer for still retries. The sender now reads `url_validation`'s three-state destination verdict instead of the fail-closed bool, which could not tell a resolved private address apart from the absence of a DNS answer
- One topic gets one runner. In-flight guards carry an owner token, so evicting a stale entry cannot let its owner release a slot someone else has taken; the single-topic background check and the checks inside check-all are bounded like every other path; a check that outlives a topic's deletion cannot file its result or latch its heartbeat against the replacement that reused the row id; and the check-all deadline is sized to the backlog instead of a count-blind 30-minute cap. Initialization is fenced behind a durable claim, so two initializers cannot both pass, Retry on a RESEARCHING topic no longer wins its own self-transition, a paused NEW topic is not promised an initialization that will never come, and a zombie cannot overwrite recovery's ERROR. A skipped non-READY topic stops being filed as a check at all, a failure that was internal rather than a source outage is recorded as such and ignored by the Silence Heartbeat, and heartbeat state is committed together with its intents and revocations, cleared when the feature is switched off, and fenced to the check the decision was computed from
- Analysis that fails is resumed rather than lost. Articles whose analysis or knowledge merge did not finish stay unprocessed and carry an attempt count, three failed cycles abandons an article, and each check re-selects the stranded rows that hash dedup can never re-offer. Only the articles the fitted prompt actually carried are marked processed. A Retry or Re-initialize rebuilds its baseline from stored articles instead of from the handful of entries that appeared since the last check, an initialization that ran no source at all says so, and an insufficient-data verdict is recorded as the refusal it is rather than as an ordinary notified success that quietly consumed the evidence
- Facts and scores survive the filters that used to eat them. Restatement matching compares word tokens rather than raw substrings, so "He won." is no longer a restatement of "She won."; citation stripping honors word boundaries and consumes whole lists; a sentence beginning "Note:" must also carry an uncertainty signal before it is removed; text is sanitized before filtering, with the non-empty invariant re-checked afterwards; a provider omitting relevance or importance no longer scores every update relevance 0.0 and importance 3, muting it; and a response that violates the live contract keeps the model's own text with the check instead of discarding a finding that may have been real
- Timestamps are trusted or refused, never invented. A required timestamp cell that will not parse raises instead of quietly becoming "now" — a value that drives scheduling, ordering, retention and the UI and sticks once anything writes the row back. Every hydrated datetime is aware UTC, the dashboard head, both acknowledgements and the history list share one ordering, due topics anchor on the newest non-future check, and a future stamp renders as future instead of collapsing to "just now"
- The dashboard toolbar holds together: search and status filters keep the active tag, tag chips are URL-encoded so a `&`, `#`, `%` or `+` survives, the stat cards say when a tag scopes them, a zero-match tag gets its own empty state, a context-menu paste re-runs the search, and an OPML import summary is rendered instead of discarded. Topic search covers descriptions as its label always claimed, matches literally rather than treating `%` and `_` as wildcards, and is insensitive to case and Unicode normalization
- Opening a topic page no longer mutates it. The acknowledgement that clears the "new info" badge is a separate idempotent POST fired after the page has actually rendered, and it is a no-op if a newer check has landed. Check Now marks a row just-checked when the check really completes rather than the moment it is queued, a failed mutation offers Reload instead of replaying an unsafe verb whose outcome is unknown, a failed knowledge diff can be retried, the sources-failing callout no longer disappears onto page 2 of the history, and the empty-articles hint names a control the topic's state actually shows
- Setup and Settings accept the configurations they document. The keyless local provider path completes setup, switching from Ollama to a cloud model clears only the URL the helper injected, enabling Exa requires a key, a fractional importance threshold is rejected rather than truncated, a cleared per-topic interval stays cleared, a save built on a stale config generation is refused with a reload message, concurrent first-run submissions cannot both publish, and a symlinked `config.yml` is written through rather than replaced
- Correlation ids follow the work. A request id and a check id are emitted as separate fields, one cycle id spans a scheduler tick and everything it launches, the id survives unhandled-exception handling and reaches the 500 response header, the first scheduler wakeup no longer inherits the setup request's id forever, and startup and LiteLLM lines render as JSON in JSON mode
- OPML import works and stays honest. Every real upload was rejected as "No file selected" because the route type-checked the wrong `UploadFile`; outlines merge into one topic only when they carry Topic Watch's own group marker, so unrelated third-party feeds sharing display text stop collapsing into one fan-out; imported names are length-capped and trimmed; overlapping imports re-check for duplicates instead of racing to a 500; and exports state their totals, row caps and per-section truncation. Topic forms hold their own contracts alongside it: a MANUAL topic needs at least one feed URL, blank names are rejected and stored names trimmed, renaming onto an existing name re-renders the form with a duplicate-name error instead of a 500, Enable/Disable submits an explicit target state so a replayed POST cannot re-enable monitoring, page numbers are clamped instead of overflowing into a false "no history", tags are normalized and deduplicated on write and read, and check-history pagination is pinned to a stable cutoff so a check landing mid-traversal cannot repeat or skip a row
- Migrations are atomic and backups are WAL-safe. Each migration body commits with its schema-version row, the ledger must be an exact contiguous prefix of the registered migrations, and `backup_database()` uses SQLite's online backup API and verifies the result with an integrity check instead of copying a file whose WAL commits were missing
- The install and update scripts fail loudly and read their own settings. All three reported success, launched the browser and installed autostart after the post-start health check never passed; `update.sh` probed the invoking process's environment for a port `install.sh` had persisted to `.env`; the entrypoint could recursively chown `/` for a relocated database path, and skipped repairing a root-owned `config.yml` whenever the directory's own owner looked right; and the documented Ollama override file was never downloaded
- The CLI and JSON API report failures instead of silence. Every command resolves the configured database path, `check` and `check-all` print stage errors, delivery failures and skips and exit non-zero, `init` delegates to the canonical initialization rather than a second copy of the workflow, doctor reports only feeds a topic still monitors, topic names render with control and bidi characters escaped, and the trigger endpoint returns the recorded stage error, delivery status and disposition rather than a response identical to a clean quiet check
- The required `pre-commit.ci - pr` check no longer fails on every pull request. The mypy hook is `language: system` and needs the project's dev environment on PATH; pre-commit.ci's sandbox has none, so since 5f2f5d0 it aborted with "Executable `mypy` not found" and left branch protection blocking auto-merge on all open PRs. The hook is now in the `ci.skip` list, which is pre-commit.ci's own escape hatch for system hooks. Type checking is unchanged: the hosted `lint` job, `make lint` and the local hook all still run the same `mypy app/ evals/` invocation

### Security

- The CSRF cookie is HMAC-signed and cross-site submissions are refused outright. A sibling site under the same registrable domain could plant a parent-domain cookie it already knew and then submit the matching form token. An unsigned cookie from an open session is adopted as the payload of its signed replacement, so mid-session requests keep working
- Requests carrying an untrusted `Host` header are rejected, closing the browser DNS-rebinding path into a console with no authentication. `localhost`, `*.localhost`, `*.local` and any IP literal stay accepted, so the stock install and the documented LAN setup are unaffected; `TOPIC_WATCH_ALLOWED_HOSTS` lists the hostname a reverse proxy forwards
- Outbound fetching is bounded and credential-safe. Every hop is streamed under a hard response-size budget, stacked `Content-Encoding` is refused and decoding is bounded on both the wire and the decoded side, redirects are never followed inside httpx and never carry `Authorization`, `Cookie` or `Host` across origins, IP literals are classified with `ipaddress` instead of prefix regexes that also matched hostnames like `127.example.com`, and DNS work runs in one process-wide resolver pool with admission control instead of leaking abandoned lookup threads. The configured LLM base URL, the Exa endpoint and the generic-HTTP Apprise notifiers all pass through that same gate, which requires HTTPS unless the host genuinely resolved private — so a single DNS blip can no longer let a public endpoint be reached in cleartext for the life of the process
- Secrets stop reaching logs, reports and pages. Feed URLs are redacted on every failure path rather than only the blocked-private one, non-HTTP notifier schemes are reduced to a provider tag and a fingerprint because their capability material lives in the authority, Apprise's own logger is silenced at the boundary, a raised private-redirect error redacts its own message text, `doctor` prints only an exception type rather than YAML or Pydantic snippets that can carry a secret, saved delivery URLs are masked on the Settings page, a submitted API key is never echoed back into a re-rendered form, and the setup and settings pages are served `Cache-Control: no-store`
- Model output is treated as untrusted throughout. All four prompts carry the untrusted-data rule and nonce-fence every model-derived input, forged prompt framing is neutralized after any line separator rather than only LF, every format character is stripped before the framing guard, and a markdown link in a knowledge summary renders as inert text instead of a clickable phishing destination inside the app's own origin
- State on disk is owner-only: the database, its WAL and SHM sidecars, the backups directory and every backup file, with startup tightening an older install. An inbound `X-Request-ID` outside 1-128 identifier characters is replaced rather than trusted, copied into every log record and echoed back

## [1.4.0] - 2026-08-19

### Added

- Live topic status: dashboard rows and the topic detail page now update themselves while a topic is new or researching. A row polls `GET /topics/{id}/row` (every 3s researching, 30s new) and the endpoint answers 204 while the status is unchanged, so htmx skips the swap and checkbox/focus state survives; the re-rendered ready/error row carries no poll attributes and therefore stops itself. The detail-page status poll carries `?since=`, so the endpoint can answer `HX-Refresh` on a transition and let the `h1` badge, the actions row and the error alert outside `#status-area` catch up
- The install scripts ask how to set up: who should reach Topic Watch (loopback vs LAN, with a warning on LAN), whether to start at boot (recommended — without it monitoring stops after a reboot and nothing says so), and for a port only when the default is taken. Answers are persisted to `.env`, which the installer upserts, so they survive a later `up -d` or a re-run; each question is skipped when its env var is already set. `install.ps1` gains the same questions and the `.env` writing it never had

### Changed

- Both compose files publish the web port on `${TOPIC_WATCH_BIND_ADDR:-127.0.0.1}` instead of every interface, matching what SECURITY.md already told users to do — Topic Watch has no authentication, and Docker inserts its port rules ahead of ufw/firewalld, so a host firewall does not contain it. Set `TOPIC_WATCH_BIND_ADDR=0.0.0.0` in `.env` for LAN access. Existing installs are unaffected: their `docker-compose.yml` is a copy made at install time and is not rewritten by `docker compose pull`
- `TZ` and `TOPIC_WATCH_PORT` are honored by both compose files; `.env.example` documented both as having no effect. The commented env examples used mapping syntax inside a list, so uncommenting them verbatim failed the YAML parse
- `make lock` and `make lock-upgrade` are runnable: `pip-compile` was never installed by anything, so the command SECURITY.md tells users to run died with "pip-compile: command not found". pip-tools stays out of the dev extras on purpose (locking it would hash-pin pip itself into the `requirements-dev.txt` that `make dev` and all three CI jobs install, and pip 26 breaks pip-tools 7.5.3); instead it gets its own `make lock-tools` target with the known-good pins, and both lock targets fail with that pointer rather than a bare command-not-found. SECURITY.md now points at `lock-upgrade`, since plain `lock` cannot raise versions
- Dependabot and pre-commit.ci PRs enable auto-merge, which then waits on the required status checks from branch protection. pip semver-major bumps are excluded: the mocked-LLM suite cannot catch litellm/instructor behavior changes
- README gains a docker pulls badge. GitHub exposes no API for container download counts, so a daily workflow scrapes the public package page and publishes a shields.io endpoint JSON to an orphan `badges` branch
- Dependencies relocked (litellm 1.96.0, starlette 1.6.0, pydantic-settings 2.15.0 and 8 more); ruff 0.16.3 and mypy 2.3.1 in pre-commit; Python base image digest bumped

### Fixed

- The install scripts' setup prompt was gated on `[ -t 0 ]`, which is false whenever the script is piped — so the documented `curl … | bash` path never asked, and every Linux/macOS user silently got no autostart, leaving monitoring stopped after the next reboot. The scripts probe `/dev/tty` instead, which is the controlling terminal regardless of how stdin is wired. The systemd unit also resolves the real `docker` path instead of assuming `/usr/bin`, which left the unit failing at boot on some distros
- `feed_backoff_base_minutes` and `feed_backoff_cap_hours` are written back to `data/config.yml` on save. Both are real `Settings` fields, used by the checker and the feed-health pages and documented as user-settable, but neither was in the write dict — so any save from `/setup` or `/settings` silently reverted the user's values to 15 minutes and 24 hours. A guard test asserts every scalar `Settings` field is written, so a field added to `Settings` but forgotten here fails the suite instead of quietly resetting itself
- README install commands that failed as written: the Ollama override `cp` only works in a git clone (a `curl` is given instead), `docker compose pull` errors on a source install that has `build: .` and no image (the update instructions are split by install path), building from source was missing the `PUID`/`PGID` step the prebuilt path already has (without it the container chowns the user's own checkout to UID 1000), `.env` is written under umask 077 to match the installer, and the no-Docker uvicorn command binds 127.0.0.1 rather than 0.0.0.0
- A failed migration no longer logs "DB restored from backup at %s". Nothing restores anything — `_backup_db` only copies — and the state it described was wrong in the other direction too: migrations commit per version and the connection is rolled back on the way out, so the DB is already left at the last applied version and needs no restore. It also stops printing "at None" when no backup was taken, which happens whenever `run_migrations` gets a `db_path` that does not exist

## [1.3.0] - 2026-08-05

### Added

- Knowledge history: every knowledge-state write is now recorded as a revision, with an inline diff timeline on the topic detail page showing exactly what the AI added or removed. Retention is capped per topic by the new config-only `knowledge_revision_limit` setting (default 50); revisions are not included in the JSON/CSV export
- Silence Heartbeat: after `silence_heartbeat_checks` consecutive checks where no source returned usable results, topic_watch sends one "sources failing" alert per affected topic, and one "sources recovered" notice when they come back. A failure shared by every topic (expired API key, no network) therefore produces one alert per topic in the same cycle; the message points at the shared cause. The dashboard and topic detail show a failing-sources badge from the first such check
- Checks where no source was even attempted (every feed in backoff, Exa disabled, no feed URLs) now record a `sources_unavailable` stage error instead of looking like healthy silence

## [1.2.5] - 2026-08-01

### Added

- Per-topic novelty instruction: a free-text field (max 500 chars) on add/edit that tells the model what counts as new for that topic, injected into the novelty prompt as user-defined criteria
- Importance scoring: every novelty result now carries a 1-5 `importance` rating, shown in the notification body, the webhook payload (new `importance` key), and the topic's check history
- Optional per-topic importance threshold: findings the model rates below it still update the knowledge state but do not notify, so a minor fact is neither delivered nor re-flagged as new next cycle. Blank (the default) notifies on any importance
- Topic detail page shows the topic's novelty instruction and each check's importance, and labels a check whose notification was suppressed by the importance gate

### Changed

- The analysis log warns when the LLM omits `importance` from its response — that silently defaults the score to 3, which would make a threshold of 4 or 5 mute the topic
- Frozen eval scenarios carry `novelty_instruction`, so a replayed scenario builds the same prompt production sent; `Expectation` gains `min_importance`
- Dependencies relocked; CI actions, ruff (v0.16.0), mypy (v2.3.0), and the Python base image digest bumped

## [1.2.4] - 2026-07-07

### Added

- Exa feed-source fetch outcomes (success and every failure reason) are now recorded in Feed Health, matching RSS sources (#53)

### Fixed

- DeepSeek reasoning models (`deepseek-reasoner`, "thinking mode") no longer fail structured analysis: the LLM client falls back TOOLS → JSON → MD_JSON when a provider rejects instructor's forced `tool_choice` with a 400, while still re-raising mode-invariant errors (context-window, `max_tokens`) and preserving rate-limit backoff (#53)

## [1.2.3] - 2026-07-05

### Added

- Exa AI search as a per-topic feed source: `ExaSettings` (env/YAML), a Settings card for the API key, per-topic source selection in add/edit, and pre-extracted content on `FeedEntry` (Exa returns article text directly, skipping a fetch)
- `check`/`init`/`doctor` surface a clear error when every feed source for a topic fails, instead of reporting no new content

### Fixed

- LLM output-token cap is now a hard ceiling, preventing Anthropic non-streaming 400s on large completions (#53)
- Permanent 4xx LLM errors surface the real provider message instead of a masked generic failure

## [1.2.2] - 2026-07-03

### Fixed

- Custom LLM `base_url` is now honored for every provider, so OpenAI-compatible gateways (e.g. OpenCode Go, a LiteLLM proxy) work via the `openai/<model>` prefix plus `base_url`. Previously `base_url` was silently dropped for cloud providers, which sent requests to the provider's default endpoint and failed with the gateway's key/model (#51)

### Changed

- Base Docker image updated to `python:3.13-slim`; Python dependency lockfiles refreshed

## [1.2.1] - 2026-07-01

### Added

- Live-refresh feed source status on the topic page: the Feed Source badges now poll every 30s (`GET /topics/{id}/feed-source`) instead of updating only when you leave and re-open the page

### Fixed

- Dashboard stat cards (Active / Total / Checks / New-info) no longer show stale counts after create, delete, toggle-active, OPML import, or a check (removed an un-invalidated 60s stats cache)
- Strip leaked `[STUB]`/`[NO CONTENT]` reliability notes and "Note on Data Quality" blocks from analysis summaries; a one-time migration scrubs already-stored `knowledge_states` summaries
- The "new info" badge now clears once you open a topic (previously it persisted until a later check happened to find nothing new)
- Setup wizard offers a "Save anyway (skip the live check)" path, so a transient provider error or a stale default model string no longer dead-ends a new user at `/setup`
- Comment out the LLM lines in `.env.example` (Docker Compose never injects `.env`, so a key set there was silently dead) and correct install-script comments that claimed `.env` holds the LLM key
- Install scripts fail fast on a failed `docker compose pull` with a GHCR-visibility/network hint instead of leaving a half-install

### Security

- Never deliver an unedited placeholder notification URL (e.g. `ntfy://your-topic-name`, which would post to public ntfy.sh); the guard blocks known placeholder tokens before send and also covers the CLI path
- Ship an empty `notifications.urls` in `config.example.yml` so the auto-copied first-run default no longer carries a live `ntfy://your-topic-name` placeholder that would deliver to public ntfy.sh
- Replace the leaky `ntfy://your-topic-name` example in the setup wizard's notification hint with a non-deliverable placeholder

## [1.2.0] - 2026-06-30

### Added

- Per-topic confidence and relevance thresholds (override global defaults; high-stakes topics can demand stricter novelty)
- Per-check LLM token cost shown in topic check history
- Topic initialization reaches READY on the first pass even when early articles are thin or off-topic, storing an explanatory baseline summary that self-heals as fuller coverage arrives on later checks
- Setup wizard pre-flight LLM credential validation (pings the model on submit; bad key/model is caught before completing setup)
- Persistent webhook retry queue that survives restarts and respects `max_retries`
- Knowledge compression that condenses over-budget knowledge via the LLM instead of truncating, preventing dropped facts from being re-detected as new
- Full settings UI that surfaces and persists all configurable fields (previously some were silently reset on save)
- Friendly empty-states, clearer error copy, and accessibility labels across the UI
- Docker PUID/PGID support for non-1000 host users (correct data-volume permissions)
- Dropped-duplicate count surfaced instead of silently discarding duplicate articles
- `apprise_timeout_seconds` to bound notification send time
- Relevance score included in notifications and the webhook JSON payload (`relevance` field)
- `topic-watch` console entry point (run the CLI as `topic-watch ...` after install)

### Changed

- Scheduler holds a database connection only per topic check (not across the whole tick); weekly VACUUM runs off the event loop
- Apprise sends are time-bounded so a hung notification can no longer freeze the scheduler
- Feed/LLM timeout config fields now reject zero or negative values
- `llm_max_retries` now governs rate-limit backoff on LLM calls (previously hardcoded)
- Dependency updates

### Fixed

- Settings POST handler no longer silently resets `min_relevance_threshold`, `secure_cookies`, and other fields on save
- Stopped a re-analysis loop and corrected `status_changed_at` handling
- Dashboard now surfaces `?error=` flash messages
- RSS provider fallback: continue past single feed failures, fall back to a second provider when all are unhealthy, and don't mark a provider unhealthy on an empty-but-OK feed
- OPML import merges feeds for same-named outlines
- Config writer creates the parent directory before writing the YAML
- Install script writes/updates PUID/PGID in `.env` without truncating existing contents
- Hardened model parsing against malformed JSON and empty-string/corrupt datetimes
- Docker entrypoint guards `chown`, validates PUID/PGID, and warns when run as root
- Docker entrypoint chowns `data/` when only the GID differs, not just the UID
- Groq, DeepSeek, Mistral, xAI, and Perplexity now recognized as cloud providers, so a stale `base_url` is dropped on provider switch instead of misrouting the call
- Init-timeout error transitions now stamp `status_changed_at`
- Removed a broken notification icon reference in the web UI

### Security

- Reject non-http(s) redirect schemes during URL fetches
- Re-validate redirect targets against SSRF on every hop
- SSRF check now fails closed on DNS resolution failure (unresolvable hosts are treated as private and blocked)
- SSRF check now blocks CGNAT (`100.64.0.0/10`) and the full IPv6 ULA range (`fc00::/7`)
- Bump python-multipart and pytest to patch known CVEs
- Exclude the secret-bearing `.env` from the Docker build context (`.dockerignore`)
- Install scripts document the `curl | bash` / `irm | iex` mutable-`main` risk and support pinning a tag/commit via `TOPIC_WATCH_REF`
- Boot/login autostart is now opt-in in the install scripts (prompt or `TOPIC_WATCH_AUTOSTART`), with documented uninstall steps

## [1.1.2] - 2026-04-04

### Fixed

- Fix failing test for empty dashboard message (test expected old wording)

## [1.1.1] - 2026-04-04

### Added

- Theme showcase GIF in README
- Contributor Covenant v3.0 Code of Conduct
- "Updating" section in README with upgrade instructions
- `--version` flag for the CLI
- GitHub Discussions enabled

### Fixed

- Generic 404/422 pages now render styled HTML instead of raw JSON for browser requests
- OpenAPI version synced with app version
- Version display now reads from pyproject.toml (single source of truth)
- Feed Health table column truncation on desktop
- Auto-mode topics now show all feed URLs on detail page
- Readable source names in articles table instead of raw feed URLs
- Page header alignment and footer positioning
- Button vertical alignment in action rows

## [1.1.0] - 2026-04-04

### Added

- OPML import/export for migrating feeds from RSS readers (FreshRSS, Miniflux, Tiny Tiny RSS)
- JSON API at `/api/v1/` for scripting and monitoring (topics, checks, knowledge state, trigger)
- Dashboard stats bar (total checks, new info found, last notification)
- Dark mode auto-detection via `prefers-color-scheme` media query
- Ollama quick start with `docker-compose.override.example.yml`
- `TopicStatus.NEW` for gradual OPML import initialization (~1 topic/min)
- Multi-provider RSS fallback (Bing News + Google News)
- Human-readable check intervals (`6h`, `1w 3d`, `2h 30m`) replacing integer hours. Range: 10 minutes to 6 months
- LLM confidence and relevance thresholds to reduce false notifications (`min_confidence_threshold`, `min_relevance_threshold`)
- Configurable LLM temperature (`llm_temperature`)
- Semantic status colors and UI design polish (table scroll, danger button)
- Docker image `latest` tag on GitHub releases (previously missing)
- Automatic cleanup of old untagged container images

### Changed

- Check interval config field renamed from `check_interval_hours` (integer) to `check_interval` (human-readable string). Old format auto-migrated.
- LLM prompts improved for more conservative novelty detection and better scope filtering
- Article truncation in prompts increased from 1000 to 1500 chars

### Fixed

- SSRF bypass via IPv6 and alternative IP encodings
- LLM novelty detection accuracy (reasoning field, relevance scoring, below-threshold article re-examination)
- Docker install script failing with `unauthorized` (GHCR package visibility + missing `latest` tag)

## [1.0.0] - 2026-03-20

### Added

- Topic monitoring with configurable RSS feeds
- LLM-powered novelty detection via LiteLLM + Instructor (structured Pydantic output)
- Knowledge state management with token budget and automatic compression
- Web dashboard with HTMX for topic management, search, and bulk operations
- Per-topic check intervals and feed health monitoring
- Apprise integration supporting 100+ notification services
- Webhook support with JSON payloads
- Notification retry queue for failed deliveries
- CLI for manual operations (`list`, `check`, `check-all`, `init`)
- Settings UI for in-app configuration
- Custom color themes (Nord, Dracula, Solarized Dark, High Contrast, Tokyo Night)
- CSRF protection on all mutation endpoints
- SSRF protection blocking private/internal network ranges on article fetches
- XSS protection with input sanitization on all user-facing outputs
- Export filename sanitization preventing header injection
- Rate limiting on API endpoints with automatic cleanup
- Docker multi-stage build with non-root user, HEALTHCHECK, and STOPSIGNAL
- Docker Compose resource limits (512M memory) and log rotation
- Auto-copy config on first run with clear setup instructions
- Configurable log level via `TOPIC_WATCH_LOG_LEVEL` environment variable
- Version display in web UI footer
- Ruff security lint rules (bandit) in CI
- CI testing on Python 3.11, 3.12, and 3.13
- Dependabot for automated dependency updates
- Reverse proxy examples (Caddy, Nginx) in README
- Comprehensive test suite (92% coverage)

[1.0.0]: https://github.com/0xzerolight/topic_watch/releases/tag/v1.0.0
