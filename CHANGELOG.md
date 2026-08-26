# Changelog

## v0.2.4

### Fixes
- **Consolidation follow-on no longer pegs the local LLM server.** Every async
  `retain` chained an unbounded full-bank consolidation pass. On a real ~9.3k-fact
  bank that is up to ~786 fresh chat calls per retain — group membership drifts as
  facts accumulate, so the old exact-source-set idempotency check never converged
  (21.7k observations existed from only 2.4k distinct source sets). With OMP firing
  retains per turn, this kept `llama-server` at 70-80% CPU for entire sessions
  (2026-08-26 incident). Now:
  - The retain follow-on is **scoped** to groups containing the just-stored facts
    and **capped** at 8 LLM merge calls (`consolidate_scoped` +
    `FOLLOW_ON_MAX_LLM_CALLS`). The explicit `POST /consolidate` endpoint keeps the
    unbounded full-bank pass for manual runs.
  - Idempotency is now coverage-based: a group is skipped when every member fact is
    already an observation source — converges even when group membership drifts.
  - The LLM's returned `source_ids` are filtered to the group's real fact IDs.
    Hallucinated IDs were the recurring `consolidation follow-on failed:
    FOREIGN KEY constraint failed` aborts in the log.
  - Observation + provenance rows are inserted in a single transaction (no more
    observations without provenance).

## v0.2.3

### Features
- `/health` now reports real LLM reachability instead of a static `{"status":"ok"}`.
  It probes the embeddings path (shared by retain and recall) with a short timeout,
  so a wedged generation backend — the 2026-08-08 Ollama incident where the process
  was up, `/api/tags` returned 200, and every `/v1/*` call 503'd for 4+ days — is
  visible via `GET /health`. The endpoint stays HTTP 200 with `status: "ok"` for
  existing liveness checks and adds:
  - `llm.reachable: true|false`
  - `llm.latency_ms` (when reachable)
  - `llm.error` (when unreachable)
  The probe bypasses the concurrency semaphore and uses a 3s timeout so health
  checks never block real work.

## v0.2.2

### Fixes
- `chunk_text`/`find_split_point` could panic on inputs where the chunk boundary landed mid
  multi-byte UTF-8 character (em dashes, emoji, checkmarks) — a real, repeatable production
  panic on any retain content containing non-ASCII characters near the 8000-char chunk edge.
- `LlmClient` requests had no timeout and no bounded retry: a wedged or momentarily-busy LLM
  backend (HTTP 503) could hang indefinitely or fail hard on transient blips. Added a
  configurable timeout (`MATAKA_LLM_TIMEOUT_SECS`, default 120s), bounded exponential-backoff
  retry on 503/timeout (`MAX_ATTEMPTS=4`), and a concurrency cap
  (`MATAKA_LLM_MAX_CONCURRENT`, default 4) so a burst can't queue unbounded requests against a
  local single-model server.

### Features
- `LlmClient` chat and embeddings can now point at independently configured backends via
  `MATAKA_LLM_CHAT_BASE_URL` (defaults to `MATAKA_LLM_BASE_URL` when unset — no behavior change
  for existing single-provider setups).
- Retain chunk size/overlap are now configurable via `MATAKA_RETAIN_CHUNK_CHARS` /
  `MATAKA_RETAIN_CHUNK_OVERLAP_CHARS` for chat backends with a smaller context window than the
  8000-char default assumes.

### Config
- Default chat model changed from `qwen2.5:7b` to `qwen2.5:3b` — validated to match 7b's
  extraction, reflect, and consolidate quality on a differential eval (2.5x smaller, faster
  retain latency, no observed quality loss on a 36-fixture sample plus a dedicated multi-fact
  synthesis/dedup scenario).


## v0.2.1

### Breaking Changes
- Default port changed from 8888 to 8889 to coexist with Hindsight. Override with `MATAKA_PORT=8888` for drop-in replacement.

### Features
- Import mode for retain endpoint: `"import": true` skips LLM extraction, 100x faster bulk migration from Hindsight.
- Graceful shutdown on SIGTERM/SIGINT.
- Default data dir changed to `~/.local/share/mataka` (absolute path).

### Documentation
- Hindsight migration guide added to README.
- Homebrew install instructions added to README.
- MATAKA_LLM_MODEL config option documented.
- Mermaid architecture diagram added to ARCHITECTURE.md.

## v0.2.0

- Per-bank SQLite sharding + read pool.
- Async operations model for retain endpoint.
- All five `tags_match` filter semantics for recall.
- tiktoken-rs token counting (replaces chars/4 estimator).
- Memory vetting system: secret/PII detection, ingress guard, at-rest scanner, egress guard.
- Facts→observations consolidation loop.
- Import mode for bulk memory migration.
- Differential test harness (mataka vs Hindsight).
- CI pipeline with release workflow.
