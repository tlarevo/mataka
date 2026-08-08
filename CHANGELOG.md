# Changelog

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
