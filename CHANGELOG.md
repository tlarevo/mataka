# Changelog

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
