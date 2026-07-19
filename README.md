# mataka

**Lean, single-binary drop-in replacement for the [Hindsight](https://github.com/vectorize-io/hindsight) agent-memory API.**

~18 MB RSS instead of multi-GB. Same API contract, same SDKs, zero config changes.

## Why mataka?

Hindsight is a powerful agent-memory dataplane, but running it locally pulls in PyTorch, embedded PostgreSQL, pgvector, FastAPI, and a Next.js control plane — multi-GB of RAM for what is fundamentally an SQLite-sized workload. mataka replaces the entire stack with a single Rust binary that speaks the same API:

| | Hindsight (local) | mataka |
|---|---|---|
| **RAM** | ~2–4 GB | ~18 MB |
| **Binary** | Docker image (multi-GB) | Single static binary |
| **Database** | Embedded PostgreSQL + pgvector | SQLite (WAL mode) |
| **Embeddings** | PyTorch / sentence-transformers | OpenAI-compatible `/embeddings` |
| **SDK compatibility** | Native | Drop-in (same API contract) |

Official Hindsight SDKs (`pip install hindsight-client`, `npm install hindsight-client`, etc.) work unmodified against mataka.

## Quick start

### From source

```bash
# Clone and build
git clone https://github.com/tlarevo/mataka.git
cd mataka
cargo build --release

# Run with a local LLM (e.g. Ollama)
MATAKA_LLM_PROVIDER=openai-compatible \
MATAKA_LLM_BASE_URL=http://localhost:11434/v1 \
MATAKA_LLM_MODEL=qwen2.5:7b \
MATAKA_EMBEDDINGS_MODEL=nomic-embed-text \
./target/release/mataka
```

### From GitHub releases

Download the binary for your platform from [Releases](https://github.com/tlarevo/mataka/releases), then run:

```bash
MATAKA_LLM_PROVIDER=openai-compatible \
MATAKA_LLM_BASE_URL=http://localhost:11434/v1 \
MATAKA_LLM_MODEL=qwen2.5:7b \
MATAKA_EMBEDDINGS_MODEL=nomic-embed-text \
./mataka
```

### Verify it's running

```bash
curl http://localhost:8888/health
# {"status":"ok"}
```

## Configuration

mataka accepts both native `MATAKA_*` and upstream `HINDSIGHT_API_*` environment variables. `MATAKA_*` wins when both are set.

| Variable | Default | Description |
|---|---|---|
| `MATAKA_LLM_PROVIDER` | `openai-compatible` | LLM provider (`openai-compatible`, `mock`) |
| `MATAKA_LLM_BASE_URL` | — | OpenAI-compatible chat endpoint |
| `MATAKA_LLM_API_KEY` | — | API key (optional for local providers) |
| `MATAKA_LLM_MODEL` | — | Chat model name |
| `MATAKA_EMBEDDINGS_MODEL` | — | Embeddings model name |
| `MATAKA_PORT` | `8888` | Server listen port |
| `MATAKA_DB` | `mataka-data` | Data directory (per-bank sharding) or legacy `.db` file path |
| `MATAKA_VET` | `redact` | Vetting mode: `strict` (reject secrets), `redact` (replace in-place), `off` |

Use `MATAKA_LLM_PROVIDER=mock` for deterministic offline development and testing.

## API endpoints

Full contract: [contract/openapi-0.8.4.json](contract/openapi-0.8.4.json) (56 documented paths; mataka implements the subset below).

### Banks

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/version` | Server version + contract compat |
| `GET` | `/v1/default/banks` | List all banks |
| `PUT` | `/v1/default/banks/{bank_id}` | Create or update a bank |
| `PATCH` | `/v1/default/banks/{bank_id}` | Partial bank update |
| `DELETE` | `/v1/default/banks/{bank_id}` | Delete a bank and its data |
| `GET` | `/v1/default/banks/{bank_id}/stats` | Bank statistics (memory count, etc.) |

### Memories

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `.../memories` | Retain facts (async by default; `"async": false` for sync) |
| `GET` | `.../memories/list` | List memories with pagination |
| `POST` | `.../memories/recall` | Recall memories (4-arm RRF + tag filtering) |
| `GET` | `.../memories/{memory_id}` | Get a single memory |
| `DELETE` | `.../memories/{memory_id}` | Delete a single memory |
| `DELETE` | `.../memories` | Clear all memories in a bank |

### Intelligence

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `.../reflect` | Grounded generation from recalled memories |
| `POST` | `.../consolidate` | Trigger facts→observations deduplication |
| `DELETE` | `.../observations` | Remove all observations (not source facts) |
| `GET` | `.../entities` | List extracted entities |

### Operations & vetting

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `.../operations` | List async operations |
| `GET` | `.../operations/{operation_id}` | Get operation status |
| `DELETE` | `.../operations/{operation_id}` | Cancel a pending/failed operation |
| `POST` | `.../operations/{operation_id}/retry` | Retry a failed operation |
| `POST` | `.../vet` | Scan bank for secrets; `{"fix": true}` to redact |

### Example: create → retain → recall

```bash
# Create a bank
curl -X PUT http://localhost:8888/v1/default/banks/my-project \
  -H 'Content-Type: application/json' \
  -d '{"name": "my-project", "mission": "Agent memory for project X"}'

# Retain facts (async — returns operation_id)
curl -X POST http://localhost:8888/v1/default/banks/my-project/memories \
  -H 'Content-Type: application/json' \
  -d '{"items": [{"content": "Alice deployed the hotfix to production on 2026-07-15", "tags": ["alice"]}]}'

# Synchronous retain (for scripts/tests)
curl -X POST http://localhost:8888/v1/default/banks/my-project/memories \
  -H 'Content-Type: application/json' \
  -d '{"items": [{"content": "..."}], "async": false}'

# Recall with tag filtering
curl -X POST http://localhost:8888/v1/default/banks/my-project/memories/recall \
  -H 'Content-Type: application/json' \
  -d '{"query": "deployments", "tags": ["alice"], "tags_match": "any_strict", "max_tokens": 2000}'

# Reflect (grounded generation from memories)
curl -X POST http://localhost:8888/v1/default/banks/my-project/reflect \
  -H 'Content-Type: application/json' \
  -d '{"query": "Summarize recent deployment activity", "budget": "2048"}'
```

### Tag filtering modes

| `tags_match` | Behavior |
|---|---|
| `any` (default) | OR match, includes untagged units |
| `all` | AND match, includes untagged units |
| `any_strict` | OR match, excludes untagged units |
| `all_strict` | AND match, excludes untagged units |
| `exact` | Set-equality on full tag scope, excludes untagged (empty tags → only untagged) |

## CLI

mataka includes a built-in vetting CLI for scanning and fixing secrets in existing banks:

```bash
# Scan a bank for secrets (dry run)
mataka vet --bank my-project

# Scan and redact all found secrets
mataka vet --bank my-project --fix
```

## Features

### Per-bank sharding

Each bank gets its own SQLite file under the data directory. Concurrent agents writing to different banks never contend. A read pool of 4 connections per bank handles parallel recalls. Legacy single-file mode (`MATAKA_DB=./mataka.db`) is supported with a startup deprecation warning.

### Async operations

Retain runs in the background by default. The response includes an `operation_id` you can poll via `GET .../operations/{operation_id}`. Failed operations can be retried. Use `"async": false` for synchronous behavior.

### Memory vetting

Deterministic secret/PII detection at three stages:

- **Ingress guard** — catches secrets before embedding (AWS keys, GitHub tokens, OpenAI/Anthropic keys, JWTs, private keys, high-entropy tokens). Configurable via `MATAKA_VET`: `strict` (reject), `redact` (replace in-place, default), `off`.
- **At-rest scanner** — `POST .../vet` scans existing memories via API; `mataka vet` CLI for batch scans.
- **Egress guard** — quarantined units excluded from recall/reflect. Belt-and-suspenders: `strict` mode also checks output text.

Redaction uses typed placeholders (`{{REDACTED:aws_access_key}}`) so facts remain useful. Four-copy atomicity: text, embeddings, FTS, and history snapshots are all updated in a single transaction.

### Consolidation

After retain completes, semantically similar facts are automatically merged into deduplicated observations with provenance tracking. Observations are first-class memory units queryable via `types: ["observation"]` in recall. Triggered automatically or on-demand via `POST .../consolidate`.

### Differential test harness

A Python/uv harness for measuring recall parity between mataka and upstream Hindsight:

```bash
cd harness
uv run run_diff.py --mataka http://localhost:8888 --hindsight http://localhost:9888
```

Outputs a markdown report with per-fixture recall@10 overlap, latency, and RSS comparison. Includes ~200 fixture payloads covering conversations, temporal facts, multi-entity sentences, and tag-isolation cases.


## Development

### Prerequisites

- Rust 1.95.0+ (pinned via `rust-toolchain.toml`)
- A mock provider for offline testing: `MATAKA_LLM_PROVIDER=mock`

### Build & test

```bash
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt -- --check
```

### Smoke test

```bash
MATAKA_LLM_PROVIDER=mock ./target/debug/mataka &
bash scripts/smoke.sh
```

### Upstream reference

Vendor the upstream Hindsight source for prompt porting and parity work:

```bash
bash scripts/fetch-upstream.sh
# Creates ./upstream/hindsight/ at the pinned SHA (contract/openapi-0.8.4.json)
```

### Concurrency test

```bash
MATAKA_LLM_PROVIDER=mock ./target/debug/mataka &
bash scripts/concurrency_test.sh
```

## Roadmap

| Milestone | Status | Description |
|---|---|---|
| **v0.1 — Parity core** | ✅ Done | Async ops, tag filtering, token counting, extraction prompts, per-bank sharding |
| **v0.2 — omp daily driver** | ✅ Done | Consolidation worker, differential test harness, memory vetting |
| v0.3 — TEMPR parity | Planned | Weighted memory edges, spreading activation, cross-encoder reranker, ONNX embeddings |
| v0.4 — Adoption surfaces | Planned | Bank config/import/export, markdown bridge, MCP server, apfel on-device provider |
| v0.5 — Management & safety | Planned | TUI (ratatui), operations monitor, memory browser, recall playground |
| v1.0 — CARA & beyond | Planned | Opinion network, preference-conditioned reflect, bank disposition profiles |

See [PARITY.md](PARITY.md) for the detailed route-by-route compatibility map.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design rationale, upstream analysis, and recall pipeline internals.

## License

[MIT](LICENSE) — incorporates code from [Hindsight](https://github.com/vectorize-io/hindsight) (MIT) with attribution in source files.
