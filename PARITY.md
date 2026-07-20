# Parity Map — mataka vs Hindsight 0.8.4

Contract pinned at `contract/openapi-0.8.4.json` (56 documented paths; 74 routes in source).
Definition of "drop-in": official Hindsight SDKs (`hindsight-client` py/ts/go/rust) work unmodified
against mataka for the tiers marked done. Ranking parity is defined as recall@k overlap ≥ 0.8
against upstream on a shared fixture corpus, not identical ordering (different embedding model +
BM25 implementation make exact ordering impossible and meaningless).

## Tier 0 — core loop (IMPLEMENTED in MVP)

| Route | Status |
|---|---|
| GET /health, GET /version | done |
| GET /v1/default/banks | done |
| PUT/PATCH/DELETE /v1/default/banks/{bank_id} | done |
| GET /v1/default/banks/{bank_id}/stats | done |
| POST .../memories (retain, single + batch) | done (synchronous; async ops in Tier 1) |
| POST .../memories/recall | done (4-arm RRF; no reranker yet) |
| POST .../reflect | done (single-shot; agentic loop in Tier 2) |
| GET .../memories/list | done |
| GET/DELETE .../memories/{memory_id} | done |
| DELETE .../memories (clear bank) | done |
| GET .../entities | done |

## Tier 1 — operational parity (NEXT)

- Async operations model: retain returns `operation_id`, background worker, GET/DELETE
  .../operations, /operations/{id}, /retry. Tokio task + operations table (already in schema).
- Consolidation worker: facts → observations (dedup, evidence-grounding). Port
  `engine/consolidation/prompts.py` (MIT, 227 lines — small).
- Cross-encoder reranking: feature-gated `ort` with ms-marco-MiniLM-L-6-v2 ONNX (~90 MB, int8 ~25 MB).
- tags / tags_match semantics: any | all | any_strict | all_strict | exact (subtle: strict
  variants exclude untagged; exact is set-equality). Port test cases from upstream.
- Native embeddings: feature-gated `ort` + intfloat/multilingual-e5-small ONNX (upstream's own
  ONNX default), removing the Ollama dependency. ~120 MB.
- GET .../tags, GET .../memories/{id}/history, PATCH .../memories/{id}
- Documents: GET/PATCH/DELETE .../documents*, chunks, reprocess
- Bank config: GET/PATCH/DELETE .../config; profile GET/PUT; import/export

## Tier 2 — full surface

- Mental models CRUD + refresh/clear/history (needs consolidation worker)
- Directives CRUD (guardrails applied in reflect)
- Agentic reflect: tool loop over Mental Models → Observations → Raw Facts priority
- Webhooks CRUD + delivery log
- Graph endpoints (bank graph, entity graph, entity regenerate)
- Audit logs, llm-requests stats, memories-timeseries
- dry-run-extract, files/retain, document-transfer, consolidation/recover, observations scopes

## Explicitly out of scope

- Control plane UI (Next.js) — mataka is headless; upstream UI can point at it if needed later
- Oracle 23ai dialect, Citus, multi-tenant schemas
- OTel/Prometheus exporters (a /metrics stub returns 200)
- MCP server endpoints (worth revisiting — upstream ships fastmcp; a Rust MCP layer is cheap)

## Env-var compatibility

mataka reads the same `HINDSIGHT_API_LLM_PROVIDER`, `HINDSIGHT_API_LLM_API_KEY`,
`HINDSIGHT_API_LLM_MODEL`, `HINDSIGHT_API_LLM_BASE_URL` variables and defaults to port 8889,
so existing docker-compose/env setups carry over.
