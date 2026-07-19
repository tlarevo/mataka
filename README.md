# mataka

Lean, single-binary drop-in replacement for the [Hindsight](https://github.com/vectorize-io/hindsight)
agent-memory dataplane API. ~18 MB RSS instead of multi-GB.

```
HINDSIGHT_API_LLM_PROVIDER=openai-compatible \
HINDSIGHT_API_LLM_BASE_URL=http://localhost:11434/v1 \
HINDSIGHT_API_LLM_MODEL=qwen2.5:7b \
HINDSIGHT_API_EMBEDDINGS_MODEL=nomic-embed-text \
./mataka   # listens on :8888, SQLite at ./mataka.db
```

Config: every knob accepts either the native `MATAKA_*` var or the `HINDSIGHT_API_*` equivalent (MATAKA_ wins if both set): `MATAKA_LLM_PROVIDER`/`HINDSIGHT_API_LLM_PROVIDER`, `..._LLM_BASE_URL`, `..._LLM_API_KEY`, `..._LLM_MODEL`, `MATAKA_EMBEDDINGS_MODEL`/`HINDSIGHT_API_EMBEDDINGS_MODEL`, `MATAKA_PORT`/`HINDSIGHT_API_PORT`. Use provider `mock` for deterministic offline dev/tests.
See PARITY.md for route coverage and ARCHITECTURE.md for design rationale.
