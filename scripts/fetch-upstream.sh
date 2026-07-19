#!/usr/bin/env bash
# Vendor the upstream Hindsight reference at the pinned commit.
# Idempotent: skips if already checked out at the correct SHA.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PIN_FILE="$REPO_ROOT/contract/UPSTREAM_PIN"
UPSTREAM_DIR="$REPO_ROOT/upstream/hindsight"

if [ ! -f "$PIN_FILE" ]; then
  echo "ERROR: $PIN_FILE not found" >&2
  exit 1
fi

PIN_SHA=$(head -1 "$PIN_FILE" | awk '{print $1}')
PIN_DATE=$(head -1 "$PIN_FILE" | awk '{print $2}')
PIN_REASON=$(head -1 "$PIN_FILE" | cut -d' ' -f3-)

echo "Upstream pin: $PIN_SHA ($PIN_DATE) — $PIN_REASON"

if [ -d "$UPSTREAM_DIR/.git" ]; then
  CURRENT_SHA=$(git -C "$UPSTREAM_DIR" rev-parse HEAD 2>/dev/null || echo "")
  if [ "$CURRENT_SHA" = "$PIN_SHA" ]; then
    echo "Upstream already at pinned SHA $PIN_SHA — skipping clone."
    exit 0
  fi
  echo "Updating upstream from $CURRENT_SHA to $PIN_SHA..."
  git -C "$UPSTREAM_DIR" fetch --depth=1 origin "$PIN_SHA"
  git -C "$UPSTREAM_DIR" checkout FETCH_HEAD
else
  echo "Cloning upstream Hindsight (shallow to pinned SHA)..."
  mkdir -p "$(dirname "$UPSTREAM_DIR")"
  git clone --depth=1 --branch "" 2>/dev/null || \
    git clone https://github.com/vectorize-io/hindsight.git "$UPSTREAM_DIR"
  git -C "$UPSTREAM_DIR" fetch --depth=1 origin "$PIN_SHA"
  git -C "$UPSTREAM_DIR" checkout FETCH_HEAD
fi

echo "Upstream vendored at $UPSTREAM_DIR @ $PIN_SHA"

# Verify referenced files exist
echo "Verifying referenced upstream paths..."
MISSING=0
for f in \
  "hindsight-api-slim/hindsight_api/engine/retain/fact_extraction.py" \
  "hindsight-api-slim/hindsight_api/engine/consolidation/prompts.py" \
  "hindsight-api-slim/hindsight_api/engine/consolidation/consolidator.py" \
  "hindsight-api-slim/hindsight_api/engine/graph_maintenance.py"; do
  if [ ! -f "$UPSTREAM_DIR/$f" ]; then
    echo "  MISSING: $f" >&2
    MISSING=1
  else
    echo "  OK: $f"
  fi
done

if [ "$MISSING" -eq 1 ]; then
  echo "ERROR: some referenced upstream files are missing at pin $PIN_SHA" >&2
  exit 1
fi

echo "Done."
