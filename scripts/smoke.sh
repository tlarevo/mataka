#!/usr/bin/env bash
# Smoke flow: health → create bank → retain 3 facts → recall → reflect → stats
# Requires: MATAKA_LLM_PROVIDER=mock, server on $PORT (default 8888)
set -euo pipefail

PORT="${MATAKA_PORT:-8888}"
BASE="http://localhost:$PORT"
BANK="smoke-test-$(date +%s)"
PASS=0
FAIL=0

check() {
  local desc="$1" expected="$2" actual="$3"
  if echo "$actual" | grep -q "$expected"; then
    echo "  ✓ $desc"
    PASS=$((PASS + 1))
  else
    echo "  ✗ $desc — expected '$expected' in: $actual"
    FAIL=$((FAIL + 1))
  fi
}

echo "=== Mataka smoke flow (port $PORT, bank $BANK) ==="

# 1. Health
echo "[1/6] Health check..."
RESP=$(curl -sf "$BASE/health")
check "health returns ok" '"status":"ok"' "$(echo "$RESP" | tr -d '[:space:]')"

# 2. Create bank
echo "[2/6] Create bank..."
RESP=$(curl -sf -X PUT "$BASE/v1/default/banks/$BANK" \
  -H 'Content-Type: application/json' \
  -d '{"name":"smoke-test","mission":"Automated smoke verification"}')
check "bank created" '"bank_id"' "$(echo "$RESP" | tr -d '[:space:]')"

# 3. Retain 3 facts from different sources
echo "[3/6] Retain 3 facts (multi-arm sources)..."
for i in 1 2 3; do
  CONTENT="Fact $i: Alice went hiking in the Alps on 2026-03-15. The weather was sunny and she saw marmots."
  TAG="source_$i"
  RESP=$(curl -sf -X POST "$BASE/v1/default/banks/$BANK/memories" \
    -H 'Content-Type: application/json' \
    -d "{\"content\":\"$CONTENT\",\"tags\":[\"$TAG\"],\"metadata\":{\"source\":\"smoke-$i\"}}")
  check "retain fact $i" '"status":"completed"' "$(echo "$RESP" | tr -d '[:space:]')"
done

# 4. Recall
echo "[4/6] Recall..."
RESP=$(curl -sf -X POST "$BASE/v1/default/banks/$BANK/memories/recall" \
  -H 'Content-Type: application/json' \
  -d '{"query":"Alice hiking Alps","max_tokens":500}')
check "recall returns results" '"results"' "$(echo "$RESP" | tr -d '[:space:]')"
check "recall has memories" '"text"' "$(echo "$RESP" | tr -d '[:space:]')"

# 5. Reflect
echo "[5/6] Reflect..."
RESP=$(curl -sf -X POST "$BASE/v1/default/banks/$BANK/reflect" \
  -H 'Content-Type: application/json' \
  -d '{"query":"What did Alice do recently?","budget":"500"}')
check "reflect returns text" '"text"' "$(echo "$RESP" | tr -d '[:space:]')"

# 6. Stats
echo "[6/6] Stats..."
RESP=$(curl -sf "$BASE/v1/default/banks/$BANK/stats")
check "stats returns count" '"total_memories"' "$(echo "$RESP" | tr -d '[:space:]')"

# Cleanup
echo "[cleanup] Delete bank..."
curl -sf -X DELETE "$BASE/v1/default/banks/$BANK" > /dev/null 2>&1 || true

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] || exit 1
