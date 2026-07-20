#!/usr/bin/env bash
# Concurrency tests for per-bank sharding + read pool (THA-137)
# Requires: MATAKA_LLM_PROVIDER=mock, server on $PORT (default 8889)
set -euo pipefail

PORT="${MATAKA_PORT:-8889}"
BASE="http://localhost:$PORT"
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

time_ms() {
  local start end
  start=$(python3 -c 'import time; print(int(time.time()*1000))')
  eval "$@" > /dev/null 2>&1
  end=$(python3 -c 'import time; print(int(time.time()*1000))')
  echo $((end - start))
}

echo "=== Mataka concurrency tests (port $PORT) ==="

# ─── Test 1: Cross-bank write parallelism ──────────────────────────────
echo ""
echo "[Test 1] Cross-bank write parallelism (4 banks × 10 iterations)..."

retain_loop() {
  local bank="$1" n="$2"
  for i in $(seq 1 $n); do
    curl -sf -X POST "$BASE/v1/default/banks/$bank/memories" \
      -H 'Content-Type: application/json' \
      -d "{\"items\":[{\"content\":\"Fact $i from $bank: Alice hiked in the Alps.\",\"tags\":[\"test\"]}]}" > /dev/null
  done
}

for i in 1 2 3 4; do
  curl -sf -X PUT "$BASE/v1/default/banks/conc-$i" \
    -H 'Content-Type: application/json' \
    -d "{\"name\":\"conc-$i\"}" > /dev/null
done

T_SINGLE=$(time_ms 'retain_loop "conc-1" 10')

T_PARALLEL=$(time_ms '
  retain_loop "conc-1" 10 &
  retain_loop "conc-2" 10 &
  retain_loop "conc-3" 10 &
  retain_loop "conc-4" 10 &
  wait
')

RATIO=$(python3 -c "print(f'{$T_PARALLEL / $T_SINGLE:.2f}')")
echo "  Single: ${T_SINGLE}ms, 4 parallel: ${T_PARALLEL}ms, ratio: ${RATIO}x"
if python3 -c "exit(0 if $T_PARALLEL < $T_SINGLE * 2.0 else 1)"; then
  check "parallel writes within 2× single" "ok" "ok"
else
  echo "  ✗ parallel writes too slow (ratio ${RATIO}x)"
  FAIL=$((FAIL + 1))
fi

for i in 1 2 3 4; do
  curl -sf -X DELETE "$BASE/v1/default/banks/conc-$i" > /dev/null 2>&1 || true
done

# ─── Test 2: Read pool — parallel recalls against ONE bank ─────────────
echo ""
echo "[Test 2] Read pool concurrency (4 parallel recalls)..."

curl -sf -X PUT "$BASE/v1/default/banks/readpool" \
  -H 'Content-Type: application/json' -d '{"name":"readpool"}' > /dev/null
for i in 1 2 3; do
  curl -sf -X POST "$BASE/v1/default/banks/readpool/memories" \
    -H 'Content-Type: application/json' \
    -d "{\"items\":[{\"content\":\"Fact $i: Alice visited Tokyo.\",\"tags\":[\"travel\"]}]}" > /dev/null
done

T_1=$(time_ms 'curl -sf -X POST "$BASE/v1/default/banks/readpool/memories/recall" -H "Content-Type: application/json" -d "{\"query\":\"Alice Tokyo\",\"max_tokens\":200}" > /dev/null')

T_4=$(time_ms '
  for i in 1 2 3 4; do
    curl -sf -X POST "$BASE/v1/default/banks/readpool/memories/recall" \
      -H "Content-Type: application/json" \
      -d "{\"query\":\"Alice Tokyo\",\"max_tokens\":200}" > /dev/null &
  done
  wait
')

RR=$(python3 -c "print(f'{$T_4 / $T_1:.2f}')")
echo "  1 recall: ${T_1}ms, 4 parallel: ${T_4}ms, ratio: ${RR}x"
if python3 -c "exit(0 if $T_4 < $T_1 * 3.0 else 1)"; then
  check "4 parallel recalls < 3× single (read pool)" "ok" "ok"
else
  echo "  ✗ reads serialized (ratio ${RR}x)"
  FAIL=$((FAIL + 1))
fi

curl -sf -X DELETE "$BASE/v1/default/banks/readpool" > /dev/null 2>&1 || true

# ─── Test 3: Bank DELETE removes files and is no longer listed ──────────
echo ""
echo "[Test 3] Bank DELETE removes files..."

curl -sf -X PUT "$BASE/v1/default/banks/del-test" \
  -H 'Content-Type: application/json' -d '{"name":"del-test"}' > /dev/null

RESP=$(curl -sf "$BASE/v1/default/banks")
check "bank listed before delete" "del-test" "$RESP"

curl -sf -X DELETE "$BASE/v1/default/banks/del-test" > /dev/null

RESP=$(curl -sf "$BASE/v1/default/banks")
if echo "$RESP" | grep -q "del-test"; then
  echo "  ✗ bank still listed after delete"
  FAIL=$((FAIL + 1))
else
  check "bank not listed after delete" "ok" "ok"
fi

# Check file removed
if ls "$PWD"/mataka-data/del-test.db 2>/dev/null; then
  echo "  ✗ bank file still on disk"
  FAIL=$((FAIL + 1))
else
  check "bank file removed from disk" "ok" "ok"
fi

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] || exit 1
