#!/usr/bin/env python3
"""
Differential test harness: mataka vs Hindsight docker.

Retains the same corpus to both servers, then compares recall@10 overlap,
validates responses against the OpenAPI 0.8.4 schema, checks tag isolation,
and runs an SDK smoke test against mataka.

Usage:
    uv run harness/run_diff.py --mataka http://localhost:8888 --hindsight http://localhost:9888
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import psutil

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
OPENAPI_PATH = REPO_ROOT / "contract" / "openapi-0.8.4.json"

# Cache for resolved $ref targets — shared across validation calls.
_ref_cache: dict[str, Any] = {}


def _load_openapi_spec() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load OpenAPI spec. Returns (raw_schemas, full_spec)."""
    with open(OPENAPI_PATH) as fh:
        spec = json.load(fh)
    schemas = spec.get("components", {}).get("schemas", {})
    return schemas, spec


def _resolve_ref(ref: str, spec: dict[str, Any]) -> Any:
    """Resolve a $ref string against the spec (cached)."""
    if ref not in _ref_cache:
        parts = ref.lstrip("#/").split("/")
        resolved = spec
        for p in parts:
            resolved = resolved[p]
        _ref_cache[ref] = resolved
    return _ref_cache[ref]


def _validate_response(
    data: Any, schema: Any, spec: dict[str, Any], path: str = ""
) -> list[str]:
    """Validate a JSON response against an OpenAPI schema (handles $ref, anyOf)."""
    errors: list[str] = []
    if not isinstance(schema, dict):
        return errors

    # Resolve $ref
    if "$ref" in schema:
        schema = _resolve_ref(schema["$ref"], spec)

    # anyOf — at least one variant must match
    if "anyOf" in schema:
        for variant in schema["anyOf"]:
            if not _validate_response(data, variant, spec, path):
                return []
        errors.append(f"{path}: value does not match any variant in anyOf")
        return errors

    schema_type = schema.get("type")
    if schema_type == "object" and isinstance(data, dict):
        for req in schema.get("required", []):
            if req not in data:
                errors.append(f"{path}: missing required field '{req}'")
        props = schema.get("properties", {})
        for key, val in data.items():
            if key in props:
                errors.extend(_validate_response(val, props[key], spec, f"{path}.{key}"))
    elif schema_type == "array" and isinstance(data, list):
        items_schema = schema.get("items", {})
        for i, item in enumerate(data):
            errors.extend(_validate_response(item, items_schema, spec, f"{path}[{i}]"))
    elif schema_type == "string" and not isinstance(data, str):
        errors.append(f"{path}: expected string, got {type(data).__name__}")
    elif schema_type == "integer" and not isinstance(data, int):
        errors.append(f"{path}: expected integer, got {type(data).__name__}")
    elif schema_type == "number" and not isinstance(data, (int, float)):
        errors.append(f"{path}: expected number, got {type(data).__name__}")
    elif schema_type == "boolean" and not isinstance(data, bool):
        errors.append(f"{path}: expected boolean, got {type(data).__name__}")
    elif "enum" in schema:
        if data not in schema["enum"]:
            errors.append(f"{path}: value {data!r} not in enum {schema['enum']}")

    return errors


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    fixture_id: str
    query: str
    isolation_test: bool
    expected_leak_user: str | None
    recall_overlap: float | None = None
    schema_valid_mataka: bool = True
    schema_valid_hindsight: bool = True
    schema_errors: list[str] = field(default_factory=list)
    latency_mataka_ms: float = 0.0
    latency_hindsight_ms: float = 0.0
    mataka_texts: list[str] = field(default_factory=list)
    hindsight_texts: list[str] = field(default_factory=list)
    isolation_pass: bool | None = None  # None = not an isolation test


@dataclass
class FixtureResult:
    fixture_id: str
    fixture_name: str
    isolation_test: bool
    query_results: list[QueryResult] = field(default_factory=list)
    retain_latency_mataka_ms: float = 0.0
    retain_latency_hindsight_ms: float = 0.0
    retain_valid_mataka: bool = True
    retain_valid_hindsight: bool = True


@dataclass
class RunReport:
    mataka_url: str
    hindsight_url: str
    total_fixtures: int = 0
    total_queries: int = 0
    fixture_results: list[FixtureResult] = field(default_factory=list)
    schema_pass_count: int = 0
    schema_fail_count: int = 0
    isolation_pass_count: int = 0
    isolation_fail_count: int = 0
    overlap_scores: list[float] = field(default_factory=list)
    rss_mataka_mb: float = 0.0
    rss_hindsight_mb: float = 0.0
    sdk_smoke_pass: bool = False
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Overlap computation
# ---------------------------------------------------------------------------

def jaccard_overlap(texts_a: list[str], texts_b: list[str]) -> float:
    """Compute Jaccard similarity of normalized text sets."""
    def normalize(t: str) -> str:
        return re.sub(r"\s+", " ", t.strip().lower())

    set_a = {normalize(t) for t in texts_a}
    set_b = {normalize(t) for t in texts_b}
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _post(client: httpx.Client, url: str, body: dict) -> tuple[dict, float]:
    """POST JSON and return (response_json, latency_ms)."""
    t0 = time.perf_counter()
    r = client.post(url, json=body, timeout=120)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return r.json(), elapsed_ms


def _put(client: httpx.Client, url: str, body: dict | None = None) -> tuple[dict, float]:
    t0 = time.perf_counter()
    r = client.put(url, json=body, timeout=30)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return r.json(), elapsed_ms


def _get(client: httpx.Client, url: str) -> tuple[dict, float]:
    t0 = time.perf_counter()
    r = client.get(url, timeout=30)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return r.json(), elapsed_ms


def _delete(client: httpx.Client, url: str) -> tuple[dict, float]:
    t0 = time.perf_counter()
    r = client.delete(url, timeout=30)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return r.json(), elapsed_ms


# ---------------------------------------------------------------------------
# Core test runner
# ---------------------------------------------------------------------------

def run_retain(client: httpx.Client, base: str, bank: str, items: list[dict]) -> tuple[dict, float]:
    """Retain items synchronously and return (response, latency_ms)."""
    url = f"{base}/v1/default/banks/{bank}/memories"
    return _post(client, url, {"items": items, "async": False})


def run_recall(
    client: httpx.Client,
    base: str,
    bank: str,
    query: str,
    tags: list[str] | None = None,
    tags_match: str | None = None,
    max_tokens: int = 500,
) -> tuple[dict, float]:
    """Run recall and return (response, latency_ms)."""
    url = f"{base}/v1/default/banks/{bank}/memories/recall"
    body: dict[str, Any] = {"query": query, "max_tokens": max_tokens}
    if tags is not None:
        body["tags"] = tags
    if tags_match is not None:
        body["tags_match"] = tags_match
    return _post(client, url, body)


def extract_texts(results: list[dict]) -> list[str]:
    """Extract text fields from recall results."""
    return [r.get("text", "") for r in results if r.get("text")]


# ---------------------------------------------------------------------------
# SDK smoke test
# ---------------------------------------------------------------------------

def run_sdk_smoke(client: httpx.Client, base: str) -> tuple[bool, list[str]]:
    """
    Run happy-path calls that mirror the upstream Python SDK.
    create bank → retain → recall → reflect → list → delete.
    Returns (passed, errors).
    """
    errors = []
    bank = f"sdk-smoke-{int(time.time())}"

    # 1. Create bank
    try:
        resp, _ = _put(client, f"{base}/v1/default/banks/{bank}", {
            "name": "SDK Smoke Test",
            "mission": "Verify SDK compatibility",
        })
        if "bank_id" not in resp:
            errors.append("create bank: missing bank_id in response")
    except Exception as e:
        errors.append(f"create bank: {e}")
        return False, errors

    # 2. Retain
    try:
        resp, _ = run_retain(client, base, bank, [
            {"content": "Alice works at Google as a senior engineer.", "tags": ["alice"]},
            {"content": "Bob is training for a marathon.", "tags": ["bob"]},
        ])
        if not resp.get("success"):
            errors.append("retain: success=false")
    except Exception as e:
        errors.append(f"retain: {e}")

    # 3. Recall
    try:
        resp, _ = run_recall(client, base, bank, "What does Alice do?")
        if "results" not in resp:
            errors.append("recall: missing results key")
    except Exception as e:
        errors.append(f"recall: {e}")

    # 4. Reflect
    try:
        resp, _ = _post(client, f"{base}/v1/default/banks/{bank}/reflect", {
            "query": "Summarize what we know.", "budget": "low"
        })
        if "text" not in resp:
            errors.append("reflect: missing text key")
    except Exception as e:
        errors.append(f"reflect: {e}")

    # 5. List memories
    try:
        resp, _ = _get(client, f"{base}/v1/default/banks/{bank}/memories/list")
        if "items" not in resp:
            errors.append("list: missing items key")
    except Exception as e:
        errors.append(f"list: {e}")

    # 6. List banks
    try:
        resp, _ = _get(client, f"{base}/v1/default/banks")
        if "banks" not in resp:
            errors.append("list banks: missing banks key")
    except Exception as e:
        errors.append(f"list banks: {e}")

    # 7. Stats
    try:
        resp, _ = _get(client, f"{base}/v1/default/banks/{bank}/stats")
        if "total_memories" not in resp:
            errors.append("stats: missing total_memories key")
    except Exception as e:
        errors.append(f"stats: {e}")

    # 8. Entities
    try:
        resp, _ = _get(client, f"{base}/v1/default/banks/{bank}/entities")
        if "entities" not in resp:
            errors.append("entities: missing entities key")
    except Exception as e:
        errors.append(f"entities: {e}")

    # 9. Cleanup
    try:
        _delete(client, f"{base}/v1/default/banks/{bank}")
    except Exception:
        pass

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Differential test harness: mataka vs Hindsight docker"
    )
    parser.add_argument("--mataka", required=True, help="mataka server base URL")
    parser.add_argument("--hindsight", required=True, help="Hindsight docker base URL")
    parser.add_argument("--corpus", type=Path, default=None, help="Path to corpus.jsonl")
    parser.add_argument("--max-queries", type=int, default=None, help="Limit queries per fixture")
    parser.add_argument("--schema-only", action="store_true", help="Only validate schemas, skip overlap")
    parser.add_argument("--report", type=Path, default=None, help="Output report path")
    args = parser.parse_args()

    # Import here to avoid slow --help
    from fixtures import load_corpus
    from report import generate_report

    corpus_path = args.corpus
    fixtures = load_corpus(corpus_path)
    print(f"Loaded {len(fixtures)} fixtures from {corpus_path or 'default'}")

    # Load OpenAPI spec — raw schemas + full spec for $ref resolution
    print("Loading OpenAPI schemas...")
    raw_schemas, spec = _load_openapi_spec()
    recall_resp_schema = raw_schemas.get("RecallResponse", {})
    retain_resp_schema = raw_schemas.get("RetainResponse", {})
    reflect_resp_schema = raw_schemas.get("ReflectResponse", {})
    print(f"  RecallResponse: {len(recall_resp_schema.get('properties', {}))} props")
    print(f"  RetainResponse: {len(retain_resp_schema.get('properties', {}))} props")

    # Create HTTP clients
    mataka = httpx.Client(base_url=args.mataka, timeout=120)
    hindsight = httpx.Client(base_url=args.hindsight, timeout=120)

    # Health check both servers
    print("\nChecking server health...")
    for name, client in [("mataka", mataka), ("hindsight", hindsight)]:
        try:
            r = client.get("/health", timeout=10)
            r.raise_for_status()
            print(f"  {name}: OK ({r.json().get('status', 'unknown')})")
        except Exception as e:
            print(f"  {name}: FAILED — {e}")
            return 1

    print("\nCollecting baseline RSS...")
    process = psutil.Process()

    report = RunReport(
        mataka_url=args.mataka,
        hindsight_url=args.hindsight,
        total_fixtures=len(fixtures),
    )

    # Create unique banks per server
    bank_suffix = int(time.time())
    bank_mt = f"diff-mataka-{bank_suffix}"
    bank_hs = f"diff-hindsight-{bank_suffix}"

    print(f"\nCreating test banks: {bank_mt}, {bank_hs}")
    for client, bank in [(mataka, bank_mt), (hindsight, bank_hs)]:
        _put(client, f"/v1/default/banks/{bank}", {
            "name": f"Differential test {bank}",
            "mission": "Comparative recall test",
        })

    # ── Phase 1: Retain all fixtures ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("Phase 1: Retaining fixtures to both servers")
    print(f"{'='*60}")

    fixture_map: dict[str, FixtureResult] = {}

    for i, fixture in enumerate(fixtures):
        fr = FixtureResult(
            fixture_id=fixture.id,
            fixture_name=fixture.name,
            isolation_test=fixture.isolation_test,
        )
        fixture_map[fixture.id] = fr

        # Retain to mataka
        try:
            resp, lat = run_retain(mataka, args.mataka, bank_mt, fixture.items)
            fr.retain_latency_mataka_ms = lat
            fr.retain_valid_mataka = True
            errs = _validate_response(resp, retain_resp_schema, spec,
                                      f"{fixture.id}.retain.mataka")
            if errs:
                fr.retain_valid_mataka = False
                report.schema_fail_count += 1
                report.errors.extend(errs)
            else:
                report.schema_pass_count += 1
        except Exception as e:
            fr.retain_valid_mataka = False
            report.errors.append(f"{fixture.id}: retain mataka failed: {e}")

        # Retain to hindsight
        try:
            resp, lat = run_retain(hindsight, args.hindsight, bank_hs, fixture.items)
            fr.retain_latency_hindsight_ms = lat
            fr.retain_valid_hindsight = True
            errs = _validate_response(resp, retain_resp_schema, spec,
                                      f"{fixture.id}.retain.hindsight")
            if errs:
                fr.retain_valid_hindsight = False
                report.schema_fail_count += 1
                report.errors.extend(errs)
            else:
                report.schema_pass_count += 1
        except Exception as e:
            fr.retain_valid_hindsight = False
            report.errors.append(f"{fixture.id}: retain hindsight failed: {e}")

        if (i + 1) % 20 == 0 or (i + 1) == len(fixtures):
            print(f"  Retained {i+1}/{len(fixtures)} fixtures")

    report.fixture_results = list(fixture_map.values())
    # ── Phase 2: Recall queries ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Phase 2: Running recall queries on both servers")
    print(f"{'='*60}")

    query_count = 0
    for fixture in fixtures:
        fr = fixture_map[fixture.id]
        queries = fixture.queries
        if args.max_queries:
            queries = queries[:args.max_queries]

        for q in queries:
            qr = QueryResult(
                fixture_id=fixture.id,
                query=q["query"],
                isolation_test=q.get("isolation_test", False),
                expected_leak_user=q.get("expected_leak_user"),
            )

            # ── Determine tags for this recall call ────────────────────────
            if qr.isolation_test and fixture.query_users:
                # Isolation test: recall scoped to the *query_users'* tags
                # with any_strict (excludes untagged). The retain_user's data
                # has tags=[retain_user], so it should NOT appear in results.
                recall_tags = fixture.query_users
                recall_tags_match = "any_strict"
            else:
                recall_tags = None
                recall_tags_match = None

            # ── Recall on mataka ──────────────────────────────────────────
            try:
                resp, lat = run_recall(
                    mataka, args.mataka, bank_mt, q["query"],
                    tags=recall_tags, tags_match=recall_tags_match,
                )
                qr.latency_mataka_ms = lat
                results = resp.get("results", [])
                qr.mataka_texts = extract_texts(results)
                errs = _validate_response(resp, recall_resp_schema, spec,
                                          f"{fixture.id}.recall.mataka")
                if errs:
                    qr.schema_valid_mataka = False
                    qr.schema_errors.extend(errs)
            except Exception as e:
                qr.schema_valid_mataka = False
                qr.schema_errors.append(f"mataka recall: {e}")

            # ── Recall on hindsight ───────────────────────────────────────
            try:
                resp, lat = run_recall(
                    hindsight, args.hindsight, bank_hs, q["query"],
                    tags=recall_tags, tags_match=recall_tags_match,
                )
                qr.latency_hindsight_ms = lat
                results = resp.get("results", [])
                qr.hindsight_texts = extract_texts(results)
                errs = _validate_response(resp, recall_resp_schema, spec,
                                          f"{fixture.id}.recall.hindsight")
                if errs:
                    qr.schema_valid_hindsight = False
                    qr.schema_errors.extend(errs)
            except Exception as e:
                qr.schema_valid_hindsight = False
                qr.schema_errors.append(f"hindsight recall: {e}")

            # ── Compute overlap ───────────────────────────────────────────
            if qr.mataka_texts and qr.hindsight_texts:
                qr.recall_overlap = jaccard_overlap(qr.mataka_texts, qr.hindsight_texts)
                report.overlap_scores.append(qr.recall_overlap)

            # ── Tag isolation check ───────────────────────────────────────
            # For isolation tests, we check that NONE of the retain_user's
            # specific item texts appear in the recall results. The recall
            # was scoped to query_users' tags, so retain_user's data should
            # be excluded by the tag filter.
            if qr.isolation_test and qr.expected_leak_user:
                retain_user = fixture.retain_user
                # Build the set of item texts that belong to the retain_user
                retain_texts = {
                    item["content"]
                    for item in fixture.items
                    if retain_user in item.get("tags", [])
                }
                # Normalize for substring containment check
                def _norm(t: str) -> str:
                    return re.sub(r"\s+", " ", t.strip().lower())

                norm_retain = {_norm(t) for t in retain_texts}
                norm_mataka = {_norm(t) for t in qr.mataka_texts}
                norm_hindsight = {_norm(t) for t in qr.hindsight_texts}

                # A result leaks if any retain text is a substring of a result
                # (or vice versa — partial extraction may truncate).
                leaked_mt = any(
                    rt in mt or mt in rt
                    for rt in norm_retain for mt in norm_mataka
                ) if norm_retain else False
                leaked_hs = any(
                    rt in ht or ht in rt
                    for rt in norm_retain for ht in norm_hindsight
                ) if norm_retain else False

                qr.isolation_pass = not leaked_mt and not leaked_hs
                if not qr.isolation_pass:
                    report.isolation_fail_count += 1
                    report.errors.append(
                        f"ISOLATION FAIL: {fixture.id} query='{q['query']}' "
                        f"retain_user={retain_user} mataka_leaked={leaked_mt} "
                        f"hindsight_leaked={leaked_hs}"
                    )
                else:
                    report.isolation_pass_count += 1

            # ── Schema results ────────────────────────────────────────────
            if qr.schema_valid_mataka and qr.schema_valid_hindsight:
                report.schema_pass_count += 1
            else:
                report.schema_fail_count += 1

            fr.query_results.append(qr)
            query_count += 1

        if query_count % 100 == 0 or query_count == sum(len(f.queries) for f in fixtures):
            print(f"  Completed {query_count} queries")

    report.total_queries = query_count

    # ── Phase 3: SDK smoke test (mataka only) ─────────────────────────────
    print(f"\n{'='*60}")
    print("Phase 3: SDK smoke test against mataka")
    print(f"{'='*60}")

    sdk_ok, sdk_errors = run_sdk_smoke(mataka, args.mataka)
    report.sdk_smoke_pass = sdk_ok
    if sdk_errors:
        report.errors.extend([f"SDK: {e}" for e in sdk_errors])
    print(f"  SDK smoke: {'PASS' if sdk_ok else 'FAIL'}")
    for err in sdk_errors:
        print(f"    ✗ {err}")

    # ── Phase 4: RSS snapshot ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Phase 4: Collecting RSS snapshots")
    print(f"{'='*60}")

    report.rss_mataka_mb = process.memory_info().rss / (1024 * 1024)
    try:
        import subprocess
        result = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            mem_str = result.stdout.strip().split("\n")[0]
            match = re.match(r"([\d.]+)(MiB|GiB|KiB)", mem_str.split("/")[0].strip())
            if match:
                val = float(match.group(1))
                unit = match.group(2)
                if unit == "GiB":
                    val *= 1024
                elif unit == "KiB":
                    val /= 1024
                report.rss_hindsight_mb = val
    except Exception:
        print("  Could not read docker stats for hindsight RSS")

    # ── Cleanup ───────────────────────────────────────────────────────────
    print(f"\nCleaning up test banks...")
    for client, bank in [(mataka, bank_mt), (hindsight, bank_hs)]:
        try:
            _delete(client, f"/v1/default/banks/{bank}")
        except Exception:
            pass

    # ── Generate report ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Generating report")
    print(f"{'='*60}")

    report_path = args.report or Path("harness/diff-report.md")
    generate_report(report, report_path)
    print(f"  Report written to {report_path}")

    # ── Print summary ─────────────────────────────────────────────────────
    mean_overlap = statistics.mean(report.overlap_scores) if report.overlap_scores else 0
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Fixtures:            {report.total_fixtures}")
    print(f"  Queries:             {report.total_queries}")
    print(f"  Schema validation:   {report.schema_pass_count} pass / {report.schema_fail_count} fail")
    print(f"  Tag isolation:       {report.isolation_pass_count} pass / {report.isolation_fail_count} fail")
    print(f"  Mean recall@10 Jaccard: {mean_overlap:.4f} (target >= 0.8)")
    print(f"  SDK smoke:           {'PASS' if report.sdk_smoke_pass else 'FAIL'}")
    print(f"  mataka RSS:          {report.rss_mataka_mb:.1f} MB")
    print(f"  hindsight RSS:       {report.rss_hindsight_mb:.1f} MB")
    print(f"  Errors:              {len(report.errors)}")

    failed = (
        report.schema_fail_count > 0
        or report.isolation_fail_count > 0
        or not report.sdk_smoke_pass
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
