"""Generate markdown report from differential test results."""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from run_diff import RunReport


def generate_report(report: RunReport, path: Path) -> None:
    """Write a markdown report from the run results."""
    lines: list[str] = []

    mean_overlap = statistics.mean(report.overlap_scores) if report.overlap_scores else 0
    median_overlap = statistics.median(report.overlap_scores) if report.overlap_scores else 0
    p95_overlap = (
        sorted(report.overlap_scores)[int(len(report.overlap_scores) * 0.95)]
        if report.overlap_scores else 0
    )

    # Collect per-query latencies across all fixtures
    all_lat_mt = [
        qr.latency_mataka_ms
        for fr in report.fixture_results
        for qr in fr.query_results
    ]
    all_lat_hs = [
        qr.latency_hindsight_ms
        for fr in report.fixture_results
        for qr in fr.query_results
    ]
    p50_mt = statistics.median(all_lat_mt) if all_lat_mt else 0
    p95_mt = (sorted(all_lat_mt)[int(len(all_lat_mt) * 0.95)] if all_lat_mt else 0)
    p50_hs = statistics.median(all_lat_hs) if all_lat_hs else 0
    p95_hs = (sorted(all_lat_hs)[int(len(all_lat_hs) * 0.95)] if all_lat_hs else 0)

    lines.append("# Differential Test Report: mataka vs Hindsight")
    lines.append("")
    lines.append(f"- **mataka**: `{report.mataka_url}`")
    lines.append(f"- **Hindsight**: `{report.hindsight_url}`")
    lines.append("")

    # Headline
    lines.append("## Headline")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Fixtures tested | {report.total_fixtures} |")
    lines.append(f"| Total queries | {report.total_queries} |")
    lines.append(f"| Schema validation | {report.schema_pass_count} pass, {report.schema_fail_count} fail |")
    lines.append(f"| Tag isolation | {report.isolation_pass_count} pass, {report.isolation_fail_count} fail |")
    lines.append(f"| Mean recall@10 Jaccard | {mean_overlap:.4f} |")
    lines.append(f"| Median recall@10 Jaccard | {median_overlap:.4f} |")
    lines.append(f"| P95 recall@10 Jaccard | {p95_overlap:.4f} |")
    lines.append(f"| mataka recall latency p50 | {p50_mt:.0f} ms |")
    lines.append(f"| mataka recall latency p95 | {p95_mt:.0f} ms |")
    lines.append(f"| hindsight recall latency p50 | {p50_hs:.0f} ms |")
    lines.append(f"| hindsight recall latency p95 | {p95_hs:.0f} ms |")
    lines.append(f"| SDK smoke (mataka) | {'PASS' if report.sdk_smoke_pass else 'FAIL'} |")
    lines.append(f"| mataka RSS | {report.rss_mataka_mb:.1f} MB |")
    lines.append(f"| Hindsight RSS | {report.rss_hindsight_mb:.1f} MB |")
    lines.append("")

    # Per-fixture breakdown
    lines.append("## Per-Fixture Results")
    lines.append("")
    lines.append("| Fixture | Type | Overlap | Latency mataka (ms) | Latency hindsight (ms) | Schema | Isolation |")
    lines.append("|---------|------|---------|---------------------|------------------------|--------|-----------|")

    for fr in report.fixture_results:
        # Compute per-fixture stats
        overlaps = [qr.recall_overlap for qr in fr.query_results if qr.recall_overlap is not None]
        mean_overlap_f = statistics.mean(overlaps) if overlaps else 0

        lat_mt = statistics.mean([qr.latency_mataka_ms for qr in fr.query_results]) if fr.query_results else 0
        lat_hs = statistics.mean([qr.latency_hindsight_ms for qr in fr.query_results]) if fr.query_results else 0

        schema_ok = (
            fr.retain_valid_mataka and fr.retain_valid_hindsight
            and all(qr.schema_valid_mataka and qr.schema_valid_hindsight for qr in fr.query_results)
        )

        iso_results = [qr.isolation_pass for qr in fr.query_results if qr.isolation_pass is not None]
        if iso_results:
            iso_str = "PASS" if all(iso_results) else "FAIL"
        else:
            iso_str = "N/A"

        lines.append(
            f"| {fr.fixture_id} | {fr.fixture_name} | {mean_overlap_f:.3f} | "
            f"{lat_mt:.0f} | {lat_hs:.0f} | "
            f"{'PASS' if schema_ok else 'FAIL'} | {iso_str} |"
        )

    lines.append("")

    # Errors
    if report.errors:
        lines.append("## Errors")
        lines.append("")
        for err in report.errors[:50]:  # cap at 50
            lines.append(f"- {err}")
        if len(report.errors) > 50:
            lines.append(f"- ... and {len(report.errors) - 50} more")
        lines.append("")

    # Overlap distribution
    if report.overlap_scores:
        lines.append("## Overlap Distribution")
        lines.append("")
        lines.append("| Bucket | Count |")
        lines.append("|--------|-------|")
        buckets = {"1.0": 0, "0.8-1.0": 0, "0.5-0.8": 0, "0.2-0.5": 0, "0.0-0.2": 0}
        for s in report.overlap_scores:
            if s >= 1.0:
                buckets["1.0"] += 1
            elif s >= 0.8:
                buckets["0.8-1.0"] += 1
            elif s >= 0.5:
                buckets["0.5-0.8"] += 1
            elif s >= 0.2:
                buckets["0.2-0.5"] += 1
            else:
                buckets["0.0-0.2"] += 1
        for bucket, count in buckets.items():
            lines.append(f"| {bucket} | {count} |")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append("*Generated by mataka differential test harness*")

    path.write_text("\n".join(lines))
