"""Re-fetch selected articles whose sapo is missing, writing a supplement beside the untouched crawl run.

Output: data/crawl_work/<source>/<run_id>/sapo_refetch/<refetch_id>/ (request journal in journal/).
scripts/export_raw.py applies it to data/raw/articles_raw.jsonl when listed in
configs/crawling/final_runs.json under sapo_supplements.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from crawl_data import WORK, dump_json, load_json, rows
from ds_project.crawling.extract import extract
from ds_project.crawling.network import Client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id", required=True, help="Crawl run whose selected records are supplemented")
    parser.add_argument("--refetch-id", required=True, help="Stable name for resume")
    parser.add_argument("--request-delay", type=float, help="Override the profile delay for this re-fetch")
    parser.add_argument("--max-consecutive-errors", type=int, default=5, help="Stop (resume later) after repeated fetch failures")
    args = parser.parse_args()
    for value in (args.run_id, args.refetch_id):
        if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
            parser.error("IDs must contain only letters, digits, hyphen or underscore")
    config = load_json(ROOT / "configs/crawling/run.json")
    profile = load_json(ROOT / "configs/crawling/profiles.json")[args.source]
    with (ROOT / "configs/crawling/sources.csv").open(encoding="utf-8-sig") as stream:
        domain = next(row["domain"] for row in csv.DictReader(stream) if row["source"] == args.source)
    run_dir = WORK / args.source / args.run_id
    with (run_dir / "selected_record_ids.csv").open(encoding="utf-8") as stream:
        selected = {row["record_id"] for row in csv.DictReader(stream)}
    targets = [row for row in rows(run_dir / "articles.jsonl") if row["record_id"] in selected and not row.get("sapo")]

    directory = run_dir / "sapo_refetch" / args.refetch_id
    directory.mkdir(parents=True, exist_ok=True)
    network = {**config, **profile.get("network", {})}
    if args.request_delay is not None:
        network["request_delay_seconds"] = args.request_delay
    snapshot = {"source": args.source, "run_id": args.run_id, "profile": profile, "network": network,
                "policy": "Raw articles.jsonl is not modified. A sapo is accepted only when the re-fetched "
                          "visible title equals the stored title; otherwise the row is kept for review."}
    if (directory / "manifest.json").exists() and load_json(directory / "manifest.json") != snapshot:
        parser.error("Re-fetch configuration changed; choose a new refetch-id")
    dump_json(directory / "manifest.json", snapshot)

    output_path = directory / "sapo_supplement.jsonl"
    done = {row["record_id"] for row in rows(output_path)}
    client = Client(domain, network, directory / "journal")
    client.load_robots()
    errors = 0
    with output_path.open("a", encoding="utf-8") as output:
        for record in targets:
            if record["record_id"] in done:
                continue
            result = {"record_id": record["record_id"], "url": record["url"], "sapo": None, "status": "fetch_error",
                      "refetched_at": datetime.now(timezone.utc).isoformat()}
            try:
                html, final_url = client.get(record["url"])
            except Exception as exc:
                errors += 1
                if errors >= args.max_consecutive_errors:
                    print(f"Stopping after {errors} consecutive fetch errors; rerun the same command to resume.", file=sys.stderr)
                    return 1
                continue  # Not written, so a resume retries it.
            errors = 0
            fresh = extract(html, final_url, profile)
            result.update(final_url=final_url, title_matches=fresh["title"] == record["title"],
                          evidence=fresh["extraction_evidence"].get("sapo"))
            if not result["title_matches"]:
                result.update(status="title_changed", refetched_title=fresh["title"])
            elif fresh["sapo"]:
                result.update(status="sapo_recovered", sapo=fresh["sapo"])
            else:
                result["status"] = "sapo_absent"
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            done.add(record["record_id"])
    counts = {}
    for row in rows(output_path):
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    report = {"source": args.source, "run_id": args.run_id, "targets": len(targets), "written": len(done),
              "remaining": len(targets) - len(done), "status_counts": counts}
    dump_json(directory / "report.json", report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["remaining"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
