"""Export the final raw dataset from the finished crawl runs (only reads data/crawl_work).

For each source (sources.csv order) the run in configs/crawling/final_runs.json gives the
selected records (selected_record_ids.csv order) with exactly the nine plan fields. A declared
sapo re-fetch fills sapo only where the crawl row has none and the re-fetch status is
'sapo_recovered'; extraction_status of a filled row is recomputed with the crawler's rule.

Writes data/raw/articles_raw.jsonl and reports/crawl_stats.csv (replace only with --replace).
Refuses locked (.running), unfinished, pilot or under-target runs (selected < raw_targets in
run.json) and unfinished sapo re-fetches.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from crawl_data import ARTICLE_FIELDS, WORK, extraction_status, load_json

DOWNLOADED_STATUSES = {"success", "partial"}
SELECTION_STATUSES = ("primary", "fallback", "not_article_page", "missing_title", "extraction_error",
                      "publication_date_unknown", "before_fallback_date", "after_end_date", "encoding_issue")


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"{path.name}:{number}: invalid JSON line") from None
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{number}: not a JSON object")
            yield row


def export_run(directory, refetch_id=None):
    """Return (selected records with nine fields, stats). Raises ValueError for a run that is not usable.

    downloaded = distinct record_id with extraction_status success/partial and a non-empty title over
    all attempts; titleless success/partial rows are counted as downloaded_without_title.
    """
    if not directory.is_dir():
        raise ValueError("run directory not found")
    if (directory / ".running").exists():
        raise ValueError("crawler lock .running present (run in progress)")
    for name in ("selected_record_ids.csv", "crawl_report.json", "discovery_report.json", "articles.jsonl"):
        if not (directory / name).exists():
            raise ValueError(f"no {name} (run not finished)")
    report = load_json(directory / "crawl_report.json")
    if report.get("pilot"):
        raise ValueError("pilot run (crawled with --pilot-per-source), not a quota run")
    if report.get("shortfall", 0) > 0:
        raise ValueError(f"crawl_report shortfall={report['shortfall']}")
    with (directory / "selected_record_ids.csv").open(encoding="utf-8") as stream:
        selected = [row["record_id"] for row in csv.DictReader(stream)]
    wanted = set(selected)
    by_id, downloaded, untitled = {}, {}, set()
    for row in read_jsonl(directory / "articles.jsonl"):
        rid = row.get("record_id")
        if rid in wanted:
            if rid in by_id:
                raise ValueError(f"selected record_id {rid} has several rows")
            if set(row) != set(ARTICLE_FIELDS):
                raise ValueError(f"record {rid}: fields differ from the nine plan fields")
            by_id[rid] = row
        if row.get("extraction_status") in DOWNLOADED_STATUSES and isinstance(rid, str):
            title = row.get("title")
            if isinstance(title, str) and title.strip():
                downloaded.setdefault(rid, row.get("category"))
            else:
                untitled.add(rid)
    missing = [rid for rid in selected if rid not in by_id]
    if missing:
        raise ValueError(f"selected_ids_without_row={len(missing)}")

    recovered = {}
    if refetch_id:
        refetch = directory / "sapo_refetch" / refetch_id
        if not (refetch / "report.json").exists():
            raise ValueError(f"sapo re-fetch {refetch_id}: no report.json (not finished)")
        remaining = load_json(refetch / "report.json").get("remaining")
        if remaining != 0:
            raise ValueError(f"sapo re-fetch {refetch_id}: remaining={remaining}")
        recovered = {row["record_id"]: row["sapo"] for row in read_jsonl(refetch / "sapo_supplement.jsonl")
                     if row["status"] == "sapo_recovered"}
    records, missing_sapo_before = [], 0
    for rid in selected:
        record = {field: by_id[rid][field] for field in ARTICLE_FIELDS}
        if not record["sapo"]:
            missing_sapo_before += 1
            if rid in recovered:
                record["sapo"] = recovered[rid]
                record["extraction_status"] = extraction_status(record)
        records.append(record)

    discovery = load_json(directory / "discovery_report.json")
    dates = sorted(r["published_at"] for r in records if r.get("published_at"))
    by_category = sorted(Counter(downloaded.values()).items(), key=lambda kv: (-kv[1], kv[0] or ""))
    stats = {
        "inventory_urls": discovery.get("unique_article_urls"),
        "inventory_complete": discovery["inventory_traversal_complete"],
        "discovery_failures": len(discovery.get("failures", [])),
        "quarantined_sitemap_urls": sum(len(r.get("quarantined_urls", [])) for r in discovery.get("xml_recoveries", [])),
        "attempted_urls": report["attempted_urls"], "downloaded": len(downloaded),
        "downloaded_without_title": len(untitled - set(downloaded)), "selected": len(records),
        "selection_status_counts": report.get("selection_status_counts", {}),
        "selected_by_year": json.dumps(report.get("selected_by_publication_year", {})),
        "success": sum(r["extraction_status"] == "success" for r in records),
        "partial": sum(r["extraction_status"] == "partial" for r in records),
        **{f"missing_{field}": sum(not r.get(field) for r in records) for field in ARTICLE_FIELDS},
        "missing_sapo_before_refetch": missing_sapo_before, "sapo_refetch_id": refetch_id or "",
        "sapo_refetched": missing_sapo_before - sum(not r.get("sapo") for r in records),
        "raw_categories": len({r["category"] for r in records if r.get("category")}),
        "date_min": dates[0] if dates else None, "date_max": dates[-1] if dates else None,
        "downloaded_by_category": json.dumps(by_category, ensure_ascii=False),
    }
    return records, stats


def write_atomic(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig" if path.suffix == ".csv" else "utf-8", newline="") as stream:
        write(stream)
    return temporary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replace", action="store_true", help="Overwrite existing articles_raw.jsonl/crawl_stats.csv")
    parser.add_argument("--final-runs", type=Path, default=ROOT / "configs/crawling/final_runs.json")
    parser.add_argument("--work-root", type=Path, default=WORK)
    parser.add_argument("--sources", type=Path, default=ROOT / "configs/crawling/sources.csv")
    parser.add_argument("--run-config", type=Path, default=ROOT / "configs/crawling/run.json")
    parser.add_argument("--out-root", type=Path, default=ROOT, help="Base for data/raw and reports (tests use a temp dir)")
    args = parser.parse_args(argv)
    raw_path, stats_path = args.out_root / "data/raw/articles_raw.jsonl", args.out_root / "reports/crawl_stats.csv"
    existing = [str(p) for p in (raw_path, stats_path) if p.exists()]
    if existing and not args.replace:
        parser.error(f"{', '.join(existing)} exists; pass --replace to overwrite")
    final = load_json(args.final_runs)
    runs, supplements = final["runs"], final.get("sapo_supplements", {})
    targets = load_json(args.run_config)["raw_targets"]
    with args.sources.open(encoding="utf-8-sig") as stream:
        sources = list(csv.DictReader(stream))
    unknown = (set(runs) | set(supplements)) - {s["source"] for s in sources}
    if unknown:
        parser.error(f"Unknown sources in {args.final_runs.name}: {sorted(unknown)}")

    problems, records, table = [], [], []
    for source in sources:
        name, group = source["source"], source["outlet_group"]
        if not runs.get(name):
            problems.append(f"{name}: no run in {args.final_runs.name}")
            continue
        try:
            rows, stats = export_run(args.work_root / name / runs[name], supplements.get(name))
        except (OSError, ValueError, KeyError) as exc:
            problems.append(f"{name} ({runs[name]}): {exc}")
            continue
        target = targets[group]
        if stats["selected"] < target:
            problems.append(f"{name} ({runs[name]}): selected={stats['selected']} < raw target {target}")
            continue
        records += rows
        table.append({"source": name, "outlet_group": group, "run_id": runs[name], "target": target,
                      "shortfall": max(0, target - stats["selected"]), **stats})
    if problems:
        parser.error("refusing to export:\n  " + "\n  ".join(problems))

    statuses = list(SELECTION_STATUSES) + sorted({s for r in table for s in r["selection_status_counts"]}
                                                 - set(SELECTION_STATUSES))
    for row in table:
        counts = row.pop("selection_status_counts")
        row.update({f"selection_{s}": counts.get(s, 0) for s in statuses})
    fields = (["source", "outlet_group", "run_id", "inventory_urls", "inventory_complete", "discovery_failures",
               "quarantined_sitemap_urls", "attempted_urls", "downloaded", "downloaded_without_title", "selected",
               "target", "shortfall"] + [f"selection_{s}" for s in statuses]
              + ["selected_by_year", "success", "partial"] + [f"missing_{f}" for f in ARTICLE_FIELDS]
              + ["missing_sapo_before_refetch", "sapo_refetch_id", "sapo_refetched", "raw_categories", "date_min",
                 "date_max", "downloaded_by_category"])

    def write_raw(stream):
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_stats(stream):
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)

    for temporary, final_path in ((write_atomic(raw_path, write_raw), raw_path),
                                  (write_atomic(stats_path, write_stats), stats_path)):
        temporary.replace(final_path)
    print(json.dumps({"records": len(records), "articles_raw": str(raw_path), "crawl_stats": str(stats_path),
                      "by_source": {r["source"]: r["selected"] for r in table},
                      "sapo_refetched": sum(r["sapo_refetched"] for r in table)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
