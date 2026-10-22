"""Crawl original Vietnamese news metadata, with persistent inventory and resume.

Working files go to data/crawl_work/<source>/<run_id>/ (operational journal in journal/);
scripts/export_raw.py builds data/raw/articles_raw.jsonl from the finished runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTICLE_FIELDS = ("record_id", "source", "url", "title", "sapo", "category", "published_at", "crawled_at", "extraction_status")
WORK = ROOT / "data/crawl_work"
sys.path.insert(0, str(ROOT / "src"))

from ds_project.crawling.discover import discover_sitemaps, discover_yan, write_inventory
from ds_project.crawling.extract import extract, parse_datetime
from ds_project.crawling.network import Client


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def rows(path):
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{number}; preserve this file for review before resuming") from exc


def extraction_status(record):
    """Parser status of a fetched page: 'success' iff title, sapo, category and published_at are all present."""
    return "success" if all(record.get(f) for f in ("title", "sapo", "category", "published_at")) else "partial"


def eligibility(record, config):
    if record["extraction_status"] not in {"success", "partial"}:
        return "extraction_error"
    if not record.get("is_article_page"):
        return "not_article_page"
    if not record.get("title") or not record["title"].strip():
        return "missing_title"
    if any("\ufffd" in str(record.get(field) or "") for field in ("title", "sapo", "category")):
        return "encoding_issue"
    published = parse_datetime(record.get("published_at"))
    if published is None:
        return "publication_date_unknown"
    # Compare publication calendar date in Vietnam, regardless of source offset.
    day = published.astimezone(timezone(timedelta(hours=7))).date()
    if day > date.fromisoformat(config["end_date"]):
        return "after_end_date"
    if day >= date.fromisoformat(config["start_date"]):
        return "primary"
    if day >= date.fromisoformat(config["fallback_start_date"]):
        return "fallback"
    return "before_fallback_date"


def ordered_candidates(candidates, seed, primary_year=2026):
    # Unknown-year URLs are tried alongside primary URLs; dated fallback archives last.
    primary = [r for r in candidates if not r.get("archive_year_hint") or r["archive_year_hint"] >= primary_year]
    fallback_years = sorted({r["archive_year_hint"] for r in candidates
                            if r.get("archive_year_hint") and r["archive_year_hint"] < primary_year}, reverse=True)
    rng = random.Random(seed)
    rng.shuffle(primary)
    result = primary
    for year in fallback_years:
        group = [r for r in candidates if r.get("archive_year_hint") == year]
        rng.shuffle(group)
        result.extend(group)
    return result


def publication_year(record):
    parsed = parse_datetime(record.get("published_at"))
    return parsed.astimezone(timezone(timedelta(hours=7))).year if parsed else None


def selected_records(records, config, target, pending_years, inventory_complete):
    """Prefer the primary period, then newer fallback years, without skipping pending higher-priority URLs."""
    primary = [row for row in records if row["selection_status"] == "primary"][:target]
    if len(primary) >= target or not inventory_complete:
        return primary
    primary_year = int(config["start_date"][:4])
    if any(year >= primary_year and count > 0 for year, count in pending_years.items()):
        return primary
    newest_pending = max((year for year, count in pending_years.items() if count > 0),
                         default=int(config["fallback_start_date"][:4]))
    fallback = [row for row in records if row["selection_status"] == "fallback"
                and publication_year(row) >= newest_pending]
    # Stable sort keeps the seeded fetch order within each actual publication year.
    fallback.sort(key=publication_year, reverse=True)
    return primary + fallback[:target - len(primary)]


def fetch_source(client, source, profile, config, directory, target, *, pilot=False, max_attempts=None):
    candidates = list(rows(directory / "discovered_urls.jsonl"))
    records_path = directory / "articles.jsonl"
    existing_data = list(rows(records_path))
    attempts_path = client.log_dir / "article_attempts.jsonl"
    attempts = {row["record_id"]: row for row in rows(attempts_path)}
    existing = []
    for row in existing_data:
        if row["record_id"] not in attempts:
            raise ValueError("Article exists without its operational journal entry; review interrupted write before resuming")
        existing.append({**row, **attempts[row["record_id"]]})
    completed = {row["requested_url"] for row in existing}
    existing_ids = {row["record_id"] for row in existing}
    counts = Counter(row["selection_status"] for row in existing)
    total_attempts = len(existing)
    primary_year = int(config["start_date"][:4])
    candidates = ordered_candidates(candidates, config["seed"], primary_year)
    discovery = load_json(directory / "discovery_report.json")
    pending_years = Counter((candidate.get("archive_year_hint") or primary_year)
                            for candidate in candidates if candidate["url"] not in completed)
    cap_reached = False
    all_processed = False
    with records_path.open("a", encoding="utf-8") as output, attempts_path.open("a", encoding="utf-8") as journal:
        for candidate in candidates:
            if len(selected_records(existing, config, target, pending_years,
                                    discovery["inventory_traversal_complete"])) >= target:
                break
            if max_attempts is not None and total_attempts >= max_attempts:
                cap_reached = True
                break
            url = candidate["url"]
            if url in completed:
                continue
            rid = hashlib.sha256((source + "\n" + url).encode()).hexdigest()
            if rid in existing_ids:
                continue
            # Only fragment removal is used in discovery; no arbitrary query stripping.
            record = {"record_id": rid, "source": source, "url": url,
                      "requested_url": url, "title": None, "sapo": None, "category": None,
                      "published_at": None, "crawled_at": datetime.now(timezone.utc).isoformat(),
                      "extraction_status": "fetch_error", "discovered_from": candidate["discovered_from"]}
            try:
                html, final_url = client.get(url)
                record.update(final_url=final_url)
            except Exception as exc:
                record["error"] = str(exc)
            else:
                try:
                    record.update(extract(html, final_url, profile))
                    record["extraction_status"] = extraction_status(record)
                except Exception as exc:
                    record.update(extraction_status="parse_error", error=str(exc))
            record["selection_status"] = eligibility(record, config)
            metadata = {key: value for key, value in record.items() if key not in ARTICLE_FIELDS}
            metadata["record_id"] = rid
            journal.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            journal.flush()
            output.write(json.dumps({key: record[key] for key in ARTICLE_FIELDS}, ensure_ascii=False) + "\n")
            output.flush()
            existing.append(record)
            completed.add(url)
            pending_years[candidate.get("archive_year_hint") or primary_year] -= 1
            counts[record["selection_status"]] += 1
            total_attempts += 1
            if total_attempts == 1 or total_attempts % 25 == 0:
                dump_json(client.log_dir / "progress.json", {
                    "source": source, "phase": "fetching", "updated_at": datetime.now(timezone.utc).isoformat(),
                    "target": target, "attempted_urls": total_attempts,
                    "selection_status_counts": dict(counts),
                    "selected_so_far": len(selected_records(existing, config, target, pending_years,
                                                             discovery["inventory_traversal_complete"])),
                })
            if total_attempts % 25 == 0:
                print(f"{source}: fetched={total_attempts}, primary={counts['primary']}, fallback_available={counts['fallback']}", flush=True)
        else:
            all_processed = True
    chosen = selected_records(existing, config, target, pending_years,
                              discovery["inventory_traversal_complete"])
    primary_records = [row for row in chosen if row["selection_status"] == "primary"]
    write_path = directory / "selected_record_ids.csv"
    with write_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["record_id", "selection_status"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in ("record_id", "selection_status")} for row in chosen)
    report = {
        "source": source, "pilot": pilot, "target": target, "attempted_urls": total_attempts,
        "selection_status_counts": dict(counts), "selected_records": len(chosen),
        "shortfall": max(0, target - len(chosen)), "fallback_used": len(chosen) - len(primary_records),
        "selected_by_publication_year": dict(sorted(Counter(publication_year(row) for row in chosen).items(), reverse=True)),
        "attempt_budget_exhausted": cap_reached, "all_inventory_processed": all_processed,
        "inventory_traversal_complete": discovery["inventory_traversal_complete"],
        "date_range": {k: config[k] for k in ("start_date", "end_date", "fallback_start_date")},
        "note": "All article formats accepted, storing nine plan fields only. Raw quota is not a clean quota: near-dedup and manual category mapping remain.",
    }
    dump_json(directory / "crawl_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", default=["all"])
    parser.add_argument("--stage", choices=["discover", "fetch", "run"], default="run")
    parser.add_argument("--run-id", required=True, help="Stable run name for resume")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/crawling/run.json")
    parser.add_argument("--max-discovery-pages", type=int, default=2000, help="Per-source request budget; reaching it is reported as incomplete")
    parser.add_argument("--max-attempts", type=int, help="Per-source total article fetch budget across resumes")
    parser.add_argument("--pilot-per-source", type=int, help="Small diagnostic run; permits incomplete inventory and never claims quota readiness")
    args = parser.parse_args()
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.run_id):
        parser.error("run-id must contain only letters, digits, hyphen or underscore")
    if args.max_discovery_pages < 1 or (args.max_attempts is not None and args.max_attempts < 1) or (args.pilot_per_source is not None and args.pilot_per_source < 1):
        parser.error("Request budgets and pilot target must be positive")
    config = load_json(args.config)
    dates = [date.fromisoformat(config[k]) for k in ("fallback_start_date", "start_date", "end_date")]
    if dates != sorted(dates):
        parser.error("Expected fallback_start_date <= start_date <= end_date")
    if config["request_delay_seconds"] < 0 or config["timeout_seconds"] <= 0 or config["retries"] < 0:
        parser.error("Invalid request settings")
    profiles = load_json(ROOT / "configs/crawling/profiles.json")
    with (ROOT / "configs/crawling/sources.csv").open(encoding="utf-8-sig") as stream:
        sources = {row["source"]: row for row in csv.DictReader(stream)}
    names = list(sources) if args.sources == ["all"] else list(dict.fromkeys(args.sources))
    if set(names) - sources.keys():
        parser.error(f"Unknown sources: {set(names) - sources.keys()}")
    failures = []
    for name in names:
        directory = WORK / name / args.run_id
        directory.mkdir(parents=True, exist_ok=True)
        log_dir = directory / "journal"
        snapshot = {"config": config, "profile": profiles[name], "source": sources[name],
                    "pilot_per_source": args.pilot_per_source, "article_fields": list(ARTICLE_FIELDS)}
        snapshot_path = directory / "run_manifest.json"
        if snapshot_path.exists() and load_json(snapshot_path) != snapshot:
            parser.error(f"Run configuration changed for {name}; choose a new run-id")
        dump_json(snapshot_path, snapshot)
        lock = directory / ".running"
        try:
            with lock.open("x") as stream:
                stream.write(str(datetime.now(timezone.utc)))
        except FileExistsError:
            parser.error(f"Run is locked: {lock}. Check that no crawler is running before removing a stale lock.")
        try:
            client = Client(sources[name]["domain"], {**config, **profiles[name].get("network", {})}, log_dir)
            client.load_robots()
            if args.stage in {"discover", "run"}:
                discover = discover_yan if profiles[name].get("discovery") == "yan" else discover_sitemaps
                if discover is discover_yan:
                    inventory, report = discover(client, config, directory, args.max_discovery_pages)
                else:
                    inventory, report = discover(client, profiles[name], config, directory, args.max_discovery_pages)
                write_inventory(directory / "discovered_urls.jsonl", inventory, report)
                print(json.dumps({"source": name, "discovery": report}, ensure_ascii=False), flush=True)
                if args.stage == "discover" and not report["inventory_traversal_complete"]:
                    failures.append({"source": name, "error": "Discovery incomplete; inspect discovery_report.json"})
            if args.stage in {"fetch", "run"}:
                report = load_json(directory / "discovery_report.json")
                if not report["inventory_traversal_complete"] and not args.pilot_per_source:
                    raise ValueError("Incomplete inventory: inspect discovery_report.json and resolve failures/budget before production sampling")
                target = args.pilot_per_source or config["raw_targets"][sources[name]["outlet_group"]]
                result = fetch_source(client, name, profiles[name], config, directory, target,
                                      pilot=bool(args.pilot_per_source), max_attempts=args.max_attempts)
                print(json.dumps(result, ensure_ascii=False), flush=True)
                if result["shortfall"]:
                    failures.append({"source": name, "error": f"Raw target shortfall: {result['shortfall']}"})
        except Exception as exc:
            failure = {"source": name, "error": str(exc)}
            failures.append(failure)
            dump_json(log_dir / "failure.json", failure)
            print(json.dumps(failure, ensure_ascii=False), file=sys.stderr, flush=True)
        finally:
            lock.unlink()
    if failures:
        print(json.dumps({"failures": failures}, ensure_ascii=False), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
