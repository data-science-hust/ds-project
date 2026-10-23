"""Build the clean dataset from data/raw/articles_raw.jsonl and reports/crawl_stats.csv.

Writes data/interim/*.csv, data/processed/articles_clean.parquet, reports/funnel_*.csv and reports/data_report.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ds_project.checking.categories import load_mapping
from ds_project.checking.clean import (CLEAN_COLUMNS, FIX_FIELDS, ISSUE_FIELDS, apply_dedup, check_and_fix, funnel,
                                       load_rules, read_crawl_stats, read_csv, read_raw, write_csv)
from ds_project.checking.report import render

PAIR_FIELDS = ["record_id_a", "record_id_b", "source_a", "source_b", "jc3", "number_conflict", "days_apart", "rule"]
REVIEW_FIELDS = PAIR_FIELDS + ["auto_linked", "same_group_already", "sapo_jc3", "word_containment", "title_a", "title_b"]
INTERIM_FILES = ("issues.csv", "rejected.csv", "fixes.csv", "dedup_pairs.csv", "dedup_review.csv", "template_series.csv")
REPORT_FILES = ("funnel_by_source.csv", "funnel_by_source_category.csv", "data_report.md")


def shown(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)).replace("\\", "/") if path.is_relative_to(ROOT) else str(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replace", action="store_true", help="Overwrite existing interim/processed/report outputs")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/articles_raw.jsonl")
    parser.add_argument("--crawl-stats", type=Path, default=ROOT / "reports/crawl_stats.csv")
    parser.add_argument("--out-root", type=Path, default=ROOT,
                        help="Base for data/interim, data/processed, reports (tests use a temp dir)")
    parser.add_argument("--sources", type=Path, default=ROOT / "configs/crawling/sources.csv")
    parser.add_argument("--mapping", type=Path, default=ROOT / "configs/crawling/category_mapping.csv")
    parser.add_argument("--rules", type=Path, default=ROOT / "configs/checking/rejection_rules.json")
    parser.add_argument("--dedup-config", type=Path, default=ROOT / "configs/checking/dedup.json")
    parser.add_argument("--decisions", type=Path, default=ROOT / "configs/checking/dedup_decisions.csv")
    args = parser.parse_args(argv)

    out = args.out_root
    interim, parquet, report_dir = out / "data/interim", out / "data/processed/articles_clean.parquet", out / "reports"
    outputs = [interim / name for name in INTERIM_FILES] + [parquet] + [report_dir / name for name in REPORT_FILES]
    existing = [shown(p) for p in outputs if p.exists()]
    if existing and not args.replace:
        parser.error(f"outputs exist ({', '.join(existing[:3])}{', ...' if len(existing) > 3 else ''}); "
                     "pass --replace to overwrite")
    for path, how in ((args.raw, "scripts/export_raw.py"), (args.crawl_stats, "scripts/export_raw.py")):
        if not path.exists():
            parser.error(f"{path} not found; run {how} first")
    sources = {row["source"]: row for row in read_csv(args.sources)}
    crawl_rows = read_csv(args.crawl_stats)
    try:
        crawl_stats = read_crawl_stats(crawl_rows, sources)
        pools = read_raw(args.raw, sources)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    mismatched = [s for s in sources if len(pools.get(s, [])) != crawl_stats[s]["selected"]]
    if mismatched:
        parser.error(f"articles_raw.jsonl and crawl_stats.csv disagree for {mismatched}; re-run scripts/export_raw.py")
    mapping, rules = load_mapping(args.mapping), load_rules(args.rules)
    dedup_config = json.loads(args.dedup_config.read_text(encoding="utf-8"))
    decisions = read_csv(args.decisions) if args.decisions.exists() else []
    bad = [d for d in decisions if d["decision"] not in {"link", "separate"}]
    if bad:
        parser.error(f"dedup decisions must be link|separate: {bad[:3]}")

    valid, rejected, issues, fixes, pool_sizes = [], [], [], [], {}
    for source in sources:
        pool = pools.get(source, [])
        pool_sizes[source] = len(pool)
        ok, bad_rows, found, fixed = check_and_fix(pool, source, sources[source]["domain"], rules, mapping)
        for record in ok:
            record["outlet_group"], record["source_run_id"] = sources[source]["outlet_group"], crawl_stats[source]["run_id"]
        valid += ok
        rejected += bad_rows
        issues += found
        fixes += fixed
    valid.sort(key=lambda r: (r["source"], r["record_id"]))
    unseen = {s for s, row in sources.items() if row["outlet_group"] == "unseen"}
    auto, review, template, dedup_notes = apply_dedup(valid, dedup_config, decisions, unseen)
    by_source, by_category = funnel(sources, crawl_stats, pool_sizes, valid, mapping)

    interim.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    write_csv(interim / "issues.csv", issues, ISSUE_FIELDS)
    write_csv(interim / "rejected.csv", rejected, ["record_id", "source", "reject_issues", "reject_reasons", "url",
                                                   "title", "sapo", "category", "published_at", "crawled_at",
                                                   "extraction_status"])
    write_csv(interim / "fixes.csv", fixes, FIX_FIELDS)
    write_csv(interim / "dedup_pairs.csv", auto, PAIR_FIELDS + ["applied", "blocked_reason", "title_a", "title_b"])
    write_csv(interim / "dedup_review.csv", sorted(review, key=lambda r: (r["rule"], -r["jc3"], r["record_id_a"],
                                                                          r["record_id_b"])), REVIEW_FIELDS)
    write_csv(interim / "template_series.csv", sorted(template, key=lambda r: (r["series_id"], -r["jc3"])),
              ["series_id", "source_a"] + PAIR_FIELDS[:2] + ["jc3", "number_conflict", "title_a", "title_b"])
    frame = pd.DataFrame([{c: r.get(c) for c in CLEAN_COLUMNS} for r in valid], columns=list(CLEAN_COLUMNS))
    frame["dup_group_size"] = frame["dup_group_size"].astype("int64")
    frame["is_dup_representative"] = frame["is_dup_representative"].astype(bool)
    frame["has_sapo"] = frame["has_sapo"].astype(bool)
    frame.to_parquet(parquet, index=False)

    rejection_counts = []
    reject_issue = Counter((i["source"], i["issue"]) for i in issues if i["decision"] == "reject")
    rule_hits = Counter((i["source"], i["reason"].split(":")[0]) for i in issues if i["issue"] == "non_headline_title")
    for (source, issue), count in sorted(reject_issue.items()):
        if issue != "non_headline_title":  # broken down per rule below
            rejection_counts.append({"source": source, "issue": issue, "records": count})
    for (source, rule), count in sorted(rule_hits.items()):
        rejection_counts.append({"source": source, "issue": f"non_headline_title:{rule}", "records": count})
    write_csv(report_dir / "funnel_by_source.csv", by_source, list(by_source[0]))
    write_csv(report_dir / "funnel_by_source_category.csv", by_category,
              ["source", "category", "downloaded", "valid", "after_dedup", "with_sapo"])

    groups = Counter(r["dup_group"] for r in valid)
    group_sources = {}
    for r in valid:
        group_sources.setdefault(r["dup_group"], set()).add(r["source"])
    excluded = Counter(r["dedup_excluded_reason"] for r in valid if r["dedup_excluded_reason"])
    summary = {
        "configs": {"mapping": shown(args.mapping), "dedup_decisions": len(decisions)},
        "totals": {k: sum(r[k] for r in by_source) for k in ("downloaded", "downloaded_without_title", "selected",
                                                              "valid", "rejected", "dedup_excluded", "after_dedup",
                                                              "with_sapo")},
        "issue_counts": {f"{k[0]}:{k[1]}": v for k, v in sorted(Counter((i["issue"], i["decision"]) for i in issues).items())},
        "fix_counts": dict(Counter(f["field"] for f in fixes)),
        "dedup": {"groups": len(groups), "multi_member_groups": sum(1 for n in groups.values() if n > 1),
                  "cross_source_groups": sum(1 for g, s in group_sources.items() if len(s) > 1),
                  "largest_group": max(groups.values(), default=0),
                  "auto_links": sum(1 for p in auto if p["applied"]),
                  "auto_link_rules": dict(Counter(p["rule"] for p in auto if p["applied"])),
                  "excluded": sum(excluded.values()), "excluded_duplicate_of_unseen": excluded["duplicate_of_unseen"],
                  "excluded_duplicate_in_group": excluded["duplicate_in_group"],
                  "blocked_links": sum(1 for p in auto if not p["applied"]),
                  "blocked_reasons": dict(Counter(p["blocked_reason"] for p in auto if not p["applied"])),
                  "review_pairs": len(review), "review_rules": dict(sorted(Counter(p["rule"] for p in review).items())),
                  "template_pairs": len(template),
                  "template_series": len({p["series_id"] for p in template}), "notes": dedup_notes},
        "category": {"valid_unmapped": sum(1 for r in valid if r["category"] is None),
                     "valid_unapproved_mapping": sum(1 for r in valid if r["category"] and
                                                     r["category_mapping_status"] != "approved"),
                     "mapping_status_counts": dict(Counter(r["category_mapping_status"] or "unmapped" for r in valid))},
    }
    created = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    (report_dir / "data_report.md").write_text(render(created, crawl_rows, summary, by_source, rejection_counts),
                                               encoding="utf-8")
    print(json.dumps({"totals": summary["totals"], "clean_parquet": shown(parquet),
                      "report": shown(report_dir / "data_report.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
