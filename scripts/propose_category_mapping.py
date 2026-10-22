"""Append proposed rows to category_mapping.csv for (source, category_raw) keys not yet listed.

Existing rows are never changed. dantri/nhandan use rule-based proposals; other
sources get an empty category with review_status=needs_review for manual mapping.
Keys come from data/raw/articles_raw.jsonl; --all-rows uses every downloaded article's
category from reports/crawl_stats.csv (downloaded_by_category).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ds_project.checking.categories import MAPPING_FIELDS, load_mapping, normalize_category, propose
from ds_project.checking.clean import read_crawl_stats, read_csv, read_raw


def new_rows(mapping, counts):
    """counts: Counter of (source, raw category). Returns proposals for unseen normalized keys."""
    keys = Counter()
    for (source, raw), count in counts.items():
        key = normalize_category(raw)
        if key is not None and (source, key) not in mapping:
            keys[(source, key)] += count
    rows = []
    for (source, key), count in sorted(keys.items()):
        row = propose(source, key)
        row["notes"] = "; ".join(filter(None, [row["notes"], f"auto-proposed, {count} records"]))
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-rows", action="store_true",
                        help="Use every downloaded article (reports/crawl_stats.csv), not only the raw dataset")
    parser.add_argument("--dry-run", action="store_true", help="Print proposals without writing")
    parser.add_argument("--mapping", type=Path, default=ROOT / "configs/crawling/category_mapping.csv")
    parser.add_argument("--sources", type=Path, default=ROOT / "configs/crawling/sources.csv")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/articles_raw.jsonl")
    parser.add_argument("--crawl-stats", type=Path, default=ROOT / "reports/crawl_stats.csv")
    args = parser.parse_args(argv)
    sources = {row["source"]: row for row in read_csv(args.sources)}
    counts = Counter()
    try:
        if args.all_rows:
            for source, stats in read_crawl_stats(read_csv(args.crawl_stats), sources).items():
                counts.update({(source, value): n for value, n in stats["downloaded_by_category"] if isinstance(value, str)})
        else:
            for source, pool in read_raw(args.raw, sources).items():
                counts.update((source, row.get("category")) for row in pool if isinstance(row.get("category"), str))
    except (OSError, ValueError, KeyError) as exc:
        parser.error(f"{exc}; run scripts/export_raw.py first")
    rows = new_rows(load_mapping(args.mapping), counts)
    if args.dry_run or not rows:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))
    else:
        text = args.mapping.read_text(encoding="utf-8-sig")
        with args.mapping.open("a", encoding="utf-8", newline="") as stream:
            if text and not text.endswith("\n"):
                stream.write("\n")
            csv.DictWriter(stream, fieldnames=MAPPING_FIELDS, lineterminator="\n").writerows(rows)
        load_mapping(args.mapping)  # re-validate
    print(json.dumps({"new_rows": len(rows), "by_source": dict(Counter(r["source"] for r in rows)),
                      "written": bool(rows) and not args.dry_run}, ensure_ascii=False))


if __name__ == "__main__":
    main()
