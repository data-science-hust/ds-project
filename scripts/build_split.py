"""Build the split (plan 1.2): manifest, reserve list, blind annotation tables, split report.

Reads data/processed/articles_clean.parquet and data/interim/{template_series,dedup_review,dedup_pairs}.csv
(from scripts/build_clean.py; never modified), configs/checking/dedup_decisions.csv and the seed in
configs/crawling/run.json. Near-duplicate pairs that clean did not merge share a split_group (seen-unseen
pairs not decided 'separate': the seen record is held in reserve). Writes
data/manifests/{split_manifest,reserve_list,annotation_map}.csv, data/annotations/blind/*.csv and
reports/split_{report.md,counts.csv}; existing outputs are replaced only with --replace. The build runs
twice in memory and must give byte-identical outputs. annotation_assignment.csv (plan 2.2) is not built here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ds_project.splitting.report import render, share_deviation, split_count_rows
from ds_project.splitting.split import (BLIND_FIELDS, MANIFEST_FIELDS, MAP_FIELDS, PAIR_KINDS, RESERVE_FIELDS,
                                        SplitError, allocate, annotation_tables, build_split_groups, eligible_pool,
                                        manifest_rows, record_sides, unmerged_pairs, verify)

OUTPUTS = {"manifest": "data/manifests/split_manifest.csv", "reserve": "data/manifests/reserve_list.csv",
           "map": "data/manifests/annotation_map.csv", "blind_dev": "data/annotations/blind/blind_dev.csv",
           "blind_rest": "data/annotations/blind/blind_train_subset_test.csv",
           "report": "reports/split_report.md", "counts": "reports/split_counts.csv"}
RECORD_COLUMNS = ("record_id", "source", "outlet_group", "title", "category", "category_mapping_status", "dup_group",
                  "is_dup_representative", "dedup_excluded_reason", "has_sapo")


def shown(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)).replace("\\", "/") if path.is_relative_to(ROOT) else str(path)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def csv_text(rows, fields):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def read_records(path):
    frame = pd.read_parquet(path)
    missing = [c for c in RECORD_COLUMNS if c not in frame.columns]
    if missing:
        raise SplitError(f"{shown(path)} lacks columns {missing}; re-run scripts/build_clean.py")
    frame = frame[list(RECORD_COLUMNS)].astype(object).where(frame[list(RECORD_COLUMNS)].notna(), None)
    records = frame.to_dict("records")
    for r in records:
        r["is_dup_representative"], r["has_sapo"] = bool(r["is_dup_representative"]), bool(r["has_sapo"])
    return records


def pair_stats(records, listed, pairs, separate, manifest, held):
    """Per pair kind: rows listed, unmerged pairs, linked in a split_group, seen-unseen held / separate."""
    outlet = record_sides(records)
    group_of = {r["record_id"]: r["dup_group"] for r in records}
    split_of = {group_of[r["record_id"]]: r["split"] for r in manifest}
    stats = {kind: Counter(listed=len(listed.get(kind, ()))) for kind in PAIR_KINDS}
    for p in pairs:
        a, b, key = p["record_id_a"], p["record_id_b"], stats[p["kind"]]
        decided = tuple(sorted((a, b))) in separate
        key["unmerged"] += 1
        key["separate"] += decided
        if outlet[a] == outlet[b]:
            key["linked"] += 1
        else:
            key["cross_outlet_separate" if decided else "cross_outlet_held"] += 1
        a, b = split_of.get(group_of[a]), split_of.get(group_of[b])
        key["split_apart"] += bool(a and b and a != b)
    return {"by_kind": stats, "held_dup_groups": len(held)}


def build(records, pairs_by_kind, decisions, sources, seed, inputs):
    """-> ({output key: text}, summary). Pure function of its inputs."""
    unknown = sorted({r["source"] for r in records} - set(sources))
    wrong_outlet = sorted({r["source"] for r in records if r["source"] in sources
                           and r["outlet_group"] != sources[r["source"]]["outlet_group"]})
    if unknown or wrong_outlet:
        raise SplitError(f"clean table disagrees with sources.csv: unknown {unknown}, outlet_group {wrong_outlet}")
    pool = eligible_pool(records)
    separate = {tuple(sorted((d["record_id_a"], d["record_id_b"]))) for d in decisions if d["decision"] == "separate"}
    pairs = unmerged_pairs(records, pairs_by_kind)
    split_group_of, held = build_split_groups(records, pairs, separate)
    assignment, subset, reserve, _, units, selectable = allocate(pool, split_group_of, sources, seed, held)
    manifest = manifest_rows(pool, assignment, subset, split_group_of, sources, seed)
    titles = {r["record_id"]: r["title"] for r in pool}
    annotation_map, blind_dev, blind_rest = annotation_tables(manifest, titles, seed)
    checks = verify(records, pool, manifest, reserve, annotation_map, (blind_dev, blind_rest), sources,
                    split_group_of, pairs, separate, held)
    checks.append("Xác định: build hai lần từ cùng input cho output giống hệt từng byte (khác thì script dừng, "
                  "không ghi file)")
    multi = {u: m for u, m in units.items() if len(m) > 1}
    unit_split = {r["split_group"]: r["split"] for r in manifest}
    placement = Counter()
    for unit, members in multi.items():
        placement[(unit_split.get(unit, "reserve"), "groups")] += 1
        placement[(unit_split.get(unit, "reserve"), "records")] += len(members)
    template = pairs_by_kind.get("template_series", ())
    groups = {"template_pairs": len(template), "template_series": len({p["series_id"] for p in template}),
              "pairs": pair_stats(records, pairs_by_kind, pairs, separate, manifest, held), "units": len(units), "multi": len(multi), "multi_records": sum(len(m) for m in multi.values()),
              "sizes": Counter(len(m) for m in multi.values()),
              "multi_category": sum(len({r["category"] for r in m}) > 1 for m in multi.values()),
              "multi_source": sum(len({r["source"] for r in m}) > 1 for m in multi.values()),
              "placement": {s: (placement[(s, "groups")], placement[(s, "records")])
                            for s in sorted({k[0] for k in placement})}}
    context = {"sources": sources, "pool": pool, "selectable": selectable, "manifest": manifest, "reserve": reserve,
               "seed": seed,
               "inputs": inputs, "valid": Counter(r["source"] for r in records), "groups": groups,
               "deviation": share_deviation(sources, pool, manifest, selectable), "checks": checks,
               "blind_dev": len(blind_dev), "blind_rest": len(blind_rest), "mapped": len(annotation_map),
               "multiline_titles": sum("\n" in b["title"] or "\r" in b["title"] for b in blind_dev + blind_rest)}
    texts = {"manifest": csv_text(manifest, MANIFEST_FIELDS), "reserve": csv_text(reserve, RESERVE_FIELDS),
             "map": csv_text(annotation_map, MAP_FIELDS), "blind_dev": csv_text(blind_dev, BLIND_FIELDS),
             "blind_rest": csv_text(blind_rest, BLIND_FIELDS), "report": render(context),
             "counts": csv_text(split_count_rows(sources, pool, manifest, reserve),
                                ["source", "category", "split", "records"])}
    summary = {"selected": len(manifest), "by_split": dict(Counter(r["split"] for r in manifest)),
               "human_subset": len(subset), "human_scope": len(annotation_map), "reserve": len(reserve),
               "null_category": sum(r["category"] is None for r in pool), "multi_record_split_groups": len(multi),
               "held_for_dedup_review": sum(r["hold_reason"] == "pending_dedup_review_with_unseen" for r in reserve),
               "max_share_deviation_pp": round(100 * max((v for d in context["deviation"].values() for v in d.values()),
                                                         default=0), 3)}
    return texts, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replace", action="store_true", help="Overwrite existing manifest/blind/report outputs")
    parser.add_argument("--clean", type=Path, default=ROOT / "data/processed/articles_clean.parquet")
    parser.add_argument("--template-series", type=Path, default=ROOT / "data/interim/template_series.csv")
    parser.add_argument("--dedup-review", type=Path, default=ROOT / "data/interim/dedup_review.csv")
    parser.add_argument("--dedup-pairs", type=Path, default=ROOT / "data/interim/dedup_pairs.csv")
    parser.add_argument("--dedup-decisions", type=Path, default=ROOT / "configs/checking/dedup_decisions.csv")
    parser.add_argument("--sources", type=Path, default=ROOT / "configs/crawling/sources.csv")
    parser.add_argument("--run-config", type=Path, default=ROOT / "configs/crawling/run.json",
                        help="Source of the split seed")
    parser.add_argument("--out-root", type=Path, default=ROOT,
                        help="Base for data/manifests, data/annotations/blind, reports (tests use a temp dir)")
    args = parser.parse_args(argv)

    paths = {key: args.out_root / rel for key, rel in OUTPUTS.items()}
    existing = [shown(p) for p in paths.values() if p.exists()]
    if existing and not args.replace:
        parser.error(f"outputs exist ({', '.join(existing[:3])}{', ...' if len(existing) > 3 else ''}); "
                     "pass --replace to overwrite")
    pair_files = {"template_series": args.template_series, "dedup_review": args.dedup_review,
                  "dedup_blocked": args.dedup_pairs}
    for path in (args.clean, *pair_files.values(), args.dedup_decisions):
        if not path.exists():
            parser.error(f"{path} not found; run scripts/build_clean.py first")
    seed = json.loads(args.run_config.read_text(encoding="utf-8"))["seed"]
    sources = {row["source"]: row for row in read_csv(args.sources)}
    inputs = [(shown(p), hashlib.sha256(p.read_bytes()).hexdigest())
              for p in (args.clean, *pair_files.values(), args.dedup_decisions)]

    def load():
        pairs = {kind: read_csv(path) for kind, path in pair_files.items()}
        # dedup_pairs.csv: only links the clean step refused (applied=False) can leave two records unmerged
        pairs["dedup_blocked"] = [p for p in pairs["dedup_blocked"] if p.get("applied") != "True"]
        return read_records(args.clean), pairs, read_csv(args.dedup_decisions)

    try:
        texts, summary = build(*load(), sources, seed, inputs)
        again, _ = build(*load(), sources, seed, inputs)
    except SplitError as exc:
        parser.error(str(exc))
    differing = [OUTPUTS[k] for k in texts if texts[k] != again[k]]
    if differing:
        parser.error(f"two builds from the same inputs differ: {differing}")
    for key, text in texts.items():
        paths[key].parent.mkdir(parents=True, exist_ok=True)
        with paths[key].open("w", encoding="utf-8-sig" if key != "report" else "utf-8", newline="") as stream:
            stream.write(text)
    summary["seed"] = seed
    summary["outputs"] = [shown(p) for p in paths.values()]
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
