import contextlib
import csv
import importlib.util
import io
import json
import re
import sys
import os
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ds_project.splitting.report import share_deviation
from ds_project.splitting.split import (HOLD_UNSEEN, MANIFEST_FIELDS, NO_CATEGORY, RESERVE_FIELDS, SplitError,
                                        allocate, annotation_tables, build_split_groups, eligible_pool,
                                        largest_remainder, manifest_rows, plan_quotas, unmerged_pairs, verify)

spec = importlib.util.spec_from_file_location("build_split", ROOT / "scripts/build_split.py")
BUILD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BUILD)

SOURCE_FIELDS = ["source", "domain", "outlet_group", "train_target", "dev_target", "test_seen_target",
                 "test_unseen_target", "human_train_subset_target", "min_clean_articles"]
SOURCES = {"s1": {"outlet_group": "seen", "train_target": "20", "dev_target": "6", "test_seen_target": "4",
                  "test_unseen_target": "0", "human_train_subset_target": "8"},
           "s2": {"outlet_group": "seen", "train_target": "20", "dev_target": "6", "test_seen_target": "4",
                  "test_unseen_target": "0", "human_train_subset_target": "8"},
           "u1": {"outlet_group": "unseen", "train_target": "0", "dev_target": "0", "test_seen_target": "0",
                  "test_unseen_target": "5", "human_train_subset_target": "0"}}


def rec(rid, source, category="A", sapo=True, dup_group=None, rep=True, excluded=None, status="approved"):
    return {"record_id": rid, "source": source, "outlet_group": SOURCES[source]["outlet_group"],
            "title": f"Tiêu đề {rid}", "category": category,
            "category_mapping_status": status if category is not None else None,
            "dup_group": dup_group or f"dg_{rid}", "is_dup_representative": rep, "dedup_excluded_reason": excluded,
            "has_sapo": sapo, "sapo": f"Sapo {rid}" if sapo else None, "url": f"https://{source}.vn/{rid}.html",
            "published_at": "2026-05-20T08:00:00+07:00"}


def dataset():
    records = []
    for source in ("s1", "s2"):
        for i in range(40):
            records.append(rec(f"{source}_{i:02d}", source, "A" if i < 24 else "B", sapo=i % 7 != 3))
    records += [rec(f"u1_{i:02d}", "u1", "C" if i < 6 else "D") for i in range(10)]
    records.append(rec("s1_null", "s1", None))
    records.append(rec("s1_dup", "s1", dup_group="dg_s1_00", rep=False, excluded="duplicate_in_group"))
    pairs = [{"series_id": "ts_1", "record_id_a": "s1_01", "record_id_b": "s1_02"},
             {"series_id": "ts_1", "record_id_a": "s1_02", "record_id_b": "s1_03"},
             {"series_id": "ts_2", "record_id_a": "s1_dup", "record_id_b": "s1_30"}]  # via a non-representative
    return records, pairs


def run(records, pairs, seed=7, separate=frozenset()):
    pool = eligible_pool(records)
    groups, held = build_split_groups(records, pairs, separate)
    assignment, subset, reserve, quotas, units, _ = allocate(pool, groups, SOURCES, seed, held)
    return pool, groups, assignment, subset, reserve, quotas


def built(records, pairs, seed=7, separate=frozenset()):
    """Everything verify() needs, from a full in-memory build."""
    pool = eligible_pool(records)
    groups, held = build_split_groups(records, pairs, separate)
    assignment, subset, reserve, _, _, _ = allocate(pool, groups, SOURCES, seed, held)
    manifest = manifest_rows(pool, assignment, subset, groups, SOURCES, seed)
    mapping, dev, rest = annotation_tables(manifest, {r["record_id"]: r for r in pool}, seed)
    return {"records": records, "pool": pool, "manifest": manifest, "reserve": reserve, "annotation_map": mapping,
            "blind": (dev, rest), "sources": SOURCES, "split_group_of": groups, "pairs": pairs,
            "separate": separate, "held": held}


class QuotaTests(unittest.TestCase):
    def test_largest_remainder_is_exact_and_breaks_ties_by_name(self):
        self.assertEqual(largest_remainder(6, {"A": 24, "B": 16}), {"A": 4, "B": 2})
        self.assertEqual(largest_remainder(1, {"B": 1, "A": 1}), {"A": 1, "B": 0})
        self.assertEqual(largest_remainder(0, {"A": 3}), {"A": 0})
        shares = largest_remainder(175, {"x": 1013, "y": 2, "z": 377, "w": 1999})
        self.assertEqual(sum(shares.values()), 175)
        with self.assertRaises(SplitError):
            largest_remainder(3, {})

    def test_target_without_selectable_records_names_source_and_split(self):
        records, _ = dataset()
        sources = {**SOURCES, "u2": {**SOURCES["u1"]}}
        with self.assertRaisesRegex(SplitError, "u2/test_unseen: target 5 but 0 selectable"):
            plan_quotas(eligible_pool([r for r in records if r["category"] is not None]), sources)

    def test_unapproved_mapping_is_refused_and_null_category_allowed(self):
        records, _ = dataset()
        self.assertEqual(len(eligible_pool(records)), 91)  # non-representative dropped, null category kept
        with self.assertRaises(SplitError):
            eligible_pool(records + [rec("s2_bad", "s2", status="proposed")])


class GroupTests(unittest.TestCase):
    def test_template_pairs_union_dup_groups_even_through_non_representatives(self):
        records, pairs = dataset()
        groups, held = build_split_groups(records, pairs)
        self.assertEqual(len({groups["dg_s1_01"], groups["dg_s1_02"], groups["dg_s1_03"]}), 1)
        self.assertEqual(groups["dg_s1_00"], groups["dg_s1_30"])  # s1_dup lives in dg_s1_00
        self.assertNotEqual(groups["dg_s1_04"], groups["dg_s1_01"])
        self.assertEqual(held, set())
        self.assertEqual(groups, build_split_groups(list(reversed(records)), list(reversed(pairs)))[0])
        with self.assertRaises(SplitError):
            build_split_groups(records, [{"series_id": "x", "record_id_a": "s1_01", "record_id_b": "missing"}])

    def test_review_pairs_are_kept_together_and_seen_side_of_unseen_pair_is_held(self):
        records, template = dataset()
        pairs = unmerged_pairs(records, {"template_series": template,
                                         "dedup_review": [{"record_id_a": "s1_05", "record_id_b": "s2_07"},
                                                          {"record_id_a": "s1_08", "record_id_b": "u1_02"},
                                                          {"record_id_a": "s1_01", "record_id_b": "s1_02"}],
                                         "dedup_blocked": [{"record_id_a": "s2_09", "record_id_b": "u1_03"}]})
        self.assertEqual(len(pairs), 6)  # s1_01/s1_02 already listed as template, kept once
        self.assertEqual(Counter(p["kind"] for p in pairs),
                         {"template_series": 3, "dedup_review": 2, "dedup_blocked": 1})
        groups, held = build_split_groups(records, pairs, {("s2_09", "u1_03")})
        self.assertEqual(groups["dg_s1_05"], groups["dg_s2_07"])
        self.assertEqual(held, {"dg_s1_08"})  # s2_09/u1_03 was decided separate: not held
        pool, _, assignment, _, reserve, _ = run(records, pairs, separate={("s2_09", "u1_03")})
        self.assertEqual(assignment.get("s1_05", "reserve"), assignment.get("s2_07", "reserve"))
        self.assertNotIn("s1_08", assignment)
        row = next(r for r in reserve if r["record_id"] == "s1_08")
        self.assertEqual((row["hold_reason"], row["category"]), (HOLD_UNSEEN, "A"))

    def test_seen_copy_inside_unseen_dup_group_counts_as_unseen(self):
        records, _ = dataset()
        records.append(rec("s2_copy", "s2", dup_group="dg_u1_04", rep=False, excluded="duplicate_of_unseen"))
        groups, held = build_split_groups(records, [{"record_id_a": "s1_10", "record_id_b": "s2_copy"}])
        self.assertNotEqual(groups["dg_s1_10"], groups["dg_u1_04"])
        self.assertEqual(held, {"dg_s1_10"})


class AllocationTests(unittest.TestCase):
    def test_exact_quotas_strata_groups_sapo_and_reserve(self):
        records, pairs = dataset()
        pool, groups, assignment, subset, reserve, quotas = run(records, pairs)
        by_id = {r["record_id"]: r for r in pool}
        counts = Counter((by_id[i]["source"], s) for i, s in assignment.items())
        self.assertEqual(counts, {("s1", "train"): 20, ("s1", "dev"): 6, ("s1", "test_seen"): 4, ("s2", "train"): 20,
                                  ("s2", "dev"): 6, ("s2", "test_seen"): 4, ("u1", "test_unseen"): 5})
        cells = Counter((by_id[i]["source"], by_id[i]["category"], s) for i, s in assignment.items())
        for (source, split, category), n in quotas.items():
            if split != "human_subset":
                self.assertEqual(cells[(source, category, split)], n)
        self.assertEqual((quotas[("s1", "dev", "A")], quotas[("s1", "dev", "B")]), (4, 2))
        self.assertTrue(all(by_id[i]["has_sapo"] for i, s in assignment.items() if s == "dev"))
        self.assertEqual(Counter(by_id[i]["source"] for i in subset), {"s1": 8, "s2": 8})
        self.assertTrue(all(assignment[i] == "train" for i in subset))
        series = {assignment.get(f"s1_0{i}", "reserve") for i in (1, 2, 3)}
        self.assertEqual(len(series), 1)
        self.assertEqual(assignment.get("s1_00", "reserve"), assignment.get("s1_30", "reserve"))
        self.assertNotIn("s1_null", assignment)
        null_row = [r for r in reserve if r["record_id"] == "s1_null"]
        self.assertEqual((null_row[0]["category"], null_row[0]["reserve_rank"]), (NO_CATEGORY, 1))
        self.assertEqual(len(reserve) + len(assignment), len(pool))
        ranks = {}
        for r in reserve:
            ranks.setdefault((r["source"], r["category"]), []).append(r["reserve_rank"])
        self.assertTrue(all(v == list(range(1, len(v) + 1)) for v in ranks.values()))
        self.assertEqual(set(reserve[0]), set(RESERVE_FIELDS))
        self.assertTrue(all(r["has_sapo"] == by_id[r["record_id"]]["has_sapo"] for r in reserve))

    def test_deterministic_for_a_seed_and_changes_with_it(self):
        records, pairs = dataset()
        first, second, other = run(records, pairs), run(list(reversed(records)), pairs), run(records, pairs, seed=8)
        self.assertEqual(first[2:5], second[2:5])
        self.assertNotEqual(first[2], other[2])

    def test_dev_without_enough_sapo_fails_loudly(self):
        records, pairs = dataset()
        records = [{**r, "has_sapo": False} if r["source"] == "s2" and r["category"] == "B" else r for r in records]
        with self.assertRaisesRegex(SplitError, "s2/B: dev quota 2 > 0 records with sapo"):
            run(records, pairs)

    def test_group_too_large_for_a_cell_is_skipped_not_split(self):
        records, _ = dataset()
        records += [rec(f"s2_x{i:02d}", "s2", "B") for i in range(20)]  # s2/B: 36 records, train quota 12
        pairs = [{"series_id": "big", "record_id_a": "s2_x00", "record_id_b": f"s2_x{i:02d}"} for i in range(1, 13)]
        pool, groups, assignment, subset, reserve, quotas = run(records, pairs)  # a 13-record series
        self.assertEqual(quotas[("s2", "train", "B")], 12)
        self.assertEqual({assignment.get(f"s2_x{i:02d}", "reserve") for i in range(13)}, {"reserve"})
        self.assertEqual(Counter(s for i, s in assignment.items() if i.startswith("s2")),
                         {"train": 20, "dev": 6, "test_seen": 4})
        records, _ = dataset()  # s2/B (16 records) as five triples + one single: dev quota 2 cannot be met
        triples = [{"series_id": f"t{i}", "record_id_a": f"s2_{i}", "record_id_b": f"s2_{i + d}"}
                   for i in range(24, 39, 3) for d in (1, 2)]
        with self.assertRaisesRegex(SplitError, "cannot fill dev exactly.*s2/B"):
            run(records, triples)

    def test_split_group_mixing_seen_and_unseen_is_refused(self):
        records, pairs = dataset()
        pool = eligible_pool(records)
        groups, _ = build_split_groups(records, pairs)
        groups["dg_u1_00"] = groups["dg_s1_06"]
        with self.assertRaisesRegex(SplitError, "mix seen and unseen"):
            allocate(pool, groups, SOURCES, 7)

    def test_annotation_tables_show_only_title_and_category(self):
        records, pairs = dataset()
        pool, groups, assignment, subset, _, _ = run(records, pairs)
        manifest = manifest_rows(pool, assignment, subset, groups, SOURCES, 7)
        mapping, dev, rest = annotation_tables(manifest, {r["record_id"]: r for r in pool}, 7)
        self.assertEqual((len(mapping), len(dev), len(rest)), (8 * 2 + 6 * 2 + 4 * 2 + 5, 12, 29))
        self.assertTrue(all(re.fullmatch(r"A\d{5}", m["annotation_id"]) for m in mapping))
        self.assertEqual(len({m["annotation_id"] for m in mapping}), len(mapping))
        record = {m["annotation_id"]: next(r for r in pool if r["record_id"] == m["record_id"]) for m in mapping}
        self.assertTrue(all((b["title"], b["category"]) == (record[b["annotation_id"]]["title"],
                                                            record[b["annotation_id"]]["category"]) for b in dev + rest))
        self.assertEqual({m["split"] for m in mapping if m["annotation_id"] in {d["annotation_id"] for d in dev}},
                         {"dev"})


class VerifyTests(unittest.TestCase):
    def setUp(self):
        records, template = dataset()
        pairs = unmerged_pairs(records, {"template_series": template,
                                         "dedup_review": [{"record_id_a": "s1_05", "record_id_b": "s2_07"},
                                                          {"record_id_a": "s1_08", "record_id_b": "u1_02"}]})
        self.ctx = built(records, pairs)

    def fails(self, pattern, **changes):
        with self.assertRaisesRegex(SplitError, pattern):
            verify(**{**self.ctx, **changes})

    def test_clean_build_passes_every_check(self):
        self.assertEqual(len(verify(**self.ctx)), 10)

    def test_each_check_fires(self):
        manifest, reserve, pool = self.ctx["manifest"], self.ctx["reserve"], self.ctx["pool"]
        first = manifest[0]
        self.fails("quota", manifest=manifest[1:], reserve=reserve + [{**first, "has_sapo": True}])
        self.fails("đúng một lần", reserve=reserve[1:])
        train = next(r for r in manifest if r["split"] == "train")
        test = next(r for r in manifest if r["split"] == "test_seen" and r["source"] == train["source"]
                    and r["category"] == train["category"])
        joined = [{**r, "split_group": train["split_group"]} if r is test else r for r in manifest]
        self.fails("một split", manifest=joined)
        pair = {"kind": "dedup_review", "record_id_a": train["record_id"], "record_id_b": test["record_id"]}
        self.fails("hai split khác nhau", pairs=[pair])
        dev = next(r for r in manifest if r["split"] == "dev")
        self.fails("có sapo", pool=[{**r, "has_sapo": False} if r["record_id"] == dev["record_id"] else r
                                    for r in pool])
        self.fails("có sapo", reserve=[{**r, "has_sapo": not r["has_sapo"]} for r in reserve])
        moved = next(r for r in manifest if r["is_human_subset"] and r["source"] == dev["source"])
        self.fails("Subset human nằm trong train",
                   manifest=[{**r, "is_human_subset": r is dev} if r is dev or r is moved else r for r in manifest])
        self.fails("không vào split", held={dev["dup_group"]})
        unseen_dup = next(r for r in manifest if r["split"] == "test_unseen")["dup_group"]
        self.fails("chung dup_group/split_group với bài unseen",
                   split_group_of={**self.ctx["split_group_of"], unseen_dup: dev["split_group"]})
        self.fails("Nguồn unseen chỉ ở test_unseen",
                   sources={**SOURCES, "u1": {**SOURCES["u1"], "outlet_group": "seen"}},
                   split_group_of={**self.ctx["split_group_of"]})
        mapping = self.ctx["annotation_map"]
        self.fails("annotation_id duy nhất",
                   annotation_map=[*mapping[:-1], {**mapping[-1], "annotation_id": mapping[0]["annotation_id"]}])

    def test_share_deviation_skips_columns_with_target_zero(self):
        ctx = self.ctx
        manifest = [{**r, "is_human_subset": False} if r["source"] == "s1" else r for r in ctx["manifest"]]
        selectable = {r["record_id"] for r in ctx["pool"] if r["category"] is not None}
        deviation = share_deviation(SOURCES, ctx["pool"], manifest, selectable)
        self.assertNotIn("human_subset", deviation["s1"])
        self.assertIn("human_subset", deviation["s2"])
        self.assertEqual(set(deviation["u1"]), {"test_unseen"})


def quiet(function, argv):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return function(argv)


class ScriptTests(unittest.TestCase):
    def write_inputs(self, root, records, pairs, sources=SOURCES):
        pd.DataFrame(records).to_parquet(root / "clean.parquet", index=False)

        def write(name, fields, rows):
            with (root / name).open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

        write("series.csv", ["series_id", "source_a", "record_id_a", "record_id_b", "jc3"],
              ({**p, "source_a": "s1", "jc3": 0.8} for p in pairs))
        write("review.csv", ["record_id_a", "record_id_b", "rule"],
              [{"record_id_a": "s1_05", "record_id_b": "s2_07", "rule": "review_band"},
               {"record_id_a": "s1_08", "record_id_b": "u1_02", "rule": "date_gap"}])
        write("pairs.csv", ["record_id_a", "record_id_b", "applied", "blocked_reason"],
              [{"record_id_a": "s1_00", "record_id_b": "s1_dup", "applied": "True", "blocked_reason": ""},
               {"record_id_a": "s2_09", "record_id_b": "s2_10", "applied": "False",
                "blocked_reason": "separate_decision"}])
        write("decisions.csv", ["record_id_a", "record_id_b", "decision", "reviewer", "notes"],
              [{"record_id_a": "s2_09", "record_id_b": "s2_10", "decision": "separate"}])
        write("sources.csv", SOURCE_FIELDS, ({"source": s, "domain": f"{s}.vn", "min_clean_articles": 0, **row}
                                             for s, row in sources.items()))
        (root / "run.json").write_text(json.dumps({"seed": 11}), encoding="utf-8")
        return ["--clean", str(root / "clean.parquet"), "--template-series", str(root / "series.csv"),
                "--dedup-review", str(root / "review.csv"), "--dedup-pairs", str(root / "pairs.csv"),
                "--dedup-decisions", str(root / "decisions.csv"), "--sources", str(root / "sources.csv"),
                "--run-config", str(root / "run.json")]

    def test_outputs_are_byte_identical_and_need_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records, pairs = dataset()
            common = self.write_inputs(root, records, pairs)
            quiet(BUILD.main, [*common, "--out-root", str(root / "a")])
            quiet(BUILD.main, [*common, "--out-root", str(root / "b")])
            for rel in BUILD.OUTPUTS.values():
                self.assertEqual((root / "a" / rel).read_bytes(), (root / "b" / rel).read_bytes(), rel)
            with self.assertRaises(SystemExit):
                quiet(BUILD.main, [*common, "--out-root", str(root / "a")])
            quiet(BUILD.main, [*common, "--out-root", str(root / "a"), "--replace"])
            manifest = BUILD.read_csv(root / "a/data/manifests/split_manifest.csv")
            self.assertEqual(list(manifest[0]), list(MANIFEST_FIELDS))
            self.assertEqual(len(manifest), 65)
            self.assertEqual({r["seed"] for r in manifest}, {"11"})
            blind = BUILD.read_csv(root / "a/data/annotations/blind/blind_dev.csv")
            self.assertEqual(list(blind[0]), ["annotation_id", "title", "category"])
            table = BUILD.read_csv(root / "a/data/processed/dataset.csv")
            self.assertEqual(list(table[0]), list(BUILD.DATASET_FIELDS))
            self.assertEqual([r["record_id"] for r in table], [r["record_id"] for r in manifest])
            mapped = {r["annotation_id"]: r["record_id"] for r in BUILD.read_csv(root / "a/data/manifests/annotation_map.csv")}
            self.assertEqual({r["annotation_id"]: r["record_id"] for r in table if r["annotation_id"]}, mapped)
            self.assertEqual({r["record_id"] for r in table if r["split"] == "train" and r["is_human_subset"] == "False"},
                             {r["record_id"] for r in table if not r["annotation_id"]})
            report = (root / "a/reports/split_report.md").read_text(encoding="utf-8")
            self.assertIn("## 7. Kiểm tra (đều đạt)", report)
            self.assertIn("annotation_assignment.csv", report)
            reserve = BUILD.read_csv(root / "a/data/manifests/reserve_list.csv")
            self.assertEqual(list(reserve[0]), list(RESERVE_FIELDS))
            self.assertEqual([r["record_id"] for r in reserve if r["hold_reason"] == HOLD_UNSEEN], ["s1_08"])
            split_of = {r["record_id"]: r["split"] for r in manifest}
            self.assertEqual(split_of.get("s1_05", "reserve"), split_of.get("s2_07", "reserve"))
            self.assertEqual(split_of.get("s2_09", "reserve"), split_of.get("s2_10", "reserve"))

    def test_target_zero_columns_and_hash_seed_independence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records, pairs = dataset()
            sources = {**SOURCES, "s1": {**SOURCES["s1"], "human_train_subset_target": "0"}}
            common = self.write_inputs(root, records, pairs, sources)
            outputs = []
            for hash_seed in ("0", "98765"):
                out = root / f"out_{hash_seed}"
                env = {**os.environ, "PYTHONHASHSEED": hash_seed}
                subprocess.run([sys.executable, str(ROOT / "scripts/build_split.py"), *common, "--out-root", str(out)],
                               check=True, env=env, capture_output=True)
                outputs.append({rel: (out / rel).read_bytes() for rel in BUILD.OUTPUTS.values()})
            self.assertEqual(outputs[0], outputs[1])
            report = outputs[0]["reports/split_report.md"].decode("utf-8")
            self.assertRegex(report, r"\| s1 \| [\d,]+ \| – \|")  # subset target 0: no deviation shown

    def test_refuses_when_two_builds_differ(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records, pairs = dataset()
            common = self.write_inputs(root, records, pairs)
            real, calls = BUILD.build, []

            def flaky(*args):
                texts, summary = real(*args)
                calls.append(1)
                return {**texts, "report": texts["report"] + "x" * len(calls)}, summary

            with mock.patch.object(BUILD, "build", flaky), self.assertRaises(SystemExit):
                quiet(BUILD.main, [*common, "--out-root", str(root / "out")])
            self.assertEqual(len(calls), 2)
            self.assertFalse((root / "out").exists())

    def test_refuses_unapproved_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records, pairs = dataset()
            common = self.write_inputs(root, records + [rec("s2_bad", "s2", status="needs_review")], pairs)
            with self.assertRaises(SystemExit):
                quiet(BUILD.main, [*common, "--out-root", str(root / "out")])
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
