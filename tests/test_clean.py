import contextlib
import csv
import importlib.util
import io
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ds_project.checking import categories
from ds_project.checking.categories import load_mapping, lookup, normalize_category, propose
from ds_project.checking.clean import (check_and_fix, funnel, load_rules, mojibake_status, read_crawl_stats,
                                       read_raw, rule_hits)
from ds_project.checking.dedup import (choose_representatives, find_duplicates, group_id, jc3, norm_title,
                                       number_conflict, number_tokens)


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROPOSE, BUILD, EXPORT = (load_script("propose_category_mapping"), load_script("build_clean"),
                          load_script("export_raw"))
DAY = 86400.0
DEDUP = json.loads((ROOT / "configs/checking/dedup.json").read_text(encoding="utf-8"))
RULES = load_rules(ROOT / "configs/checking/rejection_rules.json")
MAP = {("kenh14", "Beauty & Fashion"): {"category": categories.LD, "review_status": "proposed"},
       ("kenh14", "Thế Giới Đó Đây"): {"category": categories.TG, "review_status": "needs_review"}}


def raw(rid, title, category="Beauty &amp; Fashion", sapo="Sapo", url=None, **extra):
    return {"record_id": rid, "source": "kenh14", "url": url or f"https://kenh14.vn/{rid}.chn", "title": title,
            "sapo": sapo, "category": category, "published_at": "2026-05-20T08:12:00+07:00",
            "crawled_at": "2026-10-21T06:52:00+00:00", "extraction_status": "success", **extra}


def rec(rid, source, title, published=0.0, sapo=None):
    return {"record_id": rid, "source": source, "title": title, "sapo": sapo, "published_sort": published}


class TechnicalFixTests(unittest.TestCase):
    def test_fixes_are_logged_and_title_wording_is_preserved(self):
        title = "  Tiêu đề GỐC?! … Giữ \"nguyên\"  "
        decomposed = "Thế Giới Đó Đây"
        valid, rejected, issues, fixes = check_and_fix(
            [raw("a", title, sapo=" Sapo  "), raw("b", "Bài hai", category=decomposed, sapo="   ")],
            "kenh14", "kenh14.vn", RULES, MAP)
        self.assertEqual(rejected, [])
        first, second = valid
        self.assertEqual(first["title"], title.strip())
        self.assertEqual(first["title_raw"], title)
        self.assertEqual(first["category_raw"], "Beauty & Fashion")
        self.assertEqual(first["category"], categories.LD)
        self.assertEqual(second["category_raw"], "Thế Giới Đó Đây")
        self.assertEqual(second["category_mapping_status"], "needs_review")
        self.assertIsNone(second["sapo"])
        self.assertFalse(second["has_sapo"])
        logged = {(f["record_id"], f["field"]): f for f in fixes}
        self.assertEqual(logged[("a", "title")]["before"], title)
        self.assertEqual(logged[("a", "category_raw")]["after"], "Beauty & Fashion")
        self.assertIn(("b", "category_raw"), logged)
        self.assertIn(("b", "sapo"), logged)
        self.assertIn("missing_sapo", {i["issue"] for i in issues if i["record_id"] == "b"})

    def test_record_level_rejections_keep_reasons(self):
        rows = [raw("ok", "Tiêu đề"), raw("ok", "Trùng ID"), raw("host", "Sai host", url="https://evil.vn/x"),
                raw("blank", "   "), raw("date", "Ngày lỗi", published_at="20/05/2026"),
                raw("moji", "Tin tá»©c má»›i"), raw("fail", "Lỗi", extraction_status="fetch_error")]
        valid, rejected, issues, _ = check_and_fix(rows, "kenh14", "kenh14.vn", RULES, MAP)
        self.assertEqual([r["record_id"] for r in valid], ["ok"])
        reasons = {r["record_id"] + ":" + r["reject_issues"] for r in rejected}
        for expected in ("ok:duplicate_record_id;duplicate_url", "host:unexpected_host", "blank:missing_value",
                         "date:invalid_published_at", "moji:mojibake_confirmed", "fail:extraction_not_usable"):
            self.assertIn(expected, reasons)

    def test_mojibake_only_flagged_when_not_confirmed(self):
        self.assertEqual(mojibake_status("Tin tá»©c"), "confirmed")
        self.assertEqual(mojibake_status("săn MÃ”"), "suspected")
        self.assertIsNone(mojibake_status("TOP 5 HÃNG XE ÂM NHẠC"))


class CategoryTests(unittest.TestCase):
    def test_unmapped_key_and_missing_category_never_default(self):
        self.assertEqual(lookup(MAP, "kenh14", "Không có"), (None, None, "key not in category_mapping.csv"))
        self.assertEqual(lookup(MAP, "kenh14", None)[0], None)
        valid, _, issues, _ = check_and_fix([raw("x", "Bài", category="Mục mới"), raw("y", "Bài 2", category="")],
                                            "kenh14", "kenh14.vn", RULES, MAP)
        self.assertEqual([r["category"] for r in valid], [None, None])
        self.assertEqual([r["category_raw"] for r in valid], ["Mục mới", None])
        self.assertEqual(sum(i["issue"] == "category_unmapped" for i in issues), 2)

    def test_committed_mapping_invariants(self):
        path = ROOT / "configs/crawling/category_mapping.csv"
        mapping = load_mapping(path)  # validates columns, unique keys, groups, statuses, approved rows
        for (source, key), row in mapping.items():
            self.assertEqual(normalize_category(key), key)
            self.assertIn(row["review_status"], categories.REVIEW_STATUSES)
            self.assertTrue(row["category"] is None or row["category"] in categories.GROUPS)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):  # keys are written already normalized
                self.assertEqual(row["category_raw"], normalize_category(row["category_raw"]))

    def test_empty_mapping_row_is_unmapped_and_approved_needs_category(self):
        mapping = {("yan", "Mới"): {"category": None, "review_status": "needs_review"}}
        self.assertEqual(lookup(mapping, "yan", "Mới")[:2], (None, None))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "m.csv"
            path.write_text("source,category_raw,category,review_status,notes\nyan,Mới,,approved,\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_mapping(path)

    def test_nhandan_and_dantri_rules(self):
        self.assertEqual(propose("dantri", "Thể thao, Bóng đá")["category"], categories.TT)
        self.assertEqual(propose("dantri", "Giải trí, Thời trang")["category"], categories.LD)
        unknown = propose("dantri", "Chuyên mục lạ, Con")
        self.assertEqual((unknown["category"], unknown["review_status"]), ("", "needs_review"))
        self.assertEqual(propose("nhandan", "Chính trị,Châu Âu")["category"], categories.TG)
        self.assertEqual(propose("nhandan", "Video,Y tế")["category"], categories.SK)
        self.assertEqual(propose("nhandan", "Tin chung,Hà Nội")["category"], categories.TS)
        self.assertEqual(propose("nhandan", "Chính trị, Đại hội Đảng toàn quốc lần thứ XIV")["category"], categories.TS)
        self.assertIn("unknown tokens", propose("nhandan", "Kinh tế,Mục mới")["notes"])
        self.assertEqual(propose("yan", "Mới")["category"], "")

    def test_generator_only_adds_missing_keys(self):
        rows = PROPOSE.new_rows(MAP, {("kenh14", "Beauty &amp; Fashion"): 3, ("dantri", "Du lịch, Khám phá"): 2})
        self.assertEqual([(r["source"], r["category_raw"], r["category"]) for r in rows],
                         [("dantri", "Du lịch, Khám phá", categories.DL)])


class RejectionRuleTests(unittest.TestCase):
    def hit(self, source, title, category=None):
        return [rule["id"] for rule in rule_hits(RULES, source, title, category)]

    def test_missing_sapo_policy_only_rejects_its_source(self):
        rows = [raw("e1", "Có sapo", sapo="Lead"), raw("e2", "Không sapo", sapo=None)]
        for source, domain in (("eva", "eva.vn"), ("kenh14", "kenh14.vn")):
            pool = [{**r, "source": source, "url": f"https://{domain}/{r['record_id']}-12345.html"} for r in rows]
            valid, rejected, issues, _ = check_and_fix(pool, source, domain, RULES, MAP)
            if source == "eva":
                self.assertEqual([r["record_id"] for r in valid], ["e1"])
                self.assertEqual(rejected[0]["reject_issues"], "missing_sapo_policy")
                self.assertNotIn("non_headline_title", {i["issue"] for i in issues})
            else:
                self.assertEqual(len(valid), 2)
                self.assertIn("missing_sapo", {i["issue"] for i in issues})

    def test_non_headline_rules_are_conservative(self):
        self.assertEqual(self.hit("nhandan", "Tin tức"), ["nhandan_generic_title"])
        self.assertEqual(self.hit("nhandan", "Tin tức mới"), [])
        self.assertEqual(self.hit("nhandan", "BÁO THỜI NAY SỐ 1699", "Thời nay,Báo in"), ["nhandan_print_edition_index"])
        self.assertEqual(self.hit("nhandan", "BÁO THỜI NAY SỐ 1699", "Văn hóa"), [])
        self.assertTrue(self.hit("thanhnien", "Tin tức đặc biệt trên báo in Thanh Niên 26.7.2026"))
        self.assertTrue(self.hit("cafef", "XSMN 24/9 - Kết quả XSMN ngày 24/9/2026"))
        self.assertTrue(self.hit("cafef", "Vietlott 17/9 - Xổ số Vietlott Power 6/55 hôm nay 17/9/2026"))
        self.assertTrue(self.hit("thanhnien", "Xổ số miền Nam ngày 31 tháng 5 - Kết quả xổ số hôm nay chủ nhật"))
        self.assertTrue(self.hit("thanhnien", "Kết quả xổ số Vietlott Mega 6/45 tối 17.7, giải độc đắc tăng 15,1 tỉ đồng"))
        for title in ("Xổ số miền Nam ngày 12 tháng 9: Tìm người may mắn trúng độc đắc đài TP.HCM",
                      "Vé số Vietlott trúng độc đắc 257 tỉ đồng được bán ra tại TP.HCM",
                      "Xổ số miền Nam: Đã tìm ra người trúng 34 tỉ đồng"):
            self.assertEqual(self.hit("thanhnien", title), [], title)

    def test_rule_hit_rejects_and_logs(self):
        _, rejected, issues, _ = check_and_fix([{**raw("n", " Tin tức "), "source": "nhandan",
                                                 "url": "https://nhandan.vn/a.html"}],
                                               "nhandan", "nhandan.vn", RULES, {})
        self.assertEqual(rejected[0]["reject_issues"], "non_headline_title")
        self.assertIn("nhandan_generic_title", [i["reason"] for i in issues if i["decision"] == "reject"][0])


class DedupTests(unittest.TestCase):
    def test_number_guard(self):
        pattern = DEDUP["number_pattern"]
        a, b = number_tokens("Lãi suất ngân hàng 22/9 tại MB", pattern), number_tokens("Lãi suất ngân hàng 21/9 tại MB", pattern)
        self.assertEqual(a, {"22/9"})
        self.assertTrue(number_conflict(a, b))
        self.assertFalse(number_conflict(a, set()))
        self.assertTrue(number_conflict(a, set(), "either_side"))
        self.assertEqual(norm_title("  Giá VÀNG: tăng!!  "), "giá vàng tăng")

    def test_cross_source_near_duplicates_link_but_within_source_needs_0_9(self):
        title = "Toy Story 5 đạt doanh thu mở màn cao nhất năm 2026"
        near = "Toy Story 5 lập kỷ lục doanh thu mở màn cao nhất năm 2026"
        records = [rec("a", "tuoitre", title), rec("b", "saostar", near), rec("c", "tuoitre", near + " ở Mỹ"),
                   rec("d", "kenh14", "Lãi suất ngân hàng 22/9 tại MB, Techcombank, VPBank"),
                   rec("e", "cafef", "Lãi suất ngân hàng 21/9 tại MB, Techcombank, VPBank"),
                   rec("f", "eva", "Một tiêu đề hoàn toàn khác")]
        group_of, auto, review, template, _ = find_duplicates(records, DEDUP)
        self.assertEqual(group_of["a"][0], group_of["b"][0])
        self.assertEqual(group_of["b"][0], group_of["c"][0])  # c joins via its cross-source link to b
        self.assertNotEqual(group_of["d"][0], group_of["e"][0])  # number conflict blocks the link
        self.assertIn(("d", "e"), {(r["record_id_a"], r["record_id_b"]) for r in review})
        same_source = [rec("x", "eva", "Loại quả xưa chín rụng đầy nay thành đặc sản được ưa chuộng"),
                       rec("y", "eva", "Loại quả xưa chín rụng đầy nay thành đặc sản ở thành phố")]
        group_of, _, _, template, _ = find_duplicates(same_source, DEDUP)
        self.assertNotEqual(group_of["x"][0], group_of["y"][0])
        self.assertEqual(len(template), 1)
        group_of, _, _, _, _ = find_duplicates([rec("p", "eva", "Giá vàng hôm nay tăng mạnh"),
                                                rec("q", "eva", "Giá vàng hôm nay tăng mạnh!")], DEDUP)
        self.assertEqual(group_of["p"][0], group_of["q"][0])

    def test_exact_threshold_pair_links_despite_float_rounding(self):
        a = "Đề xuất sửa đổi Nghị định 168 về xử phạt giao thông, xe ghép sẽ bị phạt nặng?"
        b = "Đề xuất sửa đổi Nghị định 168: Xử phạt nặng xe ghép"
        self.assertEqual(jc3(a, b), 0.7)
        group_of, auto, _, _, _ = find_duplicates([rec("a", "tuoitre", a), rec("b", "cafef", b)], DEDUP)
        self.assertEqual(group_of["a"], group_of["b"])
        self.assertEqual([r["rule"] for r in auto], ["cross_source_jc3"])

    def test_number_conflict_is_not_bridged_through_a_title_without_numbers(self):
        records = [rec("a", "2sao", "5 chiếc iPhone cũ đáng mua nhất hiện nay"),
                   rec("b", "kenh14", "4 chiếc iPhone cũ đáng mua nhất hiện nay"),
                   rec("c", "cafef", "Hai chiếc iPhone cũ đáng mua nhất hiện nay")]
        group_of, auto, review, _, _ = find_duplicates(records, DEDUP)
        self.assertNotEqual(group_of["a"][0], group_of["b"][0])
        self.assertEqual(sorted(r["blocked_reason"] for r in auto if not r["applied"]), ["transitive_conflict"])
        self.assertIn("transitive_conflict", {r["rule"] for r in review})
        self.assertIn("number_conflict", {r["rule"] for r in review})

    def test_near_links_need_date_window_and_low_links_go_to_review(self):
        title, near = "Toy Story 5 đạt doanh thu mở màn cao nhất năm 2026", "Toy Story 5 lập kỷ lục doanh thu mở màn cao nhất năm 2026"
        self.assertLess(jc3(title, near), DEDUP["review_auto_link_below"])
        group_of, _, review, _, _ = find_duplicates([rec("a", "tuoitre", title, 0), rec("b", "kenh14", near, 30 * DAY)], DEDUP)
        self.assertNotEqual(group_of["a"][0], group_of["b"][0])
        self.assertEqual([(r["rule"], r["days_apart"]) for r in review], [("date_gap", 30.0)])
        group_of, _, review, _, _ = find_duplicates([rec("a", "tuoitre", title, 0), rec("b", "kenh14", near, 2 * DAY)], DEDUP)
        self.assertEqual(group_of["a"][0], group_of["b"][0])
        self.assertEqual([(r["rule"], r["auto_linked"]) for r in review], [("auto_link_check", True)])
        group_of, _, review, _, _ = find_duplicates([rec("a", "eva", "Người mua vàng lỗ nặng", 0),
                                                     rec("b", "eva", "Người mua vàng lỗ nặng", 140 * DAY)], DEDUP)
        self.assertEqual(group_of["a"][0], group_of["b"][0])  # exact titles always link, but are flagged
        self.assertEqual([r["rule"] for r in review], ["exact_date_gap"])

    def test_review_routing_for_cross_source_pairs(self):
        records = [rec("a", "tuoitre", "Giá xăng dầu hôm nay đồng loạt giảm mạnh"),
                   rec("b", "kenh14", "Giá xăng dầu hôm nay đồng loạt giảm nhẹ ở nhiều nơi")]
        score = jc3(records[0]["title"], records[1]["title"])
        self.assertTrue(DEDUP["review_low"] <= score < DEDUP["cross_source_link"], score)
        _, auto, review, _, _ = find_duplicates(records, DEDUP)
        self.assertEqual((auto, [r["rule"] for r in review]), ([], ["review_band"]))

    def test_manual_link_decision_and_separate_as_cannot_link(self):
        records = [rec("a", "afamily", "Một tiêu đề"), rec("b", "kenh14", "Tiêu đề khác hẳn")]
        group_of, auto, _, _, _ = find_duplicates(records, DEDUP, [{"record_id_a": "a", "record_id_b": "b",
                                                                   "decision": "link"}])
        self.assertEqual(group_of["a"][0], group_of["b"][0])
        self.assertEqual(auto[-1]["rule"], "manual_link")
        same = [rec("a", "afamily", "Cùng một tiêu đề"), rec("b", "kenh14", "Cùng một tiêu đề"),
                rec("c", "eva", "Cùng một tiêu đề")]
        group_of, auto, _, _, notes = find_duplicates(same, DEDUP, [{"record_id_a": "a", "record_id_b": "c",
                                                                     "decision": "separate"}])
        self.assertNotEqual(group_of["a"][0], group_of["c"][0])  # not bridged through b either
        self.assertEqual(notes, [])
        self.assertEqual(sum(not r["applied"] for r in auto), 2)

    def test_manual_separate_decision_removes_direct_link(self):
        records = [rec("a", "afamily", "Cùng một tiêu đề"), rec("b", "kenh14", "Cùng một tiêu đề")]
        group_of, _, _, _, _ = find_duplicates(records, DEDUP, [{"record_id_a": "b", "record_id_b": "a",
                                                                 "decision": "separate"}])
        self.assertNotEqual(group_of["a"][0], group_of["b"][0])

    def test_unseen_group_excludes_all_seen_copies(self):
        records = [rec("s1", "kenh14", "Tiêu đề trùng", 1.0), rec("s2", "afamily", "Tiêu đề trùng", 0.0),
                   rec("u1", "yan", "Tiêu đề trùng", 5.0), rec("u2", "saostar", "Tiêu đề trùng", 3.0),
                   rec("t1", "kenh14", "Nhóm chỉ có seen", 2.0), rec("t2", "afamily", "Nhóm chỉ có seen", 2.0)]
        group_of, _, _, _, _ = find_duplicates(records, DEDUP)
        chosen = choose_representatives(records, group_of, {"yan", "saostar"})
        self.assertEqual(chosen["u2"], (True, None))
        self.assertEqual(chosen["u1"], (False, "duplicate_in_group"))
        self.assertEqual(chosen["s1"], (False, "duplicate_of_unseen"))
        self.assertEqual(chosen["s2"], (False, "duplicate_of_unseen"))
        self.assertEqual(chosen["t1"], (True, None))  # tie on published -> smallest record_id
        self.assertEqual(chosen["t2"], (False, "duplicate_in_group"))

    def test_dup_group_ids_are_deterministic_for_singletons_and_groups(self):
        records = [rec(f"r{i}", "eva" if i % 2 else "kenh14", "Tiêu đề chung" if i < 3 else f"Bài số {i} riêng biệt")
                   for i in range(8)]
        first = find_duplicates(records, DEDUP)[0]
        shuffled = records[:]
        random.Random(7).shuffle(shuffled)
        self.assertEqual(first, find_duplicates(shuffled, DEDUP)[0])
        self.assertEqual(first["r0"], (group_id(["r2", "r1", "r0"]), 3))
        self.assertEqual(first["r5"], (group_id(["r5"]), 1))
        self.assertTrue(first["r5"][0].startswith("dg_") and len(first["r5"][0]) == 15)


def write_run(work, source, run_id, rows, selected, shortfall=0, lock=False, pilot=False):
    """Write a crawl run directory; selected=None leaves it unfinished."""
    directory = work / source / run_id
    directory.mkdir(parents=True)
    with (directory / "articles.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (directory / "discovery_report.json").write_text(json.dumps(
        {"unique_article_urls": len(rows) + 5, "inventory_traversal_complete": True, "failures": []}), encoding="utf-8")
    if selected is not None:
        (directory / "selected_record_ids.csv").write_text(
            "record_id,selection_status\n" + "".join(f"{rid},primary\n" for rid in selected), encoding="utf-8")
        (directory / "crawl_report.json").write_text(json.dumps(
            {"pilot": pilot, "shortfall": shortfall, "attempted_urls": len(rows),
             "selection_status_counts": {"primary": len(selected), "not_article_page": 1},
             "selected_by_publication_year": {"2026": len(selected)}}), encoding="utf-8")
    if lock:
        (directory / ".running").write_text("pid", encoding="utf-8")
    return directory


def write_supplement(directory, refetch_id, rows, remaining=0):
    target = directory / "sapo_refetch" / refetch_id
    target.mkdir(parents=True)
    (target / "sapo_supplement.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                                  encoding="utf-8")
    if remaining is not None:
        (target / "report.json").write_text(json.dumps({"remaining": remaining}), encoding="utf-8")


def article(rid, source, title, category="Mục A", sapo="Sapo", published="2026-05-20T08:00:00+07:00", status="success"):
    return {"record_id": rid, "source": source, "url": f"https://{source}.vn/{rid}.html", "title": title, "sapo": sapo,
            "category": category, "published_at": published, "crawled_at": "2026-10-21T06:52:00+00:00",
            "extraction_status": status}


def write_configs(root, runs, supplements=None):
    (root / "sources.csv").write_text(
        "source,domain,outlet_group,min_clean_articles\neva,eva.vn,seen,1\nyan,yan.vn,unseen,1\n", encoding="utf-8")
    (root / "run.json").write_text(json.dumps({"raw_targets": {"seen": 1, "unseen": 1}}), encoding="utf-8")
    (root / "final_runs.json").write_text(json.dumps({"runs": runs, "sapo_supplements": supplements or {}}),
                                          encoding="utf-8")
    (root / "mapping.csv").write_text("source,category_raw,category,review_status,notes\n"
                                      f"eva,Mục A,{categories.DS},proposed,\n", encoding="utf-8")
    return ["--final-runs", str(root / "final_runs.json"), "--work-root", str(root / "work"), "--sources",
            str(root / "sources.csv"), "--run-config", str(root / "run.json"), "--out-root", str(root / "out")]


def quiet(function, argv):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return function(argv)


class ExportRawTests(unittest.TestCase):
    def test_supplement_fill_status_recompute_order_and_nine_fields(self):
        rows = [article("n1", "eva", "Bài không chọn", category="Mục B"),
                article("s2", "eva", "Bài hai", sapo=None, status="partial"),
                article("t1", "eva", None, category="Trang chủ", status="partial"),
                article("e1", "eva", "Lỗi tải", status="fetch_error"),
                article("s1", "eva", "Bài một", sapo=None, status="partial"),
                article("s3", "eva", "Bài ba", sapo=None, category=None, status="partial"),
                article("s4", "eva", "Bài bốn", sapo="Có sẵn")]
        supplement = [{"record_id": "s1", "status": "sapo_recovered", "sapo": "Lead tải lại", "refetched_at": "t"},
                      {"record_id": "s2", "status": "title_changed", "sapo": None, "refetched_at": "t"},
                      {"record_id": "s3", "status": "sapo_recovered", "sapo": "Lead ba", "refetched_at": "t"},
                      {"record_id": "s4", "status": "sapo_recovered", "sapo": "Khác", "refetched_at": "t"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = write_configs(root, {"eva": "r1", "yan": "r1"}, {"eva": "fix1"})
            write_supplement(write_run(root / "work", "eva", "r1", rows, ["s4", "s1", "s2", "s3"]), "fix1", supplement)
            write_run(root / "work", "yan", "r1", [article("y1", "yan", "Bài yan")], ["y1"])
            original = (root / "work/eva/r1/articles.jsonl").read_bytes()
            quiet(EXPORT.main, common)
            self.assertEqual(original, (root / "work/eva/r1/articles.jsonl").read_bytes())  # crawl run is only read
            lines = (root / "out/data/raw/articles_raw.jsonl").read_text(encoding="utf-8").splitlines()
            stats = {r["source"]: r for r in BUILD.read_csv(root / "out/reports/crawl_stats.csv")}
        exported = [json.loads(line) for line in lines]
        self.assertEqual([r["record_id"] for r in exported], ["s4", "s1", "s2", "s3", "y1"])
        self.assertTrue(all(list(r) == list(EXPORT.ARTICLE_FIELDS) for r in exported))
        by_id = {r["record_id"]: r for r in exported}
        self.assertEqual((by_id["s1"]["sapo"], by_id["s1"]["extraction_status"]), ("Lead tải lại", "success"))
        self.assertEqual((by_id["s2"]["sapo"], by_id["s2"]["extraction_status"]), (None, "partial"))  # title changed
        self.assertEqual((by_id["s3"]["sapo"], by_id["s3"]["extraction_status"]), ("Lead ba", "partial"))  # no category
        self.assertEqual(by_id["s4"]["sapo"], "Có sẵn")  # an existing sapo is never replaced
        eva = stats["eva"]
        self.assertEqual({k: eva[k] for k in ("downloaded", "downloaded_without_title", "selected", "shortfall",
                                              "missing_sapo_before_refetch", "sapo_refetched", "missing_sapo",
                                              "success", "partial", "selection_primary", "selection_not_article_page",
                                              "run_id", "sapo_refetch_id")},
                         {"downloaded": "5", "downloaded_without_title": "1", "selected": "4", "shortfall": "0",
                          "missing_sapo_before_refetch": "3", "sapo_refetched": "2", "missing_sapo": "1",
                          "success": "2", "partial": "2", "selection_primary": "4", "selection_not_article_page": "1",
                          "run_id": "r1", "sapo_refetch_id": "fix1"})
        self.assertEqual(json.loads(eva["downloaded_by_category"]), [["Mục A", 3], [None, 1], ["Mục B", 1]])

    def test_refuses_locked_incomplete_and_unfinished_runs(self):
        cases = {".running": dict(lock=True), "no selected_record_ids.csv": dict(selected=None),
                 "shortfall=3": dict(shortfall=3), "selected_ids_without_row=1": dict(selected=["y1", "x9"]),
                 "pilot run": dict(pilot=True), "selected=0 < raw target 1": dict(selected=[])}
        for reason, options in cases.items():
            with self.subTest(reason), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                common = write_configs(root, {"eva": "r1", "yan": "r1"})
                write_run(root / "work", "eva", "r1", [article("e1", "eva", "Bài")], ["e1"])
                write_run(root / "work", "yan", "r1", [article("y1", "yan", "Bài")],
                          options.pop("selected", ["y1"]), **options)
                errors = io.StringIO()
                with self.assertRaises(SystemExit), contextlib.redirect_stderr(errors):
                    EXPORT.main(common)
                self.assertIn("yan (r1): ", errors.getvalue())
                self.assertIn(reason, errors.getvalue())
                self.assertNotIn("eva (r1)", errors.getvalue())
                self.assertFalse((root / "out/data/raw/articles_raw.jsonl").exists())
                self.assertFalse((root / "out/reports/crawl_stats.csv").exists())
        for remaining in (None, 4):  # re-fetch without report.json, or still running
            with self.subTest(remaining=remaining), tempfile.TemporaryDirectory() as directory:
                run = write_run(Path(directory), "eva", "r1", [article("e1", "eva", "Bài", sapo=None)], ["e1"])
                write_supplement(run, "fix1", [], remaining)
                with self.assertRaisesRegex(ValueError, "sapo re-fetch"):
                    EXPORT.export_run(run, "fix1")

    def test_missing_run_and_replace_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = write_configs(root, {"eva": "r1"})
            write_run(root / "work", "eva", "r1", [article("e1", "eva", "Bài")], ["e1"])
            with self.assertRaises(SystemExit):
                quiet(EXPORT.main, common)  # yan has no final run
            write_configs(root, {"eva": "r1", "yan": "r1"})
            write_run(root / "work", "yan", "r1", [article("y1", "yan", "Bài")], ["y1"])
            quiet(EXPORT.main, common)
            with self.assertRaises(SystemExit):
                quiet(EXPORT.main, common)  # outputs exist
            quiet(EXPORT.main, [*common, "--replace"])


class CleanInputTests(unittest.TestCase):
    def test_read_raw_and_crawl_stats_fail_clearly(self):
        sources = {"eva": {}, "yan": {}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jsonl"
            path.write_text(json.dumps(article("e1", "eva", "A")) + "\n" + json.dumps(article("y1", "yan", "B")) + "\n",
                            encoding="utf-8")
            self.assertEqual({s: [r["record_id"] for r in rows] for s, rows in read_raw(path, sources).items()},
                             {"eva": ["e1"], "yan": ["y1"]})
            for content in ("{broken\n", json.dumps(article("x1", "other", "A")) + "\n"):
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_raw(path, sources)
        row = {"source": "eva", "run_id": "r1", "downloaded": "3", "downloaded_without_title": "1", "selected": "2",
               "downloaded_by_category": json.dumps([["Mục A", 2], [None, 1]])}
        stats = read_crawl_stats([row], {"eva": {}})
        self.assertEqual(stats["eva"]["downloaded_by_category"], [("Mục A", 2), (None, 1)])
        with self.assertRaises(ValueError):
            read_crawl_stats([row], {"eva": {}, "yan": {}})  # source without a row
        with self.assertRaises(ValueError):
            read_crawl_stats([{**row, "downloaded": "4"}], {"eva": {}})  # categories do not sum up

    def test_generator_reads_raw_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "raw.jsonl"
            mapping = Path(directory) / "mapping.csv"
            mapping.write_text("source,category_raw,category,review_status,notes\n", encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                PROPOSE.main(["--raw", str(raw_path), "--dry-run", "--mapping", str(mapping)])  # no export yet
            raw_path.write_text(json.dumps(article("d1", "dantri", "A", category="Du lịch, Khám phá"),
                                           ensure_ascii=False) + "\n", encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                PROPOSE.main(["--raw", str(raw_path), "--dry-run", "--mapping", str(mapping)])
        proposal = json.loads(output.getvalue().splitlines()[0])
        self.assertEqual((proposal["source"], proposal["category"]), ("dantri", categories.DL))

    def test_funnel_counts(self):
        sources = {"eva": {"outlet_group": "seen", "min_clean_articles": "3"}}
        stats = {"eva": {"run_id": "r", "downloaded_without_title": 5,
                         "downloaded_by_category": [("Mục A", 2), ("Lạ", 1)]}}
        mapping = {("eva", "Mục A"): {"category": categories.DS, "review_status": "proposed"}}
        valid = [{"source": "eva", "category": categories.DS, "is_dup_representative": True,
                  "dedup_excluded_reason": None, "has_sapo": True},
                 {"source": "eva", "category": categories.DS, "is_dup_representative": False,
                  "dedup_excluded_reason": "duplicate_in_group", "has_sapo": True},
                 {"source": "eva", "category": None, "is_dup_representative": True,
                  "dedup_excluded_reason": None, "has_sapo": False}]
        by_source, by_category = funnel(sources, stats, {"eva": 4}, valid, mapping)
        row = by_source[0]
        self.assertEqual({k: row[k] for k in ("run_id", "downloaded", "downloaded_without_title", "selected", "valid",
                                              "rejected", "dedup_excluded", "after_dedup", "with_sapo", "shortfall")},
                         {"run_id": "r", "downloaded": 3, "downloaded_without_title": 5, "selected": 4, "valid": 3,
                          "rejected": 1, "dedup_excluded": 1, "after_dedup": 2, "with_sapo": 1, "shortfall": 1})
        cats = {r["category"]: r for r in by_category}
        self.assertEqual((cats[categories.DS]["downloaded"], cats[categories.DS]["after_dedup"]), (2, 1))
        self.assertEqual((cats["<unmapped>"]["downloaded"], cats["<unmapped>"]["valid"]), (1, 1))

    def test_build_clean_reads_raw_and_crawl_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = write_configs(root, {"eva": "r1", "yan": "r1"})
            write_run(root / "work", "eva", "r1", [article("e1", "eva", "Cùng một tiêu đề"),
                                                   article("e2", "eva", "Tin khác", category="Mục lạ"),
                                                   article("e3", "eva", "Tin tức", category="Mục A")], ["e1", "e2"])
            write_run(root / "work", "yan", "r1", [article("y1", "yan", "Cùng một tiêu đề")], ["y1"])
            out = root / "out"
            build = ["--out-root", str(out), "--raw", str(out / "data/raw/articles_raw.jsonl"), "--crawl-stats",
                     str(out / "reports/crawl_stats.csv"), "--sources", str(root / "sources.csv"),
                     "--mapping", str(root / "mapping.csv"), "--decisions", str(root / "none.csv")]
            with self.assertRaises(SystemExit):
                quiet(BUILD.main, build)  # export_raw.py not run: raw and crawl_stats.csv missing
            quiet(EXPORT.main, common)
            quiet(BUILD.main, build)
            frame = __import__("pandas").read_parquet(out / "data/processed/articles_clean.parquet")
            self.assertEqual(list(frame.columns), list(BUILD.CLEAN_COLUMNS))
            kept = frame.set_index("record_id")
            self.assertEqual(kept.loc["e1", "dedup_excluded_reason"], "duplicate_of_unseen")
            self.assertEqual((kept.loc["e2", "category"], kept.loc["e2", "category_mapping_status"]), (None, None))
            self.assertEqual(kept.loc["e1", "category_mapping_status"], "proposed")
            self.assertEqual(set(frame["source_run_id"]), {"r1"})
            self.assertEqual(sorted(p.name for p in (out / "data/interim").iterdir()), sorted(BUILD.INTERIM_FILES))
            self.assertEqual(sorted(p.name for p in (out / "reports").iterdir()),
                             sorted(["crawl_stats.csv", *BUILD.REPORT_FILES]))
            report = (out / "reports/data_report.md").read_text(encoding="utf-8")
            self.assertIn("## 1. Raw crawl", report)
            self.assertIn("## 2. Clean", report)
            funnel_rows = {r["source"]: r for r in BUILD.read_csv(out / "reports/funnel_by_source.csv")}
            self.assertEqual({k: funnel_rows["eva"][k] for k in ("run_id", "downloaded", "selected", "valid", "after_dedup")},
                             {"run_id": "r1", "downloaded": "3", "selected": "2", "valid": "2", "after_dedup": "1"})
            with self.assertRaises(SystemExit):
                quiet(BUILD.main, build)  # outputs exist, no --replace
            quiet(BUILD.main, [*build, "--replace"])
            stats = BUILD.read_csv(out / "reports/crawl_stats.csv")
            with (out / "reports/crawl_stats.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(stats[0]))
                writer.writeheader()
                writer.writerows([{**r, "selected": "9"} if r["source"] == "eva" else r for r in stats])
            with self.assertRaises(SystemExit):
                quiet(BUILD.main, [*build, "--replace"])  # raw and crawl_stats disagree


if __name__ == "__main__":
    unittest.main()
