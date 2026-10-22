import contextlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ds_project.crawling.discover import archive_year, article_url, parse_sitemap
from ds_project.crawling.extract import extract, parse_datetime

SPEC = importlib.util.spec_from_file_location("crawl_data", ROOT / "scripts/crawl_data.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CrawlingTests(unittest.TestCase):
    profile = {"title": "h1", "sapo": ".sapo", "body": "article", "category": ""}

    def test_visible_title_wins_over_rewritten_seo_and_missing_sapo_stays_missing(self):
        html = '''<h1>Tiêu đề GỐC?!…</h1><meta property="og:title" content="SEO title">
        <meta name="description" content="Not a verified sapo"><article><p>Body</p></article>
        <script type="application/ld+json">{"@type":"NewsArticle","headline":"Shortened...",
        "datePublished":"2026-09-30T20:30:00&#x2B;07:00"}</script>'''
        result = extract(html.encode(), "https://example.vn/a-12345.html", self.profile)
        self.assertEqual(result["title"], "Tiêu đề GỐC?!…")
        self.assertIsNone(result["sapo"])
        self.assertEqual(result["published_at"], "2026-09-30T20:30:00+07:00")

    def test_related_article_date_is_not_used_for_main_article(self):
        html = '''<h1>Article</h1><article><p>Body</p></article><script type="application/ld+json">
        {"@type":"ItemList","itemListElement":[{"@type":"NewsArticle",
        "datePublished":"2026-01-01T12:00:00+07:00"}]}</script>'''
        self.assertIsNone(extract(html.encode(), "https://example.vn/a-12345.html", self.profile)["published_at"])

    def test_verified_breadcrumb_syntax_repair_preserves_category_text(self):
        html = '''<h1>Original title</h1><script type="application/ld+json">
        {"@type":"BreadcrumbList","itemListElement":[{"@type":"ListItem","position":2,
        "item":{"@id":"https://example.vn/xa-hoi.chn" "name":"Xã hội"}}]}</script>'''
        profile = {**self.profile, "repair_breadcrumb_missing_comma": True}
        result = extract(html.encode(), "https://example.vn/a-12345.html", profile)
        self.assertEqual(result["category"], "Xã hội")
        self.assertIn("breadcrumb_missing_comma_repaired", result["extraction_warnings"])
        self.assertIsNone(extract(html.encode(), "https://example.vn/a-12345.html", self.profile)["category"])

    def test_naive_timestamp_requires_approved_source_timezone_policy(self):
        html = b'<h1>A</h1><script type="application/ld+json">{"@type":"NewsArticle","datePublished":"2026-10-21T09:30:00.000000"}</script>'
        self.assertIsNone(extract(html, "https://cafef.vn/a-12345.chn", self.profile)["published_at"])
        profile = {**self.profile, "naive_timestamp_offset": "+07:00", "timezone_policy": "Approved for CafeF"}
        result = extract(html, "https://cafef.vn/a-12345.chn", profile)
        self.assertEqual(result["published_at"], "2026-10-21T09:30:00+07:00")
        self.assertIn("timezone_assumed_by_approved_source_policy", result["extraction_warnings"])

    def test_all_formats_allowed_but_non_article_pages_rejected(self):
        config = {"start_date": "2026-01-01", "end_date": "2026-10-21", "fallback_start_date": "2025-01-01"}
        for schema in ("NewsArticle", "ImageGallery", "VideoObject", "LiveBlogPosting"):
            html = '<h1>Tiêu đề</h1><script type="application/ld+json">' + (
                '{"@type":"' + schema + '","datePublished":"2026-01-01T12:00:00+07:00"}') + '</script>'
            result = extract(html.encode(), "https://example.vn/a-12345.html", self.profile)
            self.assertEqual(RUNNER.eligibility({**result, "extraction_status": "partial"}, config), "primary")
        result = extract(b"<h1>Homepage</h1>", "https://example.vn/", self.profile)
        self.assertEqual(RUNNER.eligibility({**result, "extraction_status": "partial"}, config), "not_article_page")

    def test_sitemap_publication_date_is_distinct_from_lastmod(self):
        body = b'''<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:news="http://www.google.com/schemas/sitemap-news/0.9">
        <url><loc>https://example.vn/a-12345.html</loc><lastmod>2026-10-21</lastmod>
        <news:news><news:publication_date>2025-12-01</news:publication_date></news:news></url></urlset>'''
        kind, entries = parse_sitemap(body)
        self.assertEqual(kind, "urlset")
        self.assertEqual(entries[0]["publication_date_hint"], "2025-12-01")
        with self.assertRaises(ValueError):
            parse_sitemap(b"<html><body>Homepage</body></html>")
        self.assertEqual(archive_year("https://example.vn/sitemap-article-10-2026-01.xml"), 2026)

    def test_date_boundaries_timezone_and_fallback(self):
        config = {"start_date": "2026-01-01", "end_date": "2026-10-21", "fallback_start_date": "2025-01-01"}
        record = {"extraction_status": "success", "is_article_page": True, "title": "A"}
        for timestamp, expected in [("2025-12-31T18:00:00Z", "primary"),
                                    ("2025-01-01T00:00:00+07:00", "fallback"),
                                    ("2024-12-31T23:00:00+07:00", "before_fallback_date"),
                                    ("2026-10-22T00:00:00+07:00", "after_end_date")]:
            self.assertEqual(RUNNER.eligibility({**record, "published_at": timestamp}, config), expected)
        self.assertIsNone(parse_datetime("2026-01-01T12:00:00"))
        self.assertEqual(parse_datetime("2026-01-01T12:00:00+7:00").utcoffset().total_seconds(), 7 * 3600)

    def test_xml_recovery_preserves_all_original_urls_and_logs_repairs(self):
        body = b'<urlset><url><loc>https://example.vn/a-12345.html?x=1&amp;y=2</loc><title>A&#1;B</title></url><url><loc>https://example.vn/b-12345.html</loc></url></urlset>'
        with self.assertRaises(Exception):
            parse_sitemap(body)
        repairs = []
        _, entries = parse_sitemap(body, allow_recovery=True, recovery_log=repairs)
        self.assertEqual([e['url'] for e in entries], ['https://example.vn/a-12345.html?x=1&y=2', 'https://example.vn/b-12345.html'])
        self.assertEqual(repairs[0]['records'], 2)
        self.assertTrue(repairs[0]['original_loc_urls_preserved'])

    def test_xml_recovery_rejects_truncation_and_changed_url(self):
        for body in [
            b'<urlset><url><loc>https://example.vn/a-12345.html</loc><title>A&#1;B</title></url>',
            b'<urlset><url><loc>https://example.vn/a&#1;-12345.html</loc></url></urlset>',
            b'<urlset><url><title>A&#1;B</title></url></urlset>',
        ]:
            with self.subTest(body=body), self.assertRaises(ValueError):
                parse_sitemap(body, allow_recovery=True)

    def test_quarantine_requires_opt_in_and_does_not_rewrite_broken_urls(self):
        body = b'<urlset><url><loc>https://example.vn/a\x08-12345.html</loc></url><url><loc>https://example.vn/b-12345.html</loc></url></urlset>'
        with self.assertRaises(ValueError):
            parse_sitemap(body, allow_recovery=True)
        repairs = []
        _, entries = parse_sitemap(body, allow_recovery=True, quarantine_invalid_urls=True, recovery_log=repairs)
        self.assertEqual([entry['url'] for entry in entries], ['https://example.vn/b-12345.html'])
        self.assertEqual(len(repairs[0]['quarantined_urls']), 1)
        self.assertEqual(repairs[0]['records'], 2)
        self.assertEqual(repairs[0]['valid_records'], 1)
        self.assertIn('\x08', repairs[0]['quarantined_urls'][0]['original_loc'])

    def test_tag_listing_pages_are_not_articles(self):
        self.assertFalse(article_url("https://vietnamnet.vn/du-lich-duong-sat-tag681215.html"))
        self.assertFalse(article_url("https://2sao.vn/vietjet-tag5009596.html"))
        for url in ["https://afamily.vn/me-tag-ten-con-tren-mxh-17226012345.chn",
                    "https://dantri.com.vn/o-to-xe-may/kia-sportage-2026-ra-mat-20260101120000123.htm",
                    "https://vietnamnet.vn/iphone-18-lo-thiet-ke-2566728.html"]:
            self.assertTrue(article_url(url), url)

    def test_tag_sitemaps_are_skipped_without_fetching(self):
        from ds_project.crawling.discover import discover_sitemaps
        index = b'''<sitemapindex><sitemap><loc>https://example.vn/sitemap-tags-17.xml</loc></sitemap>
        <sitemap><loc>https://example.vn/sitemap-article-1.xml</loc></sitemap></sitemapindex>'''
        urlset = b'<urlset><url><loc>https://example.vn/a-12345.html</loc></url></urlset>'

        class FakeClient:
            requested = []

            def get(self, url):
                self.requested.append(url)
                return (index if url.endswith("sitemap.xml") else urlset), url

            def allowed_host(self, url):
                return url.startswith("https://example.vn/")

        client = FakeClient()
        config = {"fallback_start_date": "2024-01-01", "start_date": "2026-01-01", "end_date": "2026-10-21"}
        with tempfile.TemporaryDirectory() as directory:
            inventory, report = discover_sitemaps(client, {"sitemaps": ["https://example.vn/sitemap.xml"]},
                                                  config, Path(directory), 10)
        self.assertEqual(report["skipped_non_article_sitemaps"], ["https://example.vn/sitemap-tags-17.xml"])
        self.assertNotIn("https://example.vn/sitemap-tags-17.xml", client.requested)
        self.assertEqual([row["url"] for row in inventory], ["https://example.vn/a-12345.html"])

    def test_yan_sapo_requires_fully_bold_first_paragraph(self):
        profile = {**self.profile, "sapo": "#contentBody > p:first-of-type", "sapo_require_full_bold_paragraph": True}
        page = '<h1>T</h1><div id="contentBody">{}<p>Body</p></div>'
        cases = [('<p><strong>Lead in đậm.</strong></p>', "Lead in đậm."),
                 ('<p><strong>Lead</strong> <b>hai phần</b><br></p>', "Lead hai phần"),
                 ('<p>Đoạn thường có <strong>một từ</strong> đậm.</p>', None),
                 ('<p>Đoạn thường.</p>', None),
                 ('<p><strong> </strong></p>', None)]
        for first, expected in cases:
            with self.subTest(first=first):
                result = extract(page.format(first).encode(), "https://yan.vn/a-12345.html", profile)
                self.assertEqual(result["sapo"], expected)
                self.assertEqual("sapo" in result["extraction_evidence"], expected is not None)

    def test_dantri_emagazine_sapo_variants(self):
        profile = {**self.profile, "sapo": "[data-slot='sapo'], .singular-sapo, h2.e-magazine__sapo, article.e-magazine > h2, body.emagazine h1 + h2"}
        pages = ['<h1>T</h1><article class="e-magazine"><h2 class="e-magazine__sapo">(Dân trí) - Lead</h2></article>',
                 '<article class="e-magazine dbiz"><h1>T</h1><h2>(Dân trí) - Lead</h2><div class="e-magazine__body"><h2>Mục</h2></div></article>',
                 '<body class="emagazine"><div><h1>T</h1><h2>(Dân trí) - Lead</h2></div></body>']
        for html in pages:
            self.assertEqual(extract(html.encode(), "https://dantri.com.vn/a-12345.htm", profile)["sapo"], "(Dân trí) - Lead")

    def test_nine_field_output_resume_and_fallback_preserve_raw(self):
        config = {"start_date": "2026-01-01", "end_date": "2026-10-21",
                  "fallback_start_date": "2024-01-01", "seed": 2026}

        class FakeClient:
            def __init__(self, directory):
                self.log_dir = directory / "journal"
                self.log_dir.mkdir()
                self.calls = []

            def get(self, url):
                self.calls.append(url)
                year = "2025" if "old" in url else "2026"
                html = '<h1>Title?!</h1><script type="application/ld+json">' + (
                    '{"@type":"NewsArticle","datePublished":"' + year + '-02-01T12:00:00+07:00"}') + '</script>'
                return html.encode(), url

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            client = FakeClient(directory)
            inventory = [{"url": "https://example.vn/new-12345.html", "discovered_from": "sitemap", "archive_year_hint": 2026},
                         {"url": "https://example.vn/old-12345.html", "discovered_from": "sitemap", "archive_year_hint": 2025},
                         {"url": "https://example.vn/older-12345.html", "discovered_from": "sitemap", "archive_year_hint": 2024}]
            (directory / "discovered_urls.jsonl").write_text("\n".join(json.dumps(row) for row in inventory), encoding="utf-8")
            (directory / "discovery_report.json").write_text('{"inventory_traversal_complete":true}', encoding="utf-8")
            first = RUNNER.fetch_source(client, "example", self.profile, config, directory, 2)
            content = (directory / "articles.jsonl").read_bytes()
            second = RUNNER.fetch_source(client, "example", self.profile, config, directory, 2)
            self.assertEqual(content, (directory / "articles.jsonl").read_bytes())
            self.assertEqual(len(client.calls), 2)
            self.assertEqual(first["selected_records"], 2)
            self.assertEqual(second["fallback_used"], 1)
            self.assertEqual(second["selected_by_publication_year"], {2026: 1, 2025: 1})
            self.assertEqual(list(directory.rglob("*.html")), [])
            for row in RUNNER.rows(directory / "articles.jsonl"):
                self.assertEqual(set(row), set(RUNNER.ARTICLE_FIELDS))
                self.assertIsNone(row["sapo"])

    def test_fallback_year_order_and_unprocessed_newer_urls(self):
        candidates = [{"url": str(year), "archive_year_hint": year} for year in [2024, 2025, 2026]]
        self.assertEqual([row["archive_year_hint"] for row in RUNNER.ordered_candidates(candidates, 2026)], [2026, 2025, 2024])
        config = {"start_date": "2026-01-01", "fallback_start_date": "2024-01-01"}
        records = [{"record_id": "new", "selection_status": "primary", "published_at": "2026-01-01T12:00:00+07:00"},
                   {"record_id": "old", "selection_status": "fallback", "published_at": "2024-02-01T12:00:00+07:00"}]
        self.assertEqual(len(RUNNER.selected_records(records, config, 2, {2025: 1, 2024: 2}, True)), 1)
        records.append({"record_id": "middle", "selection_status": "fallback", "published_at": "2025-02-01T12:00:00+07:00"})
        chosen = RUNNER.selected_records(records, config, 2, {2025: 1, 2024: 2}, True)
        self.assertEqual([row["record_id"] for row in chosen], ["new", "middle"])
        self.assertEqual(len(RUNNER.selected_records(records, config, 2, {2026: 1}, True)), 1)
        self.assertEqual(len(RUNNER.selected_records(records, config, 2, {}, False)), 1)

    def test_extraction_status_rule(self):
        record = {"title": "T", "sapo": "S", "category": "C", "published_at": "2026-01-01T00:00:00+07:00"}
        self.assertEqual(RUNNER.extraction_status(record), "success")
        for field in ("title", "sapo", "category", "published_at"):
            self.assertEqual(RUNNER.extraction_status({**record, field: None}), "partial", field)

    def test_2024_is_eligible_only_after_newly_authorized_cutoff(self):
        config = {"start_date": "2026-01-01", "end_date": "2026-10-21", "fallback_start_date": "2024-01-01"}
        row = {"extraction_status": "partial", "is_article_page": True, "title": "A", "published_at": "2024-01-01T00:00:00+07:00"}
        self.assertEqual(RUNNER.eligibility(row, config), "fallback")
        row["published_at"] = "2023-12-31T23:59:59+07:00"
        self.assertEqual(RUNNER.eligibility(row, config), "before_fallback_date")


class FakeNetwork:
    """Offline stand-in for network.Client: robots/request logs go to the given journal directory."""
    refetch = False

    def __init__(self, domain, config, log_dir):
        self.log_dir = log_dir
        log_dir.mkdir(parents=True, exist_ok=True)

    def load_robots(self):
        (self.log_dir / "robots.txt").write_text("User-agent: *\nAllow: /\n", encoding="utf-8")

    def get(self, url):
        with (self.log_dir / "requests.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"url": url}) + "\n")
        number = int(url.rsplit("-", 1)[1].split(".")[0])
        sapo = f'<h2 class="knc-sapo">Lead {number}</h2>' if FakeNetwork.refetch or number % 2 == 0 else ""
        return (f'<html><body><h1 class="kbwc-title">Title {number}</h1>{sapo}<a data-role="cate-name">Xã hội</a>'
                '<div class="knc-content">body</div><script type="application/ld+json">{"@type":"NewsArticle",'
                '"datePublished":"2026-03-01T10:00:00+07:00"}</script></body></html>').encode(), url


class WorkLayoutTests(unittest.TestCase):
    """Entry points write only under data/crawl_work (journal/ for operational logs), never to logs/."""

    def files(self, base):
        return sorted(p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file())

    def test_crawl_and_refetch_write_run_and_journal_layout(self):
        crawl, refetch = load_script("crawl_data"), load_script("refetch_missing_sapo")

        def discover(client, profile, config, directory, max_pages):
            inventory = [{"url": f"https://kenh14.vn/a-{100000 + i}.chn", "discovered_from": "sitemap",
                          "archive_year_hint": 2026} for i in range(6)]
            return inventory, {"inventory_traversal_complete": True, "failures": []}

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            work = base / "data/crawl_work"
            config = json.loads((ROOT / "configs/crawling/run.json").read_text(encoding="utf-8"))
            config.update(raw_targets={"seen": 4, "unseen": 4}, request_delay_seconds=0)
            (base / "run.json").write_text(json.dumps(config), encoding="utf-8")
            FakeNetwork.refetch = False
            argv = ["crawl_data", "--sources", "kenh14", "--run-id", "r1", "--config", str(base / "run.json")]
            with (mock.patch.object(crawl, "WORK", work), mock.patch.object(crawl, "Client", FakeNetwork),
                  mock.patch.object(crawl, "discover_sitemaps", discover), mock.patch.object(sys, "argv", argv),
                  contextlib.redirect_stdout(io.StringIO())):
                self.assertEqual(crawl.main(), 0)
            run = work / "kenh14/r1"
            self.assertEqual(self.files(base / "data"), [f"crawl_work/kenh14/r1/{name}" for name in (
                "articles.jsonl", "crawl_report.json", "discovered_urls.jsonl", "discovery_report.json",
                "journal/article_attempts.jsonl", "journal/progress.json", "journal/requests.jsonl",
                "journal/robots.txt", "run_manifest.json", "selected_record_ids.csv")])
            FakeNetwork.refetch = True
            argv = ["refetch", "--source", "kenh14", "--run-id", "r1", "--refetch-id", "f1", "--request-delay", "0"]
            with (mock.patch.object(refetch, "WORK", work), mock.patch.object(refetch, "Client", FakeNetwork),
                  mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO())):
                self.assertIn(refetch.main(), (0, None))
            self.assertEqual(self.files(run / "sapo_refetch"), ["f1/journal/requests.jsonl", "f1/journal/robots.txt",
                                                               "f1/manifest.json", "f1/report.json",
                                                               "f1/sapo_supplement.jsonl"])
            self.assertTrue(all(row["status"] == "sapo_recovered" for row in RUNNER.rows(
                run / "sapo_refetch/f1/sapo_supplement.jsonl")))
            self.assertFalse(any("logs" in p.parts for p in base.rglob("*")))

    def test_batch_writes_status_and_console_log_under_batches(self):
        batch = load_script("run_crawling_batch")
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            shutil.copytree(ROOT / "configs/crawling", base / "configs/crawling")
            work = base / "data/crawl_work"
            old = work / "kenh14/old/discovery_cache"
            old.mkdir(parents=True)
            (old / "a.xml").write_text("<x/>", encoding="utf-8")
            commands = []

            class FakeProcess:
                pid, returncode = 1, 0

                def __init__(self, command, cwd, stdout, stderr):
                    commands.append(command)
                    run = work / command[command.index("--sources") + 1] / command[command.index("--run-id") + 1]
                    (run / "crawl_report.json").write_text('{"shortfall": 0}', encoding="utf-8")
                    stdout.write("child output\n")

                def poll(self):
                    return 0

            argv = ["batch", "--run-id", "b1", "--sources", "kenh14", "--reuse-discovery-cache"]
            with (mock.patch.object(batch, "ROOT", base), mock.patch.object(batch, "WORK", work),
                  mock.patch.object(batch.subprocess, "Popen", FakeProcess), mock.patch.object(sys, "argv", argv)):
                self.assertEqual(batch.main(), 0)
            self.assertEqual(len(commands), 1)
            self.assertEqual([p for p in self.files(base) if not p.startswith("configs/")], [
                "data/crawl_work/_batches/b1/kenh14.console.log", "data/crawl_work/_batches/b1/status.json",
                "data/crawl_work/kenh14/b1/cache_reuse.json", "data/crawl_work/kenh14/b1/crawl_report.json",
                "data/crawl_work/kenh14/b1/discovery_cache/a.xml", "data/crawl_work/kenh14/old/discovery_cache/a.xml"])
            status = json.loads((work / "_batches/b1/status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["sources"]["kenh14"]["state"], "complete")
            reused = json.loads((work / "kenh14/b1/cache_reuse.json").read_text(encoding="utf-8"))
            self.assertEqual([Path(p).as_posix() for p in reused], ["data/crawl_work/kenh14/old/discovery_cache/a.xml"])


if __name__ == "__main__":
    unittest.main()
