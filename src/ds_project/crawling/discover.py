"""Build URL inventories without applying category quotas or SEO labels."""

import hashlib
import json
import re
from collections import deque
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlsplit
from xml.etree import ElementTree as ET

from bs4 import BeautifulSoup


def article_url(url):
    path = urlsplit(url).path
    # Tag listing ids (2sao/VietNamNet "-tag681215.html") also satisfy the article-id pattern.
    return bool(re.search(r"\d{5,}[^/]*\.(?:html?|chn)$", path, re.I)) and not re.search(r"-tag\d+\.html?$", path, re.I)


def archive_year(url):
    match = re.search(r"(?:^|[-/])(20\d{2})(?:[-/.]|$)", urlsplit(url).path)
    return int(match[1]) if match else None


def local_name(tag):
    return tag.rsplit("}", 1)[-1]


def parse_sitemap(body, *, allow_recovery=False, quarantine_invalid_urls=False, recovery_log=None):
    recovery = None
    try:
        root = ET.fromstring(body)
    except ET.ParseError as strict_error:
        if not allow_recovery:
            raise
        from lxml import etree
        if b"<!DOCTYPE" in body.upper():
            raise ValueError("Refusing recovery of XML with a DOCTYPE") from strict_error
        parser = etree.XMLParser(recover=True, resolve_entities=False, no_network=True)
        root = etree.fromstring(body, parser)
        if root is None:
            raise ValueError("XML recovery produced no root") from strict_error
        recovery = {"strict_error": str(strict_error), "parser": "lxml_recover",
                    "parser_messages": [str(entry) for entry in parser.error_log][:20]}
    kind = local_name(root.tag)
    if kind not in {"sitemapindex", "urlset"}:
        raise ValueError(f"Expected sitemap XML, received {kind}")
    entries = []
    for child in root:
        loc = next((n.text for n in child if isinstance(n.tag, str) and local_name(n.tag) == "loc"), None)
        if loc:
            published = next((n.text for n in child.iter() if isinstance(n.tag, str) and local_name(n.tag) == "publication_date"), None)
            entries.append({"url": loc.strip(), "publication_date_hint": published})
    if recovery is not None:
        tag = b"url" if kind == "urlset" else b"sitemap"
        prefix = rb"(?:[A-Za-z_][\w.-]*:)?"
        count = len(re.findall(rb"<" + prefix + tag + rb"(?:\s[^>]*|)>", body))
        locs = re.findall(rb"<" + prefix + tag + rb"(?:\s[^>]*|)>\s*<" + prefix +
                          rb"loc(?:\s[^>]*|)>(.*?)</" + prefix + rb"loc\s*>", body, re.S)
        if not re.search(rb"</" + prefix + kind.encode() + rb"\s*>\s*$", body):
            raise ValueError("Recovered XML is truncated: closing root missing")
        if len(entries) != count or len(locs) != count:
            raise ValueError(f"XML recovery lost structure: expected {count} records, parsed {len(entries)}, original locs {len(locs)}")
        verified_entries = []
        quarantined = []
        discrepancies = []
        for index, loc in enumerate(locs):
            try:
                original = ET.fromstring(b"<loc>" + loc + b"</loc>")
            except ET.ParseError as exc:
                raw_url = loc.decode("utf-8").strip()
                invalid_controls = any(ord(c) < 32 and c not in "\t\r\n" for c in raw_url)
                for ref in re.findall(r"&#(x[0-9a-fA-F]+|[0-9]+);", raw_url):
                    code = int(ref[1:], 16) if ref.startswith("x") else int(ref)
                    invalid_controls |= code < 32 and code not in {9, 10, 13}
                if quarantine_invalid_urls and invalid_controls:
                    quarantined.append({"record_index": index, "original_loc": raw_url,
                                        "reason": "forbidden_control_character_in_loc"})
                    continue
                raise ValueError("Original loc URL itself contains malformed XML; recovery refused") from exc
            if len(original) or original.text is None:
                raise ValueError("Original loc URL has unexpected nested markup or is empty")
            original_url = original.text.strip()
            if not original_url or "\ufffd" in original_url or any(ord(c) < 32 for c in original_url):
                if quarantine_invalid_urls:
                    quarantined.append({"record_index": index, "original_loc": original_url,
                                        "reason": "invalid_character_in_loc_url"})
                    continue
                raise ValueError("Invalid characters in original loc URL")
            if original_url != entries[index]["url"]:
                discrepancies.append({"record_index": index, "original_url": original_url,
                                      "recovery_parser_url": entries[index]["url"]})
            # A source loc parsed independently in strict mode is authoritative;
            # do not use any URL rewritten by the recovering parser.
            verified_entries.append({**entries[index], "url": original_url})
        entries = verified_entries
        recovery.update(records=count, valid_records=len(entries), original_loc_urls_preserved=True,
                        quarantined_urls=quarantined, recovered_url_discrepancies=discrepancies)
        if recovery_log is not None:
            recovery_log.append(recovery)
    return kind, entries


def discover_sitemaps(client, profile, config, directory, max_pages):
    queue = deque(profile["sitemaps"])
    # Verified SAOstar monthly endpoints work even though the 2026 index redirects.
    if profile.get("monthly_sitemap_template"):
        from datetime import date
        end = date.fromisoformat(config["end_date"])
        first = date.fromisoformat(config["fallback_start_date"])
        primary_year = int(config["start_date"][:4])
        years = list(range(end.year, primary_year - 1, -1)) + list(range(primary_year - 1, first.year - 1, -1))
        queue.extend(profile["monthly_sitemap_template"].format(year=year, month=month)
                     for year in years
                     for month in range(1, 13) if (year, month) <= (end.year, end.month))
    visited = set()
    candidates = {}
    failures = []
    skipped = []
    recoveries = []
    cache = directory / "discovery_cache"
    cache.mkdir(parents=True, exist_ok=True)
    while queue and len(visited) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        year = archive_year(url)
        if year and not int(config["fallback_start_date"][:4]) <= year <= int(config["end_date"][:4]):
            continue
        if re.search(r"categor|topics|collection|sitemap-tags?\b", urlsplit(url).path, re.I):
            skipped.append(url)
            continue
        visited.add(url)
        try:
            key = hashlib.sha256(url.encode()).hexdigest()
            path = cache / (key + ".xml")
            if path.exists():
                body = path.read_bytes()
            else:
                body, _ = client.get(url)
                path.write_bytes(body)
            recovered = []
            kind, entries = parse_sitemap(body, allow_recovery=profile.get("recover_malformed_sitemaps", False),
                                          quarantine_invalid_urls=profile.get("quarantine_invalid_sitemap_urls", False),
                                          recovery_log=recovered)
            recoveries.extend({"url": url, **note} for note in recovered)
            if kind == "sitemapindex":
                queue.extend(entry["url"] for entry in entries)
            else:
                for entry in entries:
                    item_url = urldefrag(entry["url"])[0]
                    if client.allowed_host(item_url) and article_url(item_url):
                        candidate = {**entry, "url": item_url, "discovered_from": url,
                                     "archive_year_hint": year}
                        old = candidates.get(item_url)
                        if old is None or (not old.get("publication_date_hint") and entry.get("publication_date_hint")):
                            candidates[item_url] = candidate
        except Exception as exc:
            failures.append({"url": url, "error": str(exc)})
    pending = [url for url in queue if url not in visited]
    return list(candidates.values()), {
        "method": "sitemaps", "pages_visited": len(visited), "failures": failures,
        "budget_exhausted": bool(pending), "pending_pages": len(pending),
        "skipped_non_article_sitemaps": skipped,
        "xml_recoveries": recoveries,
        "inventory_traversal_complete": not pending and not failures,
        "coverage_note": "Inventory covers reachable configured article sitemaps; this does not prove every published article is included.",
    }


def discover_yan(client, config, directory, max_pages):
    """Exhaust all-news and category feeds linked by the site's own navigation."""
    candidates = {}
    hashes = set()
    failures = []
    feeds = deque([{"url": "https://www.yan.vn/", "zone": "0", "category": "0", "page": 0}])
    visited_feeds = set()
    ended_feeds = []
    skipped_media = []
    cache = directory / "discovery_cache"
    cache.mkdir(parents=True, exist_ok=True)
    visited = 0
    # Page-zero category HTML supplies zoneid/cate to the public JS endpoint.
    while feeds and visited < max_pages:
        feed = feeds.popleft()
        page = feed["page"]
        url = feed["url"] if page == 0 else (
            "https://www.yan.vn/ContentList/TinMoiLoadMore"
            f"?id={feed['zone']}&cat={feed['category']}&mobile=false&LoadMore={page}"
        )
        try:
            path = cache / ("yan-" + hashlib.sha256(url.encode()).hexdigest() + ".json")
            if path.exists():
                parsed = json.loads(path.read_text(encoding="utf-8"))
            else:
                body, _ = client.get(url)
                soup = BeautifulSoup(body.decode("utf-8"), "html.parser")
                scripts = "\n".join(node.get_text() for node in soup.find_all("script", src=False))
                zone = re.search(r"\b(?:let|var|const)\s+zoneid\s*=\s*['\"]?(\d+)", scripts)
                category = re.search(r"\b(?:let|var|const)\s+cate\s*=\s*['\"]?(\d+)", scripts)
                container = (soup.select_one("#LoadMore") if page == 0 else soup) or soup
                parsed = {"digest": hashlib.sha256(body).hexdigest(),
                          "empty": not body.strip() or (not soup.select("html") and not soup.get_text(strip=True)),
                          "zone": zone[1] if zone else None, "category": category[1] if category else "0",
                          "category_paths": sorted({a["href"] for a in soup.select("a[href]") if re.fullmatch(r"/[^/?#]+-\d+[zc]", a["href"])}),
                          "article_links": [a["href"] for a in container.select("a[href]")]}
                path.write_text(json.dumps(parsed, ensure_ascii=False), encoding="utf-8")
            visited += 1
            digest = (feed["url"], parsed["digest"])
            if digest in hashes and not parsed["empty"]:
                failures.append({"url": url, "error": "Repeated page; archive pagination may be capped"})
                continue
            hashes.add(digest)
            if page == 0:
                if feed["url"] == "https://www.yan.vn/":
                    for href in parsed["category_paths"]:
                        target = urljoin(url, href)
                        if target not in visited_feeds:
                            visited_feeds.add(target)
                            feeds.append({"url": target, "page": 0})
                if not parsed["zone"]:
                    failures.append({"url": url, "error": "Missing zoneid in category page; needs parser review"})
                    continue
                feed["zone"] = parsed["zone"]
                feed["category"] = parsed["category"]
            found = []
            for link in parsed["article_links"]:
                target = urldefrag(urljoin(url, link))[0]
                if client.allowed_host(target) and article_url(target):
                    found.append(target)
                    candidates.setdefault(target, {"url": target, "discovered_from": url,
                                                   "publication_date_hint": None,
                                                   "archive_year_hint": None})
            if not found and page > 0:
                # A valid empty response or the endpoint's empty template ends the feed.
                if parsed["empty"]:
                    ended_feeds.append(feed["url"])
                else:
                    failures.append({"url": url, "error": "No article links in nonempty response"})
                continue
            feed["page"] += 1
            # Finish a feed before the next so its site pagination remains ordered.
            feeds.appendleft(feed)
        except Exception as exc:
            failures.append({"url": url, "error": str(exc)})
        if visited and visited % 50 == 0:
            print(f"YAN discovery: pages={visited}, unique_urls={len(candidates)}, pending_feeds={len(feeds)}", flush=True)
    return list(candidates.values()), {
        "method": "yan_public_all_news_and_category_pagination", "pages_visited": visited,
        "failures": failures, "budget_exhausted": bool(feeds),
        "ended_feeds": ended_feeds, "skipped_media_feeds": skipped_media,
        "inventory_traversal_complete": not feeds and not failures,
        "coverage_note": "YAN has no working sitemap in survey. Exhausting reachable category feeds does not prove archive completeness; dates are checked on article pages.",
    }


def write_inventory(path, candidates, report):
    with path.open("w", encoding="utf-8") as stream:
        for candidate in sorted(candidates, key=lambda row: row["url"]):
            stream.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    report["unique_article_urls"] = len(candidates)
    path.with_name("discovery_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
