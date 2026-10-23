"""Record checks, technical fixes, category join and dedup -> clean table and funnel counts.

Every rejection, flag and fix is logged by record_id; the raw input is only read.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

from .categories import lookup, normalize_category
from .dedup import choose_representatives, find_duplicates

FIELDS = ("record_id", "source", "url", "title", "sapo", "category", "published_at", "crawled_at", "extraction_status")
REQUIRED = ("record_id", "source", "url", "title", "published_at", "crawled_at", "extraction_status")
DOWNLOADED_STATUSES = {"success", "partial"}
CLEAN_COLUMNS = ("record_id", "source", "outlet_group", "url", "title", "title_raw", "sapo", "category", "category_raw",
                 "category_mapping_status", "published_at", "crawled_at", "extraction_status", "dup_group",
                 "dup_group_size", "is_dup_representative", "dedup_excluded_reason", "has_sapo", "source_run_id")
ISSUE_FIELDS = ("record_id", "source", "issue", "field", "decision", "reason")
FIX_FIELDS = ("record_id", "source", "field", "before", "after", "reason")
UNMAPPED = "<unmapped>"
VN = timezone(timedelta(hours=7))
# UTF-8 bytes of Vietnamese letters decoded as cp1252/latin-1: lead char + a byte-range char.
MOJIBAKE = re.compile("(?:Ã|Â|Ä|Æ|á»|áº|á¼|á½)[\u0080-¿ŒœŠšŸŽžƒ"
                      "ˆ˜–—‘-„†-•…‰‹›€™]")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mojibake_status(text):
    """None, 'suspected', or 'confirmed' when a cp1252/latin-1 round-trip yields clean UTF-8."""
    if not isinstance(text, str) or not MOJIBAKE.search(text):
        return None
    for encoding in ("cp1252", "latin-1"):
        try:
            repaired = text.encode(encoding).decode("utf-8")
        except UnicodeError:
            continue
        if repaired != text and not MOJIBAKE.search(repaired):
            return "confirmed"
    return "suspected"


def parse_time(value):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None


def host_matches(url, domain):
    host = urlsplit(url).hostname or ""
    return host == domain or host.endswith("." + domain)


def load_rules(path):
    rules = json.loads(Path(path).read_text(encoding="utf-8"))["rules"]
    for rule in rules:
        if rule["match"] not in {"exact", "regex", "missing_sapo"}:
            raise ValueError(f"rule {rule['id']}: match must be exact, regex or missing_sapo")
        rule["compiled"] = re.compile(rule["pattern"]) if rule["match"] == "regex" else None
    return rules


def rule_hits(rules, source, title, category_raw):
    hits = []
    for rule in rules:
        if rule.get("source") not in (None, source) or rule["match"] == "missing_sapo":
            continue
        if rule.get("category_contains") and rule["category_contains"] not in (category_raw or ""):
            continue
        if (title == rule["pattern"]) if rule["match"] == "exact" else bool(rule["compiled"].match(title)):
            hits.append(rule)
    return hits


def read_raw(path, sources):
    """Return {source: rows in file order}; ValueError on a broken file."""
    pools = {}
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"{path}:{number}: invalid JSON line; re-run scripts/export_raw.py") from None
            if not isinstance(row, dict) or row.get("source") not in sources:
                raise ValueError(f"{path}:{number}: not an object with a source from sources.csv")
            pools.setdefault(row["source"], []).append(row)
    return pools


def read_crawl_stats(rows, sources):
    """crawl_stats.csv rows -> {source: stats}; ValueError when a source is missing or counts disagree."""
    stats = {}
    for row in rows:
        by_category = [(category, int(count)) for category, count in json.loads(row["downloaded_by_category"])]
        stats[row["source"]] = {"run_id": row["run_id"], "downloaded": int(row["downloaded"]),
                                "downloaded_without_title": int(row["downloaded_without_title"]),
                                "selected": int(row["selected"]), "downloaded_by_category": by_category}
        if sum(count for _, count in by_category) != stats[row["source"]]["downloaded"]:
            raise ValueError(f"crawl_stats.csv: {row['source']} downloaded_by_category does not sum to downloaded")
    missing = [source for source in sources if source not in stats]
    if missing:
        raise ValueError(f"crawl_stats.csv has no row for {missing}; re-run scripts/export_raw.py")
    return stats


def check_and_fix(pool, source, domain, rules, mapping):
    """Return (valid, rejected, issues, fixes) for one source's pool."""
    valid, rejected, issues, fixes = [], [], [], []
    seen_ids, seen_urls = set(), set()
    for row in pool:
        rid = row.get("record_id") if isinstance(row.get("record_id"), str) else ""
        found = []

        def issue(kind, field, decision, reason=""):
            found.append({"record_id": rid, "source": source, "issue": kind, "field": field,
                          "decision": decision, "reason": reason})

        for field in FIELDS:
            value = row.get(field)
            if field not in row:
                issue("missing_field", field, "reject" if field in REQUIRED else "flag", "field absent")
            elif value is not None and not isinstance(value, str):
                issue("invalid_type", field, "reject", type(value).__name__)
            elif field in REQUIRED and (value is None or not value.strip()):
                issue("missing_value", field, "reject", "empty or whitespace")
        text = {f: row.get(f) if isinstance(row.get(f), str) else None for f in FIELDS}
        if text["source"] and text["source"] != source:
            issue("source_mismatch", "source", "reject", f"expected {source}")
        url = text["url"]
        if url and url.strip():
            try:
                parts = urlsplit(url)
                if parts.scheme not in {"http", "https"} or not parts.hostname or any(c.isspace() for c in url):
                    issue("invalid_url", "url", "reject", "not an absolute http(s) URL")
                elif not host_matches(url, domain):
                    issue("unexpected_host", "url", "reject", f"{parts.hostname} not under {domain}")
            except ValueError as exc:
                issue("invalid_url", "url", "reject", str(exc))
        published = parse_time(text["published_at"]) if text["published_at"] else None
        if text["published_at"] and published is None:
            issue("invalid_published_at", "published_at", "reject", "not ISO 8601")
        elif published is not None and published.tzinfo is None:
            issue("published_at_without_timezone", "published_at", "flag", "ordering assumes UTC+07:00 only")
        if text["crawled_at"] and parse_time(text["crawled_at"]) is None:
            issue("invalid_crawled_at", "crawled_at", "flag", "not ISO 8601")
        if text["extraction_status"] and text["extraction_status"] not in DOWNLOADED_STATUSES:
            issue("extraction_not_usable", "extraction_status", "reject", text["extraction_status"])

        title = unicodedata.normalize("NFC", text["title"]).strip() if text["title"] else None
        sapo = unicodedata.normalize("NFC", text["sapo"]).strip() if text["sapo"] else None
        sapo = sapo or None
        category_raw = normalize_category(text["category"])
        for field, before, after, reason in (
            ("title", text["title"], title, "strip outer whitespace + NFC; wording unchanged"),
            ("sapo", text["sapo"], sapo, "strip outer whitespace + NFC; wording unchanged"),
            ("category_raw", text["category"], category_raw, "html.unescape + NFC + strip; empty -> null"),
        ):
            if before is not None and before != after:
                fixes.append({"record_id": rid, "source": source, "field": field, "before": before,
                              "after": after, "reason": reason})

        for field, value in (("title", title), ("sapo", sapo), ("category", category_raw)):
            if value and "�" in value:
                issue("encoding_replacement_character", field, "reject" if field == "title" else "flag",
                      "U+FFFD: characters lost at decode time")
            status = mojibake_status(value)
            if status:
                confirmed_title = status == "confirmed" and field == "title"
                issue(f"mojibake_{status}", field, "reject" if confirmed_title else "flag",
                      "cp1252/latin-1 round-trip decodes to UTF-8" if status == "confirmed" else "pattern only; review")
        if rid in seen_ids:
            issue("duplicate_record_id", "record_id", "reject", "record_id already in pool; first row kept")
        if url in seen_urls:
            issue("duplicate_url", "url", "reject", "URL already in pool; first row kept")
        if title:
            for rule in rule_hits(rules, source, title, category_raw):
                issue("non_headline_title", "title", "reject", f"{rule['id']}: {rule['reason']}")
        if not sapo:
            policy = [r for r in rules if r["match"] == "missing_sapo" and r.get("source") in (None, source)]
            for rule in policy:
                issue("missing_sapo_policy", "sapo", "reject", f"{rule['id']}: {rule['reason']}")
            if not policy:
                issue("missing_sapo", "sapo", "report", "kept; counted in funnel")
        category, mapping_status, why = lookup(mapping, source, category_raw)
        if category is None:
            issue("category_unmapped", "category", "report", why)

        issues.extend(found)
        reasons = [f["issue"] for f in found if f["decision"] == "reject"]
        if reasons:
            rejected.append({**{f: row.get(f) for f in FIELDS}, "reject_issues": ";".join(dict.fromkeys(reasons)),
                             "reject_reasons": " | ".join(f["reason"] for f in found if f["decision"] == "reject")})
            continue
        seen_ids.add(rid)
        seen_urls.add(url)
        if published.tzinfo is None:
            published = published.replace(tzinfo=VN)
        valid.append({"record_id": rid, "source": source, "url": url, "title": title, "title_raw": text["title"],
                      "sapo": sapo, "category": category, "category_raw": category_raw,
                      "category_mapping_status": mapping_status, "published_at": text["published_at"],
                      "crawled_at": text["crawled_at"], "extraction_status": text["extraction_status"],
                      "has_sapo": sapo is not None, "published_sort": published.timestamp()})
    return valid, rejected, issues, fixes


def apply_dedup(valid, config, decisions, unseen_sources):
    group_of, auto, review, template, notes = find_duplicates(valid, config, decisions)
    chosen = choose_representatives(valid, group_of, unseen_sources)
    for record in valid:
        record["dup_group"], record["dup_group_size"] = group_of[record["record_id"]]
        record["is_dup_representative"], record["dedup_excluded_reason"] = chosen[record["record_id"]]
    return auto, review, template, notes


def funnel(sources, crawl_stats, pool_sizes, valid, mapping):
    """Return (by_source, by_category) funnel rows: downloaded -> valid -> after_dedup -> with_sapo."""
    by_source, by_category = [], []
    cat_counts = defaultdict(Counter)
    # downloaded covers every crawl attempt, so it is mapped from crawl_stats rather than the valid pool.
    for source, stats in crawl_stats.items():
        for category_raw, count in stats["downloaded_by_category"]:
            category = lookup(mapping, source, normalize_category(category_raw))[0]
            cat_counts[(source, category or UNMAPPED)]["downloaded"] += count
    for record in valid:
        key = (record["source"], record["category"] or UNMAPPED)
        kept = record["is_dup_representative"] and not record["dedup_excluded_reason"]
        cat_counts[key]["valid"] += 1
        cat_counts[key]["after_dedup"] += kept
        cat_counts[key]["with_sapo"] += kept and record["has_sapo"]
    for source, config in sources.items():
        totals = Counter()
        for (s, _), counter in cat_counts.items():
            if s == source:
                totals.update(counter)
        target = int(config["min_clean_articles"])
        excluded = sum(1 for r in valid if r["source"] == source and r["dedup_excluded_reason"])
        by_source.append({"source": source, "outlet_group": config["outlet_group"],
                          "run_id": crawl_stats[source]["run_id"], "downloaded": totals["downloaded"],
                          "downloaded_without_title": crawl_stats[source]["downloaded_without_title"],
                          "selected": pool_sizes.get(source, 0), "valid": totals["valid"],
                          "rejected": pool_sizes.get(source, 0) - totals["valid"], "dedup_excluded": excluded,
                          "after_dedup": totals["after_dedup"], "with_sapo": totals["with_sapo"],
                          "min_clean_articles": target, "shortfall": max(0, target - totals["after_dedup"])})
    for (source, category), counter in sorted(cat_counts.items()):
        by_category.append({"source": source, "category": category,
                            **{k: counter[k] for k in ("downloaded", "valid", "after_dedup", "with_sapo")}})
    return by_source, by_category
