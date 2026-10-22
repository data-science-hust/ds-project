"""Extract original visible headlines/leads with evidence, never SEO fallbacks."""

import json
import re
from html import unescape
from datetime import datetime

from bs4 import BeautifulSoup


def flatten(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from flatten(child)
    elif isinstance(value, list):
        for child in value:
            yield from flatten(child)


def parse_datetime(value):
    if not isinstance(value, str):
        return None
    # Some source JSON-LD uses +7:00 or a space before the explicit offset.
    value = re.sub(r"\s+([+-]\d{1,2}:\d{2})$", r"\1", unescape(value).strip())
    value = re.sub(r"([+-])(\d):(\d{2})$", r"\g<1>0\2:\3", value)
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return result if result.tzinfo is not None else None
    except ValueError:
        return None


def extract(html, url, profile):
    soup = BeautifulSoup(html.decode("utf-8"), "html.parser")
    evidence = {}
    warnings = []

    def visible(field):
        selector = profile.get(field)
        node = soup.select_one(selector) if selector else None
        if node is None:
            return None
        text = node.get_text(" ", strip=True)
        # YAN has no lead element; only a first paragraph bolded in full is a lead.
        if field == "sapo" and profile.get("sapo_require_full_bold_paragraph"):
            bold = " ".join(child.get_text(" ", strip=True) for child in node.find_all(["strong", "b"], recursive=False))
            if " ".join(bold.split()) != " ".join(text.split()):
                return None
        if not text:
            return None
        evidence[field] = {"method": "css", "selector": selector}
        return text

    nodes = []
    documents = []
    for script in soup.select('script[type="application/ld+json"]'):
        text = script.string or script.get_text()
        try:
            obj = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            warnings.append("invalid_jsonld")
            # Verified Kênh14 breadcrumb omits a comma between @id and name.
            # Repair syntax in memory only; never alter source strings/headlines.
            if profile.get("repair_breadcrumb_missing_comma") and '"BreadcrumbList"' in text:
                repaired = re.sub(r'("@id"\s*:\s*"(?:[^"\\]|\\.)*")\s+("name"\s*:)', r'\1,\2', text)
                try:
                    obj = json.loads(repaired)
                    warnings.append("breadcrumb_missing_comma_repaired")
                except json.JSONDecodeError:
                    continue
            else:
                continue
        documents.append(obj)
        nodes.extend(flatten(obj))
    # Only top-level article objects; never use dates from related ItemList entries.
    articles = []
    for item in documents:
        items = item if isinstance(item, list) else [item]
        expanded = []
        for obj in items:
            if isinstance(obj, dict):
                expanded.extend(obj.get("@graph", [obj]))
        articles.extend(obj for obj in expanded if isinstance(obj, dict) and
                        any(t in {"NewsArticle", "Article", "ReportageNewsArticle", "LiveBlogPosting", "VideoObject", "ImageGallery"}
                            for t in (obj.get("@type", []) if isinstance(obj.get("@type"), list) else [obj.get("@type")])))
    article = articles[0] if articles else {}
    title = visible("title")
    sapo = visible("sapo")
    og = soup.select_one('meta[property="og:type"]')
    is_article_page = bool(articles or (og and (og.get("content") == "article" or str(og.get("content", "")).startswith("video"))))
    if not title and is_article_page:
        heading = soup.select_one("h1")
        if heading:
            title = heading.get_text(" ", strip=True) or None
            evidence["title"] = {"method": "css", "selector": "h1", "article_metadata_verified": True}
    category = None
    meta_section = soup.select_one('meta[property="article:section"]')
    if meta_section and meta_section.get("content"):
        category = meta_section["content"].strip()
        evidence["category"] = {"method": "meta", "property": "article:section"}
    elif isinstance(article.get("articleSection"), str):
        category = article["articleSection"]
        evidence["category"] = {"method": "jsonld", "field": "articleSection"}
    else:
        crumbs = [n for n in nodes if n.get("@type") == "BreadcrumbList"]
        if crumbs:
            items = [n for n in flatten(crumbs[0].get("itemListElement", [])) if n.get("@type") == "ListItem"]
            if items:
                last = items[-1]
                category = last.get("name") or (last.get("item", {}).get("name") if isinstance(last.get("item"), dict) else None)
                evidence["category"] = {"method": "jsonld", "field": "BreadcrumbList"}
        if not category:
            selector = profile.get("category")
            items = soup.select(selector) if selector else []
            if items:
                category = items[-1].get_text(" ", strip=True) or None
                evidence["category"] = {"method": "css", "selector": selector}

    date_candidates = [article.get("datePublished")]
    for node in soup.select('meta[property="article:published_time"], meta[itemprop="datePublished"], time[datetime]'):
        date_candidates.append(node.get("content") or node.get("datetime"))
    published = next((parse_datetime(v) for v in date_candidates if parse_datetime(v)), None)
    offset = profile.get("naive_timestamp_offset")
    if published is None and offset:
        # Only sources with an explicitly approved profile policy may assume a zone.
        zone = datetime.fromisoformat("2000-01-01T00:00:00" + offset).tzinfo
        for value in date_candidates:
            if not isinstance(value, str) or not re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", value):
                continue
            try:
                naive = datetime.fromisoformat(unescape(value).strip())
            except ValueError:
                continue
            if naive.tzinfo is None:
                published = naive.replace(tzinfo=zone)
                warnings.append("timezone_assumed_by_approved_source_policy")
                evidence["timestamp_timezone_policy"] = {"offset": offset, "policy": profile.get("timezone_policy")}
                break
    if published:
        evidence["published_at"] = {"method": "source_timestamp", "raw_candidates": date_candidates}

    return {"title": title, "sapo": sapo, "category": category,
            "published_at": published.isoformat() if published else None,
            "published_at_raw": date_candidates, "is_article_page": is_article_page,
            "extraction_evidence": evidence, "extraction_warnings": warnings}
