"""Cross-source exact + near-duplicate title grouping (dup_group) and representative choice.

Similarity is char-3gram Jaccard (JC3) on a comparison-only normalized title; the
display title is never modified. All pairs are scored exactly via sparse binary
matmul in chunks (no blocking).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from itertools import combinations

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

PUNCT = re.compile(r"[^\w\s]|_")
WORD = re.compile(r"\w+")


def norm_title(text):
    """NFC -> lower -> punctuation to space -> collapse whitespace (comparison key only)."""
    text = PUNCT.sub(" ", unicodedata.normalize("NFC", text or "").lower())
    return " ".join(text.split())


def char3(text):
    padded = f" {text} "
    return list({padded[k:k + 3] for k in range(len(padded) - 2)})


def jc3(a, b):
    """JC3 of two raw strings (normalized first); None when either side is empty."""
    sa, sb = set(char3(norm_title(a))) if a else set(), set(char3(norm_title(b))) if b else set()
    if not sa or not sb:
        return None
    return len(sa & sb) / len(sa | sb)


def word_containment(a, b):
    wa, wb = set(WORD.findall(norm_title(a))), set(WORD.findall(norm_title(b)))
    return len(wa & wb) / min(len(wa), len(wb)) if wa and wb else None


def number_tokens(title, pattern):
    return set(re.findall(pattern, title or ""))


def number_conflict(a, b, mode="both_sides"):
    """both_sides: each title has a number the other lacks; either_side: any unmatched number."""
    only_a, only_b = bool(a - b), bool(b - a)
    return (only_a and only_b) if mode == "both_sides" else (only_a or only_b)


def similar_pairs(texts, floor, chunk_size=500):
    """Yield (i, j, jc3) for i < j with JC3 >= floor over unique normalized texts.

    Intersections are integer counts and the ratio is float64, so a pair whose exact
    JC3 equals a threshold (e.g. 35/50 = 0.7) passes a `>=` check.
    """
    if len(texts) < 2:
        return
    matrix = CountVectorizer(analyzer=char3, binary=True, dtype=np.int32).fit_transform(texts).tocsr()
    sizes = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float64)
    transposed = matrix.T.tocsr()
    for start in range(0, len(texts), chunk_size):
        inter = (matrix[start:start + chunk_size] @ transposed).toarray().astype(np.float64)
        rows = np.arange(start, start + inter.shape[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            jac = np.nan_to_num(inter / (sizes[rows, None] + sizes[None, :] - inter))
        jac[np.arange(inter.shape[0])[:, None] >= (np.arange(len(texts))[None, :] - start)] = 0  # keep j > i
        for r, j in zip(*np.nonzero(jac >= floor)):
            yield int(rows[r]), int(j), float(jac[r, j])


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}
        self.members = {item: [item] for item in items}

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            keep, drop = min(ra, rb), max(ra, rb)
            self.parent[drop] = keep
            self.members[keep] += self.members.pop(drop)


def group_id(members):
    return "dg_" + hashlib.sha256("\n".join(sorted(members)).encode()).hexdigest()[:12]


def days_apart(a, b):
    ta, tb = a.get("published_sort"), b.get("published_sort")
    return None if ta is None or tb is None else abs(ta - tb) / 86400


def find_duplicates(records, config, decisions=()):
    """Group records by title duplication.

    records: dicts with record_id, source, title, sapo, published_sort (epoch seconds).
    decisions: dicts with record_id_a, record_id_b, decision ('link'|'separate').
    Returns (group_of, auto_pairs, review_pairs, template_pairs, notes).

    Near links (non-exact) also need |published_at gap| <= near_link_max_days. Auto
    links are applied in a fixed order (exact first, then JC3 desc) with cannot-link
    constraints: a link is refused when it would put two records with a number
    conflict, or a manual 'separate' pair, into one group (no bridging through a
    third title). Manual 'link' decisions are applied last and always honoured.
    """
    by_id = {r["record_id"]: r for r in records}
    pattern, mode = config["number_pattern"], config["number_conflict"]
    window = config.get("near_link_max_days")
    check_below = config.get("review_auto_link_below", 0)
    numbers = {rid: number_tokens(r["title"], pattern) for rid, r in by_id.items()}
    members = defaultdict(list)
    for rid in sorted(by_id):
        members[norm_title(by_id[rid]["title"])].append(rid)

    def pair_row(a, b, score, rule):
        a, b = sorted((a, b))
        ra, rb = by_id[a], by_id[b]
        gap = days_apart(ra, rb)
        return {"record_id_a": a, "record_id_b": b, "source_a": ra["source"], "source_b": rb["source"],
                "jc3": round(score, 4), "number_conflict": number_conflict(numbers[a], numbers[b], mode),
                "days_apart": None if gap is None else round(gap, 1), "rule": rule,
                "title_a": ra["title"], "title_b": rb["title"]}

    def in_window(a, b):
        gap = days_apart(by_id[a], by_id[b])
        return window is None or gap is None or gap <= window

    keys = [key for key in sorted(members) if key]  # titles with no word characters are never compared
    auto, review, template = [], [], []
    for key in keys:
        for a, b in combinations(members[key], 2):
            auto.append(pair_row(a, b, 1.0, "exact_norm_title"))
    floor = min(config["floor"], config["review_low"])
    for i, j, score in similar_pairs(keys, floor, config.get("chunk_size", 500)):
        for a in members[keys[i]]:
            for b in members[keys[j]]:
                cross = by_id[a]["source"] != by_id[b]["source"]
                conflict = number_conflict(numbers[a], numbers[b], mode)
                rule = ("cross_source_jc3" if cross and score >= config["cross_source_link"] else
                        "any_source_jc3" if score >= config["any_source_link"] else None)
                if rule and not conflict and in_window(a, b):
                    auto.append(pair_row(a, b, score, rule))
                elif cross:
                    if rule:  # above a link threshold but blocked
                        review.append({**pair_row(a, b, score, "number_conflict" if conflict else "date_gap"),
                                       "auto_linked": False})
                    elif score >= config["review_low"]:
                        review.append({**pair_row(a, b, score, "review_band"), "auto_linked": False})
                elif score >= config["template_series_min"]:
                    template.append(pair_row(a, b, score, "within_source_template"))

    separate, notes = set(), []
    for decision in decisions:
        a, b = sorted((decision["record_id_a"], decision["record_id_b"]))
        if a not in by_id or b not in by_id:
            notes.append(f"decision skipped, record not in valid pool: {a} / {b}")
        elif decision["decision"] == "separate":
            separate.add((a, b))

    def cannot_link(x, y):
        return number_conflict(numbers[x], numbers[y], mode) or tuple(sorted((x, y))) in separate

    union = UnionFind(by_id)
    auto.sort(key=lambda r: (r["rule"] != "exact_norm_title", -r["jc3"], r["record_id_a"], r["record_id_b"]))
    for row in auto:
        a, b = row["record_id_a"], row["record_id_b"]
        row["applied"], row["blocked_reason"] = True, None
        ra, rb = union.find(a), union.find(b)
        if (a, b) in separate:
            row["applied"], row["blocked_reason"] = False, "separate_decision"
        elif ra != rb and any(cannot_link(x, y) for x in union.members[ra] for y in union.members[rb]):
            row["applied"], row["blocked_reason"] = False, "transitive_conflict"
            review.append({**row, "rule": "transitive_conflict", "auto_linked": False})
        else:
            union.union(a, b)
            if row["rule"] == "exact_norm_title" and not in_window(a, b):
                review.append({**row, "rule": "exact_date_gap", "auto_linked": True})
            elif row["rule"] != "exact_norm_title" and row["jc3"] < check_below:
                review.append({**row, "rule": "auto_link_check", "auto_linked": True})
    for decision in decisions:
        a, b = sorted((decision["record_id_a"], decision["record_id_b"]))
        if decision["decision"] == "link" and a in by_id and b in by_id:
            union.union(a, b)
            auto.append({**pair_row(a, b, jc3(by_id[a]["title"], by_id[b]["title"]) or 0.0, "manual_link"),
                         "applied": True, "blocked_reason": None})
    for a, b in sorted(separate):
        if union.find(a) == union.find(b):
            notes.append(f"separate decision not honoured: {a} / {b} share a group via a manual link")

    components = defaultdict(list)
    for rid in by_id:
        components[union.find(rid)].append(rid)
    group_of = {}
    for ids in components.values():
        gid = group_id(ids)
        for rid in ids:
            group_of[rid] = (gid, len(ids))
    for row in review:
        row["same_group_already"] = group_of[row["record_id_a"]][0] == group_of[row["record_id_b"]][0]
        sapo = jc3(by_id[row["record_id_a"]].get("sapo"), by_id[row["record_id_b"]].get("sapo"))
        containment = word_containment(row["title_a"], row["title_b"])
        row["sapo_jc3"] = None if sapo is None else round(sapo, 4)
        row["word_containment"] = None if containment is None else round(containment, 4)
    series = UnionFind({rid for row in template for rid in (row["record_id_a"], row["record_id_b"])})
    for row in template:
        series.union(row["record_id_a"], row["record_id_b"])
    for row in template:
        root = series.find(row["record_id_a"])
        row["series_id"] = "ts_" + hashlib.sha256(root.encode()).hexdigest()[:12]
    return group_of, auto, review, template, notes


def choose_representatives(records, group_of, unseen_sources):
    """Return {record_id: (is_representative, excluded_reason)}.

    Group with an unseen-source record: keep one unseen copy (earliest published_at,
    then record_id); every seen copy -> 'duplicate_of_unseen'. Otherwise keep the
    earliest published_at, then record_id; the rest -> 'duplicate_in_group'.
    """
    groups = defaultdict(list)
    for record in records:
        groups[group_of[record["record_id"]][0]].append(record)
    result = {}
    for members in groups.values():
        key = lambda r: (r["published_sort"], r["record_id"])
        unseen = [r for r in members if r["source"] in unseen_sources]
        keep = min(unseen or members, key=key)["record_id"]
        for r in members:
            if r["record_id"] == keep:
                result[r["record_id"]] = (True, None)
            elif unseen and r["source"] not in unseen_sources:
                result[r["record_id"]] = (False, "duplicate_of_unseen")
            else:
                result[r["record_id"]] = (False, "duplicate_in_group")
    return result
