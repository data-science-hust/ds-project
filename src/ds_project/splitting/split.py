"""Group-aware stratified split, reserve list and blind annotation tables (plan 1.2).

Input is the clean table (data/processed/articles_clean.parquet), the near-duplicate pairs the clean step
listed but did not merge (data/interim/{template_series,dedup_review,dedup_pairs}.csv) and the manual dedup
decisions (configs/checking/dedup_decisions.csv); all are only read. Everything is deterministic for one seed: the
string seeds of random.Random are hashed with SHA-512, so results do not depend on PYTHONHASHSEED.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from random import Random

SEEN_SPLITS = ("dev", "test_seen", "train")  # assignment order; dev first (needs sapo), train last
SPLITS = ("train", "dev", "test_seen", "test_unseen")
SUBSET = "human_subset"  # inside train, not a split
TARGETS = {"train": "train_target", "dev": "dev_target", "test_seen": "test_seen_target",
           "test_unseen": "test_unseen_target", SUBSET: "human_train_subset_target"}
NO_CATEGORY = "(không có category)"
MANIFEST_FIELDS = ("record_id", "source", "split", "dup_group", "is_human_subset", "seed", "split_group", "category",
                   "outlet_group")
RESERVE_FIELDS = ("record_id", "source", "category", "dup_group", "split_group", "reserve_rank", "has_sapo",
                  "hold_reason")
HOLD_UNSEEN = "pending_dedup_review_with_unseen"  # unmerged pair with an unseen record, not decided 'separate'
HOLD_NO_CATEGORY = "no_category"
PAIR_KINDS = ("template_series", "dedup_review", "dedup_blocked")  # precedence when one pair is listed twice
MAP_FIELDS = ("annotation_id", "record_id", "split", "source")
BLIND_FIELDS = ("annotation_id", "title")


class SplitError(ValueError):
    """The pool or the quotas do not allow the exact split; the message carries the numbers."""


def largest_remainder(total, counts):
    """Split integer `total` over keys proportionally to `counts` (exact integer arithmetic).

    Floors first; the leftover units go to the largest remainders, ties broken by key name."""
    size = sum(counts.values())
    if not total:
        return {key: 0 for key in counts}
    if size <= 0:
        raise SplitError(f"cannot allocate {total} over an empty pool")
    shares = {key: total * n // size for key, n in counts.items()}
    order = sorted(counts, key=lambda key: (-(total * counts[key] % size), key))
    for key in order[:total - sum(shares.values())]:
        shares[key] += 1
    return shares


def eligible_pool(records):
    """Dup representatives without dedup exclusion; refuses categories whose mapping is not approved."""
    pool = [r for r in records if r["is_dup_representative"] and not r["dedup_excluded_reason"]]
    bad = Counter((r["source"], r["category"], r["category_mapping_status"]) for r in pool
                  if r["category"] is not None and r["category_mapping_status"] != "approved")
    if bad:
        shown = ", ".join(f"{s}/{c}/{st}: {n}" for (s, c, st), n in sorted(bad.items(), key=str)[:5])
        raise SplitError(f"{sum(bad.values())} eligible records have a category mapping that is not approved "
                         f"({shown}); approve configs/crawling/category_mapping.csv and re-run build_clean.py")
    return pool


def unmerged_pairs(records, pairs_by_kind):
    """Near-duplicate pairs whose records the clean step left in different dup_groups.

    `pairs_by_kind` is {kind: rows with record_id_a/record_id_b} (template_series.csv, dedup_review.csv,
    dedup_pairs.csv); returns [{kind, record_id_a, record_id_b}] sorted, one row per record pair."""
    group_of = {r["record_id"]: r["dup_group"] for r in records}
    found = {}
    for kind in PAIR_KINDS:
        for pair in pairs_by_kind.get(kind, ()):
            a, b = pair["record_id_a"], pair["record_id_b"]
            missing = [x for x in (a, b) if x not in group_of]
            if missing:
                raise SplitError(f"{kind} names record_id {missing[0][:16]}... not in the clean table; "
                                 "re-run build_clean.py")
            if group_of[a] != group_of[b]:
                found.setdefault(tuple(sorted((a, b))), kind)
    return [{"kind": kind, "record_id_a": a, "record_id_b": b} for (a, b), kind in sorted(found.items())]


def record_sides(records):
    """{record_id: outlet_group of its dup_group's representative} (seen/unseen side used for pairs)."""
    side = {r["dup_group"]: r["outlet_group"] for r in records}
    side.update({r["dup_group"]: r["outlet_group"] for r in records if r["is_dup_representative"]})
    return {r["record_id"]: side[r["dup_group"]] for r in records}


def build_split_groups(records, pairs, separate=frozenset()):
    """({dup_group: split_group}, {held dup_group}) from the unmerged near-duplicate pairs.

    dup_groups linked by a pair on the same outlet side share one split_group (union-find), so near-identical
    titles never sit in two splits. A seen-unseen pair cannot share a split: unless the pair was decided
    'separate' (`separate` = {sorted record_id pair}), the seen side is held out of the splits until the dedup
    review decides it (a 'link' would exclude it as duplicate_of_unseen). `records` are all clean rows (a pair
    may name a non-representative; its dup_group is used, and the side is that of the group's representative,
    so a seen copy inside an unseen-led dup_group counts as unseen)."""
    group_of = {r["record_id"]: r["dup_group"] for r in records}
    outlet_of = record_sides(records)
    parent, held = {}, set()

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for pair in pairs:
        ids = (pair["record_id_a"], pair["record_id_b"])
        missing = [x for x in ids if x not in group_of]
        if missing:
            raise SplitError(f"pair names record_id {missing[0][:16]}... not in the clean table; re-run build_clean.py")
        if outlet_of[ids[0]] != outlet_of[ids[1]]:
            if tuple(sorted(ids)) not in separate:
                held.update(group_of[x] for x in ids if outlet_of[x] != "unseen")
            continue
        a, b = find(group_of[ids[0]]), find(group_of[ids[1]])
        if a != b:
            parent[max(a, b)] = min(a, b)
    members = defaultdict(list)
    for dup_group in sorted(set(group_of.values())):
        members[find(dup_group)].append(dup_group)
    split_group = {}
    for dup_groups in members.values():
        sid = "sg_" + hashlib.sha256("\n".join(dup_groups).encode("utf-8")).hexdigest()[:12]
        split_group.update({g: sid for g in dup_groups})
    return split_group, held


def stratum(record):
    return record["category"] if record["category"] is not None else NO_CATEGORY


def plan_quotas(selectable, sources):
    """{(source, split, category): records} with largest-remainder shares of the selectable records per source.

    `sources` is {source: sources.csv row} in config order; `selectable` excludes null-category and held units."""
    counts = defaultdict(Counter)
    for r in selectable:
        counts[r["source"]][r["category"]] += 1
    quotas = {}
    for source, row in sources.items():
        for split, column in TARGETS.items():
            target = int(row[column])
            if target and not counts[source]:
                raise SplitError(f"{source}/{split}: target {target} but 0 selectable records in the pool")
            for category, n in largest_remainder(target, counts[source]).items():
                quotas[(source, split, category)] = n
    problems = []
    sapo = Counter((r["source"], r["category"]) for r in selectable if r["has_sapo"])
    for source in sources:
        for category, available in sorted(counts[source].items()):
            q = {split: quotas[(source, split, category)] for split in TARGETS}
            need = sum(q[s] for s in SPLITS)
            if need > available:
                problems.append(f"{source}/{category}: quota {need} > {available} selectable records")
            if q["dev"] > sapo[(source, category)]:
                problems.append(f"{source}/{category}: dev quota {q['dev']} > {sapo[(source, category)]} "
                                "records with sapo")
            if q[SUBSET] > q["train"]:
                problems.append(f"{source}/{category}: human subset {q[SUBSET]} > train {q['train']}")
    if problems:
        raise SplitError("quota plan does not fit the pool: " + "; ".join(problems))
    return quotas


def cell_orders(units, seed, salt=""):
    """{(source, stratum): [split_group, ...]} shuffled with Random(f"{seed}:{source}:{stratum}{salt}").

    `units` is {split_group: member records}; a unit appears in every cell it has a record in."""
    cells = defaultdict(set)
    for unit, members in units.items():
        for r in members:
            cells[(r["source"], stratum(r))].add(unit)
    orders = {}
    for (source, category), group_ids in sorted(cells.items()):
        order = sorted(group_ids)
        Random(f"{seed}:{source}:{category}{salt}").shuffle(order)
        orders[(source, category)] = order
    return orders


def walk_order(orders, source_rank):
    """All units in one sequence: each cell advances at the same relative pace (a unit is reached at its
    earliest fractional position over its cells), so units spanning cells compete fairly for capacity."""
    first = {}
    for (source, category), order in orders.items():
        for index, unit in enumerate(order):
            # float is exact enough: distinct (2i+1)/(2n) with n <= 10^4 differ by far more than float rounding
            key = ((2 * index + 1) / (2 * len(order)), source_rank[source], category, index)
            if unit not in first or key < first[unit]:
                first[unit] = key
    return sorted(first, key=lambda unit: (first[unit], unit))


def fill(split, walk, units, quotas, taken, require_sapo=False):
    """Take units in walk order when every cell they touch has room; returns the unfilled remainder."""
    remaining = {(s, c): n for (s, sp, c), n in quotas.items() if sp == split and n}
    for unit in walk:
        members = units[unit]
        if unit in taken or (require_sapo and not all(r["has_sapo"] for r in members)):
            continue
        need = Counter((r["source"], r["category"]) for r in members)
        if all(remaining.get(cell, 0) >= n for cell, n in need.items()):
            taken[unit] = split
            for cell, n in need.items():
                remaining[cell] -= n
    return {cell: n for cell, n in remaining.items() if n}


def allocate(pool, split_group_of, sources, seed, held=frozenset()):
    """Assign split_groups to splits with exact record quotas; units with a held dup_group or a null category
    are not selectable and stay in reserve with a hold_reason.

    Returns (assignment {record_id: split}, subset {record_id}, reserve rows, quotas, units, selectable ids)."""
    source_rank = {source: i for i, source in enumerate(sources)}
    units = defaultdict(list)
    for r in sorted(pool, key=lambda r: r["record_id"]):
        units[split_group_of[r["dup_group"]]].append(r)
    units = dict(units)
    outlets = {u: {sources[r["source"]]["outlet_group"] for r in m} for u, m in units.items()}
    mixed = [u for u, o in outlets.items() if len(o) > 1]
    if mixed:
        raise SplitError(f"{len(mixed)} split_groups mix seen and unseen sources (e.g. {mixed[0]}); fix dedup first")
    hold = {}
    for u, members in units.items():
        if any(r["dup_group"] in held for r in members):
            hold[u] = HOLD_UNSEEN
        elif any(r["category"] is None for r in members):
            hold[u] = HOLD_NO_CATEGORY
    selectable = {u: m for u, m in units.items() if u not in hold}
    quotas = plan_quotas([r for m in selectable.values() for r in m], sources)
    orders = cell_orders(units, seed)
    walk = [u for u in walk_order(orders, source_rank) if u in selectable]
    taken = {}
    pool_sapo = Counter((r["source"], r["category"]) for r in pool if r["has_sapo"])
    for split in (*SEEN_SPLITS, "test_unseen"):
        short = fill(split, walk, selectable, quotas, taken, require_sapo=split == "dev")
        if short:
            detail = "; ".join(f"{s}/{c}: {n} of {quotas[(s, split, c)]} missing"
                               + (f" ({pool_sapo[(s, c)]} records with sapo in pool)" if split == "dev" else "")
                               for (s, c), n in sorted(short.items()))
            raise SplitError(f"cannot fill {split} exactly with whole split_groups: {detail}")
    train_units = {u: selectable[u] for u, split in taken.items() if split == "train"}
    subset_taken = {}
    short = fill(SUBSET, walk_order(cell_orders(train_units, seed, f":{SUBSET}"), source_rank), train_units,
                 quotas, subset_taken)
    if short:
        detail = "; ".join(f"{s}/{c}: {n} missing" for (s, c), n in sorted(short.items()))
        raise SplitError(f"cannot fill the human subset exactly inside train: {detail}")
    assignment = {r["record_id"]: split for u, split in taken.items() for r in units[u]}
    subset = {r["record_id"] for u in subset_taken for r in units[u]}
    reserve = []
    for (source, category), order in sorted(orders.items(), key=lambda kv: (source_rank[kv[0][0]], kv[0][1])):
        rank = 0
        for unit in order:
            if unit in taken:
                continue
            for r in units[unit]:
                if (r["source"], stratum(r)) == (source, category):
                    rank += 1
                    reserve.append({"record_id": r["record_id"], "source": source, "category": category,
                                    "dup_group": r["dup_group"], "split_group": unit, "reserve_rank": rank,
                                    "has_sapo": r["has_sapo"], "hold_reason": hold.get(unit, "")})
    return assignment, subset, reserve, quotas, units, {r["record_id"] for m in selectable.values() for r in m}


def manifest_rows(pool, assignment, subset, split_group_of, sources, seed):
    split_rank = {split: i for i, split in enumerate(SPLITS)}
    source_rank = {source: i for i, source in enumerate(sources)}
    rows = [{"record_id": r["record_id"], "source": r["source"], "split": assignment[r["record_id"]],
             "dup_group": r["dup_group"], "is_human_subset": r["record_id"] in subset, "seed": seed,
             "split_group": split_group_of[r["dup_group"]], "category": r["category"],
             "outlet_group": r["outlet_group"]} for r in pool if r["record_id"] in assignment]
    return sorted(rows, key=lambda r: (source_rank[r["source"]], split_rank[r["split"]], r["category"],
                                       r["record_id"]))


def human_scope(manifest):
    return [r for r in manifest if r["split"] != "train" or r["is_human_subset"]]


def annotation_tables(manifest, titles, seed):
    """(annotation map, blind dev rows, blind subset+test rows) for the human scope.

    The scope is ordered by record_id then shuffled with its own seed; IDs are 'A' + a seeded sample of
    10000..99999, so neither the ID nor the row order follows source, category, split or record_id."""
    scope = sorted(human_scope(manifest), key=lambda r: r["record_id"])
    Random(f"{seed}:annotation_order").shuffle(scope)
    numbers = Random(f"{seed}:annotation_id").sample(range(10000, 100000), len(scope))
    mapping = [{"annotation_id": f"A{n}", "record_id": r["record_id"], "split": r["split"], "source": r["source"]}
               for n, r in zip(numbers, scope)]
    dev = [{"annotation_id": m["annotation_id"], "title": titles[m["record_id"]]} for m in mapping
           if m["split"] == "dev"]
    rest = [{"annotation_id": m["annotation_id"], "title": titles[m["record_id"]]} for m in mapping
            if m["split"] != "dev"]
    return sorted(mapping, key=lambda m: m["annotation_id"]), dev, rest


def verify(records, pool, manifest, reserve, annotation_map, blind, sources, split_group_of, pairs=(),
           separate=frozenset(), held=frozenset()):
    """Integrity checks; raises SplitError on the first failure, returns the list of passed checks."""
    passed = []

    def check(ok, text, failure):
        if not ok:
            raise SplitError(f"integrity check failed: {text}: {failure}")
        passed.append(text)

    counts = Counter((r["source"], r["split"]) for r in manifest)
    subset = Counter(r["source"] for r in manifest if r["is_human_subset"])
    wrong = [f"{s}/{sp}: {counts[(s, sp)]} != {row[TARGETS[sp]]}" for s, row in sources.items() for sp in SPLITS
             if counts[(s, sp)] != int(row[TARGETS[sp]])]
    wrong += [f"{s}/human_subset: {subset[s]} != {row[TARGETS[SUBSET]]}" for s, row in sources.items()
              if subset[s] != int(row[TARGETS[SUBSET]])]
    check(not wrong, "Số bài đúng quota theo nguồn × split và subset human", "; ".join(wrong[:5]))
    ids = [r["record_id"] for r in manifest] + [r["record_id"] for r in reserve]
    check(len(ids) == len(set(ids)) and set(ids) == {r["record_id"] for r in pool},
          "Mỗi bài đủ điều kiện nằm đúng một lần trong manifest hoặc reserve", "record_id duplicated or missing")
    by_unit, by_dup = defaultdict(set), defaultdict(set)
    for r in manifest:
        by_unit[r["split_group"]].add(r["split"])
        by_dup[r["dup_group"]].add(r["split"])
    split_of_unit = {u: next(iter(s)) for u, s in by_unit.items()}
    leaked = [u for u in (r["split_group"] for r in reserve) if u in split_of_unit]
    check(all(len(s) == 1 for s in by_unit.values()) and all(len(s) == 1 for s in by_dup.values()) and not leaked,
          "Mỗi split_group/dup_group chỉ thuộc một split (không tách giữa split và reserve)",
          f"{sum(len(s) > 1 for s in by_unit.values())} split_groups span splits, {len(leaked)} split into reserve")
    outlet = {s: row["outlet_group"] for s, row in sources.items()}
    dup_outlets = defaultdict(set)
    for r in records:
        dup_outlets[split_group_of[r["dup_group"]]].add(outlet[r["source"]])
    bad = [u for u, split in split_of_unit.items() if split != "test_unseen" and "unseen" in dup_outlets[u]]
    check(not bad, "Không bài nào trong train/dev/test_seen chung dup_group/split_group với bài unseen (bản seen "
          "trùng unseen đã bị loại ở bước clean)", f"{len(bad)} seen split_groups touch an unseen record")
    group_of = {r["record_id"]: r["dup_group"] for r in records}
    split_of_dup = {r["dup_group"]: r["split"] for r in manifest}
    crossing = []
    for p in pairs:
        a, b = (split_of_dup.get(group_of[p[k]]) for k in ("record_id_a", "record_id_b"))
        if a and b and a != b and tuple(sorted((p["record_id_a"], p["record_id_b"]))) not in separate:
            crossing.append(p["kind"])
    check(not crossing, "Không cặp gần trùng chưa gộp nào (chuỗi template, `dedup_review.csv`, liên kết bị chặn trong "
          "`dedup_pairs.csv`; trừ cặp đã quyết định `separate`) có hai bài ở hai split khác nhau",
          f"{len(crossing)} pairs cross splits ({dict(Counter(crossing))})")
    check(not any(split_of_dup.get(g) for g in held), "Bài seen có cặp gần trùng chưa review với bài unseen không vào "
          "split nào (giữ trong reserve)", "held dup_group selected")
    check(all(r["split"] == "train" for r in manifest if r["is_human_subset"]), "Subset human nằm trong train",
          "subset record outside train")
    has_sapo = {r["record_id"]: r["has_sapo"] for r in pool}
    check(all(has_sapo[r["record_id"]] for r in manifest if r["split"] == "dev")
          and all(r["has_sapo"] == has_sapo[r["record_id"]] for r in reserve),
          "Mọi bài dev có sapo; cột has_sapo của reserve khớp bảng clean", "dev record without sapo or reserve mismatch")
    check(all(outlet[r["source"]] == ("unseen" if r["split"] == "test_unseen" else "seen") for r in manifest),
          "Nguồn unseen chỉ ở test_unseen, nguồn seen không ở test_unseen", "outlet/split mismatch")
    scope = {r["record_id"] for r in human_scope(manifest)}
    mapped = [m["record_id"] for m in annotation_map]
    annotation_ids = [m["annotation_id"] for m in annotation_map]
    blind_ids = [b["annotation_id"] for rows in blind for b in rows]
    check(len(mapped) == len(set(mapped)) and set(mapped) == scope and len(set(annotation_ids)) == len(mapped)
          and sorted(blind_ids) == sorted(annotation_ids),
          "Bảng blind và annotation_map phủ đúng phạm vi human, annotation_id duy nhất", "mismatch")
    return passed
