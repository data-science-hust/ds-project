"""Category normalization, mapping lookup and rule-based proposals (raw -> common group).

Mapping rows are proposals until the team reviews them; nothing here defaults an
unknown key to a catch-all group.
"""

from __future__ import annotations

import csv
import html
import re
import unicodedata
from pathlib import Path

MAPPING_FIELDS = ("source", "category_raw", "category", "review_status", "notes")
REVIEW_STATUSES = {"proposed", "needs_review", "approved"}

TS, TG, KT, BDS, PL = "Thời sự - Xã hội", "Thế giới", "Kinh tế - Tài chính", "Bất động sản", "Pháp luật"
GT, TT, GD, SK, DS = "Giải trí - Văn hóa", "Thể thao", "Giáo dục", "Sức khỏe", "Đời sống - Gia đình"
LD, CN, XE, DL, KH = "Làm đẹp - Thời trang", "Công nghệ - Khoa học", "Xe", "Du lịch - Ẩm thực", "Khác"
TV = "Tử vi - Phong thủy"  # group decision 2026-10-23: astrology sections kept apart from Đời sống
GROUPS = (TS, TG, KT, BDS, PL, GT, TT, GD, SK, DS, LD, CN, XE, DL, TV, KH)


def normalize_category(value):
    """Technical fix only: HTML entities, NFC, outer whitespace; empty -> None."""
    if not isinstance(value, str):
        return None
    value = unicodedata.normalize("NFC", html.unescape(value)).strip()
    return value or None


def load_mapping(path):
    """Return {(source, normalized category_raw): row}; reject duplicate keys and unknown groups."""
    mapping = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != MAPPING_FIELDS:
            raise ValueError(f"{path}: columns must be exactly {','.join(MAPPING_FIELDS)}")
        for number, row in enumerate(reader, 2):
            key = (row["source"].strip(), normalize_category(row["category_raw"]))
            if key[1] is None:
                raise ValueError(f"{path}:{number}: empty category_raw is not a mapping key")
            if key in mapping:
                raise ValueError(f"{path}:{number}: duplicate key {key}")
            category = (row["category"] or "").strip()
            if category and category not in GROUPS:
                raise ValueError(f"{path}:{number}: unknown common group {category!r}")
            if row["review_status"] not in REVIEW_STATUSES:
                raise ValueError(f"{path}:{number}: review_status must be one of {sorted(REVIEW_STATUSES)}")
            if row["review_status"] == "approved" and not category:
                raise ValueError(f"{path}:{number}: an approved row needs a category")
            mapping[key] = {**row, "category_raw": key[1], "category": category or None}
    return mapping


def lookup(mapping, source, category_raw):
    """(category, review_status, reason). Missing key or empty category -> (None, None, why), never a default."""
    if category_raw is None:
        return None, None, "missing category_raw"
    row = mapping.get((source, category_raw))
    if row is None:
        return None, None, "key not in category_mapping.csv"
    if row["category"] is None:
        return None, None, f"mapping row has no category yet ({row['review_status']})"
    return row["category"], row["review_status"], ""


# --- dantri: "Parent, Child[, Grandchild]" -> map by parent, with child overrides.
DT_PARENT = {"Thời sự": TS, "Pháp luật": PL, "Thế giới": TG, "Thể thao": TT, "Kinh doanh": KT, "Sức khỏe": SK,
             "Giáo dục": GD, "Giải trí": GT, "Bất động sản": BDS, "Công nghệ": CN, "Khoa học": CN,
             "Lao động - Việc làm": TS, "Đời sống": DS, "Nội vụ": TS, "Du lịch": DL, "Ô tô - Xe máy": XE,
             "Tình yêu - Giới tính": DS, "Tâm điểm": TS, "Tấm lòng nhân ái": TS, "Bạn đọc": TS, "Thời tiết": TS, "Tết": DS}
DT_NOTE = {"Lao động - Việc làm": "labor/jobs; alt Kinh tế - Tài chính", "Nội vụ": "civil service/BHXH/salary policy",
           "Tâm điểm": "opinion column", "Tấm lòng nhân ái": "charity appeals; alt Đời sống - Gia đình", "Tết": "Tết specials, mixed"}
DT_CHILD = {"Giải trí, Thời trang": (LD, "dantri Giải trí > Thời trang; alt Giải trí - Văn hóa"),
            "Bạn đọc, Tư vấn pháp luật": (PL, ""), "Lao động - Việc làm, Làm giàu": (KT, ""),
            "Kinh doanh, AMACCAO - Vươn mình cùng kỷ nguyên mới": (KT, "sponsored series"),
            "Tết, Du xuân": (DL, "alt Đời sống - Gia đình"), "Tết, Bếp ngon": (DL, ""),
            "Bất động sản, Nội thất": (BDS, "interior; alt Đời sống - Gia đình"),
            "Thế giới, Kiều bào": (TG, "overseas Vietnamese; alt Thời sự - Xã hội"), "Du lịch, Món ngon - Điểm đẹp": (DL, "")}


def propose_dantri(category):
    if category in DT_CHILD:
        return DT_CHILD[category]
    parent = category.split(", ")[0]
    if parent in DT_PARENT:
        return DT_PARENT[parent], DT_NOTE.get(parent, "")
    return None, "no dantri parent rule; map manually"


# --- nhandan: multi-valued tags joined by ',' without space; some tag names contain ', '.
ND_REGION = {"Đồng bằng sông Cửu Long", "Đồng bằng sông Hồng", "Bắc Trung Bộ", "Duyên hải Nam Trung Bộ và Tây Nguyên",
             "Đông Nam Bộ", "Trung du và miền núi phía Bắc", "Hà Nội", "TP Hồ Chí Minh"}
ND_IGNORE = ND_REGION | {
    "Tin chung", "Tin tức", "Video", "Short Video", "Ảnh", "Infographic", "E-Magazine", "Tin vắn", "Tin đọc nhanh",
    "Tin mới nhận", "Báo in", "Nhân Dân hằng ngày", "Thời Nay", "Thời nay", "NHÂN DÂN CUỐI TUẦN", "NHÂN DÂN HẰNG THÁNG",
    "Chuyên đề", "Theo dòng sự kiện", "Tin tức tuần qua", "Ống kính phóng viên", "Điểm sáng", "Tiếng nói từ cơ sở",
    "Sức sống Việt", "Đẹp...+", "Góc quan sát", "Nhân vật & Sự kiện", "Thời sự", "Nhịp sống thời đại"}
ND_TOK = {}
for _group, _tokens in (
    (TS, ["Chính trị", "Xây dựng Đảng", "Quốc hội Việt Nam", "Đại biểu với cử tri", "Xã luận", "Truyền thống vẻ vang",
          "Công an Nhân dân Việt Nam", "50 năm Thành phố mang tên Bác", "Báo chí cách mạng Việt Nam", "Luật Thủ đô năm 2026",
          "Chính trị, Đại hội Đảng toàn quốc lần thứ XIV", "Xã hội", "Media Xã hội", "Người tốt việc tốt", "BHXH và cuộc sống",
          "Đô thị", "Chuyện thường ngày", "Phóng sự - Điều tra", "Xây dựng nông thôn mới", "Môi trường", "Media Môi trường",
          "Thời tiết", "Chuyện phường, chuyện xã", "Nhịp sống", "Media Chính trị"]),
    (KT, ["Kinh tế", "Media Kinh tế", "Tài chính – Chứng khoán", "Thông tin hàng hóa", "Quy hoạch - Đầu tư"]),
    (TG, ["Thế giới", "Châu Âu", "Châu Á-TBD", "Châu Mỹ", "Châu Phi", "Trung Đông", "ASEAN", "Bình luận quốc tế",
          "Quốc tế", "Media Thế giới"]),
    (PL, ["Pháp luật", "Phòng, chống tội phạm công nghệ cao", "Nhận diện thủ đoạn"]),
    (TT, ["Thể thao", "World Cup 2026"]), (SK, ["Y tế"]), (GD, ["Giáo dục"]),
    (CN, ["Khoa học - Công nghệ", "Chuyển đổi số"]),
    (GT, ["Văn hóa", "Văn hóa - Văn nghệ", "Media Văn hóa", "Sáng tác", "Đời sống văn hóa", "Trò chuyện cuối tuần"]),
    (DL, ["Du lịch", "Media Du lịch"]), (KH, ["Bạn cần biết", "Thông tin doanh nghiệp"]),
):
    ND_TOK.update(dict.fromkeys(_tokens, _group))
ND_PRINT_INDEX = {"Thời nay,Báo in", "Báo in,Nhân Dân hằng ngày"}


def nhandan_tokens(category):
    return [token.strip() for token in re.split(r",(?!\s)", category) if token.strip()]


def propose_nhandan(category):
    tokens = nhandan_tokens(category)
    unknown = [t for t in tokens if t not in ND_TOK and t not in ND_IGNORE]
    topical = [ND_TOK[t] for t in tokens if t in ND_TOK]
    # Foreign-region tags win (diplomacy stories tagged Chính trị/Kinh tế + region); else first topical tag.
    group = TG if TG in topical else (topical[0] if topical else None)
    notes = []
    if unknown:
        notes.append("unknown tokens: " + "|".join(unknown))
    if len(set(topical)) > 1:
        notes.append("multi-topic tags: " + "|".join(dict.fromkeys(topical)))
    if group is None:
        if any(t in ND_REGION for t in tokens):
            group = TS
            notes.append("region-only tags (local news, mixed topics) -> default Thời sự - Xã hội")
        else:
            group = KH
            notes.append("format-only tag (Video/Ảnh/Infographic/Báo in)")
    if category in ND_PRINT_INDEX:
        notes.append('print-edition index page (e.g. "BÁO THỜI NAY SỐ 1699"), not an article -> reject in checking')
    return group, "; ".join(notes)


RULES = {"dantri": propose_dantri, "nhandan": propose_nhandan}


def propose(source, category_raw):
    """Return a mapping row proposal for a normalized key; sources without a rule get an empty category."""
    rule = RULES.get(source)
    if rule is None:
        group, note = None, "no rule for this source; fill category manually"
    else:
        group, note = rule(category_raw)
    status = "proposed" if group and not note else "needs_review"
    return {"source": source, "category_raw": category_raw, "category": group or "",
            "review_status": status, "notes": note}
