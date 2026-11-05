"""Warsaw 25 Stylometric Features adapted for Vietnamese headlines.

Reference:
W. Michaluk et al., 'Click It or Leave It: Detecting and Spoiling Clickbait
with Informativeness Measures and Large Language Models' (2026).
Table 5 (Appendix A.1): 25 informativeness/stylometric measures.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ds_project.features.text_norm import normalize_text

FEATURE_NAMES = [
    "char_count",
    "word_count",
    "mean_word_length",
    "common_words_ratio",
    "capital_letters_ratio",
    "capital_words_count",
    "punctuation_ratio",
    "bait_punctuation_count",
    "non_bait_punctuation_count",
    "numbers_count",
    "pronouns_count",
    "second_person_pronouns_count",
    "superlatives_ratio",
    "speculatives_count",
    "bait_phrases_count",
    "similarity_score",
    "polarity_score",
    "subjectivity_score",
    "ttr",
    "cttr",
    "maas_index",
    "hdd",
    "fres",
    "fkgl",
    "ari",
]

# Vietnamese stopwords (common function words)
VI_STOPWORDS = {
    "và", "của", "là", "có", "được", "cho", "trong", "để", "với", "các",
    "những", "một", "này", "đó", "đã", "sẽ", "đang", "khi", "tại", "về",
    "như", "ra", "vào", "lại", "người", "từ", "ở", "nhiều", "không", "cũng",
    "thì", "mà", "nhưng", "bị", "bởi", "do", "bằng", "lên", "xuống", "sau",
    "trước", "nên", "hay", "hoặc", "theo", "chỉ", "qua", "đến"
}

# Vietnamese personal pronouns (1st, 2nd, 3rd persons, demonstratives)
VI_PRONOUNS = {
    "tôi", "tao", "mình", "chúng tôi", "chúng ta", "chúng tao", "chúng mình",
    "anh", "chị", "em", "ông", "bà", "cô", "chú", "bác", "hắn", "y", "nó",
    "họ", "chúng nó", "người ta", "ai", "ai đó"
}

# Vietnamese 2nd person pronouns (direct reader address, key curiosity cue)
VI_SECOND_PERSON = {
    "bạn", "các bạn", "quý vị", "độc giả", "mày", "ngươi", "anh em", "chúng mày"
}

# Superlative markers in Vietnamese headlines
VI_SUPERLATIVES = {
    "nhất", "hàng đầu", "vô địch", "đỉnh cao", "tuyệt nhất", "số 1", "kỷ lục",
    "bậc nhất", "chưa từng có", "duy nhất", "tột đỉnh", "cực đỉnh", "đắt đỏ nhất",
    "giàu nhất", "đẹp nhất", "khủng nhất", "lớn nhất", "cao nhất"
}

# Speculative / hedge markers
VI_SPECULATIVES = {
    "có thể", "liệu", "phải chăng", "nghi vấn", "dường như", "chăng", "nghi ngờ",
    "chưa rõ", "tin đồn", "đồn đoán", "dấu hỏi", "nguy cơ", "sắp", "chuẩn bị"
}

# Clickbait phrases and lexemes in Vietnamese headlines
VI_BAIT_PHRASES = [
    "bật mí", "tiết lộ", "lộ diện", "ngỡ ngàng", "ngã ngửa", "bàng hoàng",
    "gây sốc", "kinh ngạc", "không ngờ", "cái kết", "sự thật", "danh tính",
    "đây là lý do", "khiến ai cũng", "dân mạng", "nóng mắt", "đứng hình",
    "khó tin", "thực hư", "ít ai biết", "bất ngờ", "té ngửa", "chao đảo",
    "hóa ra", "bóc phốt", "gây bão", "dậy sóng", "thót tim", "sững sờ",
    "xôn xao", "biển người", "quy tắc sống còn", "canh bạc", "sốc", "sự thật bất ngờ"
]

# Sentiment lexicons
VI_POSITIVE_WORDS = {
    "thành công", "xuất sắc", "chiến thắng", "tuyệt vời", "khen ngợi", "vinh danh",
    "hạnh phúc", "yêu thích", "tự hào", "khởi sắc", "tăng mạnh", "thắng lợi",
    "đột phá", "hồi sinh", "phát tài", "may mắn", "khen ngợi", "tỏa sáng", "thán phục"
}

VI_NEGATIVE_WORDS = {
    "thất bại", "tai nạn", "tử vong", "chết", "kinh hoàng", "bắt giữ", "khởi tố",
    "tranh cãi", "chỉ trích", "đau lòng", "thảm kịch", "lao dốc", "sụp đổ", "nạn nhân",
    "lừa đảo", "suy thoái", "vỡ nợ", "nguy kịch", "tang thương", "hỗn loạn"
}

# Subjectivity / emotional intensification lexicons
VI_SUBJECTIVE_WORDS = {
    "kinh ngạc", "đáng sợ", "xúc động", "nghẹn ngào", "bàng hoàng", "đau xót",
    "vui mừng", "phẫn nộ", "bất bình", "choáng váng", "kinh hoàng", "hụt hẫng",
    "ngỡ ngàng", "thót tim", "sững sờ", "nghẹn lòng", "cay đắng", "tột cùng"
}

BAIT_PUNCT_CHARS = set('!"“”?()#…')
NON_BAIT_PUNCT_CHARS = set("$%&,.;:-/")
ALL_PUNCT_CHARS = set(string.punctuation).union(set("“”‘’…—–•"))


def _compute_cosine_char_3gram(s1: str, s2: str) -> float:
    """Compute character 3-gram cosine similarity between two strings."""
    if not s1 or not s2:
        return 0.0
    s1_norm = s1.lower().strip()
    s2_norm = s2.lower().strip()
    if len(s1_norm) < 3 or len(s2_norm) < 3:
        # Fallback to token overlap
        t1, t2 = set(s1_norm.split()), set(s2_norm.split())
        if not t1 or not t2:
            return 0.0
        return len(t1 & t2) / math.sqrt(len(t1) * len(t2))

    c1 = Counter(s1_norm[i:i + 3] for i in range(len(s1_norm) - 2))
    c2 = Counter(s2_norm[i:i + 3] for i in range(len(s2_norm) - 2))

    common = set(c1.keys()) & set(c2.keys())
    if not common:
        return 0.0

    dot = sum(c1[g] * c2[g] for g in common)
    norm1 = math.sqrt(sum(v * v for v in c1.values()))
    norm2 = math.sqrt(sum(v * v for v in c2.values()))
    if norm1 <= 0.0 or norm2 <= 0.0:
        return 0.0
    return float(dot / (norm1 * norm2))


def _compute_hdd(words: list[str], sample_size: int = 10) -> float:
    """Hypergeometric Diversity (HD-D) index of lexical diversity."""
    n = len(words)
    if n <= 1:
        return 0.0
    types_counter = Counter(words)
    v = len(types_counter)
    if v <= 1:
        return 0.0
    sample_size = min(sample_size, n)
    if sample_size <= 1:
        return float(v / n)

    prob_sum = 0.0
    # For each word type, compute probability that it is drawn in a hypergeometric sample
    for count in types_counter.values():
        if n - count >= sample_size:
            # comb(n - count, sample_size) / comb(n, sample_size)
            non_appearance = math.comb(n - count, sample_size) / math.comb(n, sample_size)
        else:
            non_appearance = 0.0
        prob_sum += (1.0 - non_appearance)

    return float(prob_sum / sample_size)


def extract_headline_features(title: str, sapo: str | None = None) -> dict[str, float]:
    """Extract the 25 Warsaw stylometric features for a single headline.

    Parameters
    ----------
    title : str
        Cleaned headline text.
    sapo : str or None, default None
        Article lead paragraph (sapo) for title-body similarity.

    Returns
    -------
    dict[str, float]
        Dictionary mapping feature name to numeric value.
    """
    clean_title = normalize_text(title)
    lower_title = clean_title.lower()

    # 1. Character count
    char_count = float(len(clean_title))

    # Tokenization by whitespace
    raw_tokens = clean_title.split()
    word_count = float(len(raw_tokens))
    words_clean = [t.strip(string.punctuation).lower() for t in raw_tokens if t.strip(string.punctuation)]
    n_tokens = len(words_clean) if words_clean else len(raw_tokens)

    # 2. Word count
    effective_word_count = float(max(n_tokens, 1))

    # 3. Mean word length
    if raw_tokens:
        mean_word_len = float(sum(len(w) for w in raw_tokens) / len(raw_tokens))
    else:
        mean_word_len = 0.0

    # 4. Common words ratio (Vietnamese stopwords)
    stopword_count = sum(1 for w in words_clean if w in VI_STOPWORDS)
    common_words_ratio = float(stopword_count / effective_word_count)

    # 5. Capital letters ratio
    upper_chars = sum(1 for c in clean_title if c.isupper())
    capitals_ratio = float(upper_chars / max(char_count, 1.0))

    # 6. Capital words count (all-uppercase tokens longer than 1 character)
    capital_words_count = float(sum(1 for w in raw_tokens if len(w) > 1 and w.isupper()))

    # 7. Punctuation ratio
    punct_count = sum(1 for c in clean_title if c in ALL_PUNCT_CHARS)
    punct_ratio = float(punct_count / max(char_count, 1.0))

    # 8. Bait punctuation count (!, ", “, ”, ?, (, ), #, …)
    bait_punct_count = float(sum(1 for c in clean_title if c in BAIT_PUNCT_CHARS) + clean_title.count("..."))

    # 9. Non-bait punctuation count ($, %, &, ,, ., ;, :, -, /)
    non_bait_punct_count = float(sum(1 for c in clean_title if c in NON_BAIT_PUNCT_CHARS))

    # 10. Numbers count (distinct numeric tokens)
    numeric_tokens = re.findall(r"\b\d+(?:[.,]\d+)?\b", clean_title)
    numbers_count = float(len(numeric_tokens))

    # 11. Pronouns count
    pronouns_count = 0.0
    for p in VI_PRONOUNS:
        if " " in p:
            pronouns_count += lower_title.count(p)
        else:
            pronouns_count += sum(1 for w in words_clean if w == p)

    # 12. 2nd person pronouns count
    second_person_count = 0.0
    for p in VI_SECOND_PERSON:
        if " " in p:
            second_person_count += lower_title.count(p)
        else:
            second_person_count += sum(1 for w in words_clean if w == p)

    # 13. Superlatives ratio
    superlative_count = 0.0
    for s in VI_SUPERLATIVES:
        if " " in s:
            superlative_count += lower_title.count(s)
        else:
            superlative_count += sum(1 for w in words_clean if w == s)
    superlatives_ratio = float(superlative_count / effective_word_count)

    # 14. Speculatives count
    speculatives_count = 0.0
    for sp in VI_SPECULATIVES:
        if " " in sp:
            speculatives_count += lower_title.count(sp)
        else:
            speculatives_count += sum(1 for w in words_clean if w == sp)

    # 15. Bait phrases count
    bait_phrases_count = float(sum(1 for phrase in VI_BAIT_PHRASES if phrase in lower_title))

    # 16. Similarity score (title - sapo similarity)
    similarity_score = 0.0
    if sapo and isinstance(sapo, str) and sapo.strip():
        similarity_score = _compute_cosine_char_3gram(clean_title, sapo)

    # 17. Polarity score (-1.0 to 1.0)
    pos_count = sum(1 for w in words_clean if w in VI_POSITIVE_WORDS)
    neg_count = sum(1 for w in words_clean if w in VI_NEGATIVE_WORDS)
    total_sentiment = pos_count + neg_count
    if total_sentiment > 0:
        polarity_score = float((pos_count - neg_count) / total_sentiment)
    else:
        polarity_score = 0.0

    # 18. Subjectivity score
    subj_count = sum(1 for w in words_clean if w in VI_SUBJECTIVE_WORDS)
    subjectivity_score = float(subj_count / effective_word_count)

    # 19. TTR (Type-Token Ratio)
    tokens_for_div = words_clean if words_clean else [clean_title]
    n_tokens_div = len(tokens_for_div)
    v_types = len(set(tokens_for_div))
    ttr = float(v_types / max(n_tokens_div, 1))

    # 20. CTTR (Corrected Type-Token Ratio): V / sqrt(2N)
    cttr = float(v_types / math.sqrt(2 * max(n_tokens_div, 1)))

    # 21. Maas Index: (ln N - ln V) / (ln N)^2
    if n_tokens_div > 1 and v_types > 0 and v_types < n_tokens_div:
        ln_n = math.log(n_tokens_div)
        ln_v = math.log(v_types)
        maas_index = float((ln_n - ln_v) / (ln_n ** 2))
    else:
        maas_index = 0.0

    # 22. HD-D (Hypergeometric Diversity)
    hdd = _compute_hdd(tokens_for_div)

    # Readability sentence count (detecting sentence separators, minimum 1)
    sentences = re.split(r"[.!?;\n]+", clean_title)
    n_sentences = float(max(len([s for s in sentences if s.strip()]), 1))

    # 23. FRES (Flesch Reading-Ease Score)
    # FRES = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    # In Vietnamese phonology, 1 word token is 1 syllable (syllables / words == 1.0)
    words_per_sent = float(effective_word_count / n_sentences)
    syllables_per_word = 1.0
    fres = float(206.835 - 1.015 * words_per_sent - 84.6 * syllables_per_word)

    # 24. FKGL (Flesch-Kincaid Grade Level)
    # FKGL = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
    fkgl = float(0.39 * words_per_sent + 11.8 * syllables_per_word - 15.59)

    # 25. ARI (Automated Readability Index)
    # ARI = 4.71 * (characters / words) + 0.5 * (words / sentences) - 21.43
    chars_per_word = float(char_count / effective_word_count)
    ari = float(4.71 * chars_per_word + 0.5 * words_per_sent - 21.43)

    return {
        "char_count": char_count,
        "word_count": word_count,
        "mean_word_length": mean_word_len,
        "common_words_ratio": common_words_ratio,
        "capital_letters_ratio": capitals_ratio,
        "capital_words_count": capital_words_count,
        "punctuation_ratio": punct_ratio,
        "bait_punctuation_count": bait_punct_count,
        "non_bait_punctuation_count": non_bait_punct_count,
        "numbers_count": numbers_count,
        "pronouns_count": pronouns_count,
        "second_person_pronouns_count": second_person_count,
        "superlatives_ratio": superlatives_ratio,
        "speculatives_count": speculatives_count,
        "bait_phrases_count": bait_phrases_count,
        "similarity_score": similarity_score,
        "polarity_score": polarity_score,
        "subjectivity_score": subjectivity_score,
        "ttr": ttr,
        "cttr": cttr,
        "maas_index": maas_index,
        "hdd": hdd,
        "fres": fres,
        "fkgl": fkgl,
        "ari": ari,
    }


class WarsawStylometricExtractor(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for the 25 Warsaw stylometric features."""

    def __init__(self) -> None:
        self.feature_names_ = list(FEATURE_NAMES)

    def fit(self, X: pd.DataFrame | Sequence[str], y: None = None) -> WarsawStylometricExtractor:
        return self

    def transform(self, X: pd.DataFrame | Sequence[str]) -> np.ndarray:
        """Extract features into a 2D numpy array of shape (N, 25)."""
        rows: list[dict[str, float]] = []

        if isinstance(X, pd.DataFrame):
            titles = X["title"].tolist() if "title" in X.columns else X.iloc[:, 0].tolist()
            sapos = X["sapo"].tolist() if "sapo" in X.columns else [None] * len(titles)
            for t, s in zip(titles, sapos):
                rows.append(extract_headline_features(t, s))
        else:
            for item in X:
                if isinstance(item, tuple) and len(item) == 2:
                    rows.append(extract_headline_features(item[0], item[1]))
                else:
                    rows.append(extract_headline_features(str(item), None))

        df_out = pd.DataFrame(rows, columns=self.feature_names_)
        return df_out.to_numpy(dtype=np.float64)

    def transform_df(self, X: pd.DataFrame | Sequence[str]) -> pd.DataFrame:
        """Extract features as a pandas DataFrame with named columns."""
        arr = self.transform(X)
        return pd.DataFrame(arr, columns=self.feature_names_)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> list[str]:
        return list(self.feature_names_)
