"""语义与线索特征：条款/场景相似度、道义极性词典、角色线索正则。

后端可插拔：
  * "tfidf"  —— 默认，sklearn TfidfVectorizer（word 1-2gram，sublinear tf），零下载、CPU 秒级
  * "st"     —— sentence-transformers（若已安装；MVP 环境未装 torch，自动回退）
A 同学的所有实验默认走 tfidf 后端，报告里已注明该替换对结论的影响方向。
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Iterable, Sequence

# ---------------------------------------------------------------- 角色线索
# 顺序即优先级：例外 > 条件 > 定义 > 引用 > 救济 > 义务
ROLE_CUES: list[tuple[str, re.Pattern]] = [
    ("EXC", re.compile(
        r"\b(except\b|excepting\b|unless\b|subject to\b|notwithstanding\b|"
        r"other than\b|save (?:as|for)\b|provided,? however\b|"
        r"to the extent (?:that )?(?:permitted|allowed))", re.I)),
    ("COND", re.compile(
        r"\b(if\b|provided that\b|in the event (?:that|of)\b|on condition that\b|"
        r"conditional upon\b|upon the occurrence\b|only if\b)", re.I)),
    ("DEF", re.compile(
        r"\b(means\b|shall mean\b|refers to\b|is defined as\b|as defined (?:in|below)\b|"
        r"hereinafter\b|\" means)", re.I)),
    ("SCOPE", re.compile(
        r"\b(as set (?:out|forth) in\b|pursuant to\b|in accordance with\b|"
        r"as (?:defined|specified|described) in\b|subject to (?:section|clause|article)\b|"
        r"under (?:section|clause|article)\b)", re.I)),
    ("REMEDY", re.compile(
        r"\b(indemnif\w*|liable\b|liability\b|damages\b|terminat\w*|"
        r"reimburse\w*|penalt\w*|specific performance)\b", re.I)),
]

# 交叉引用锚点（文档 1.4 节要求的高置信度标记）
XREF_CUES = re.compile(
    r"\b(as set (?:out|forth) in|pursuant to|in accordance with|subject to|"
    r"notwithstanding|as defined in|as specified in|under section|"
    r"in the event of|prior to|following the)\b", re.I)

TEMPORAL_CUES = re.compile(
    r"\b(upon|after|before|prior to|following|thereafter|hereafter|"
    r"from time to time|during the term|expiration|termination of)\b", re.I)

# 道义极性（条款侧）
PERMISSIVE_CUES = re.compile(
    r"\b(may\b|is entitled to|are entitled to|grants?\b|granted\b|right to|"
    r"permitted to|authoriz\w+|licensed to|at (?:its|their) (?:sole )?discretion|"
    r"unlimited|non-exclusive|without (?:the )?(?:prior )?(?:written )?consent)", re.I)

PROHIBITIVE_CUES = re.compile(
    r"\b(shall not|may not|must not|is not permitted|are not permitted|"
    r"prohibited|forbidden|shall refrain|under no circumstances|"
    r"no\b[^.]{0,60}\bshall\b)", re.I)

# 场景侧的行为线索
SCENARIO_BREACH_CUES = re.compile(
    r"\b(without (?:obtaining|the )?(?:prior |explicit |written )?(?:consent|approval|notice)|"
    r"fail(?:ed|s|ure)? to|did not|didn't|unable to|refus\w+|declin\w+|"
    r"breach\w*|violat\w+|exceed\w*|beyond the scope|unauthoriz\w+|"
    r"ignor\w+|omit\w*|neglect\w*|disclos\w+ (?:confidential|without))", re.I)

SCENARIO_COMPLIANT_CUES = re.compile(
    r"\b(in accordance with|pursuant to|as permitted|as required by|"
    r"consistent with|in compliance with|duly|properly|"
    r"obtained (?:the )?(?:prior )?(?:written )?consent|gave notice|"
    r"within the (?:scope|terms)|as (?:set out|provided))", re.I)

SCENARIO_NA_CUES = re.compile(
    r"\b(unrelated|does not apply|not applicable|no (?:such )?(?:obligation|provision)|"
    r"silent on|outside the scope|different agreement|not governed)\b", re.I)

NEGATION = re.compile(r"\b(not|no|never|neither|nor|without|none|cannot)\b", re.I)

_SENT_SPLIT = re.compile(r"(?<=[.;])\s+(?=[A-Z(])")
# 法律文本里真正的"语义转折点"：例外/条件/引用引导词
_SUBCLAUSE_SPLIT = re.compile(
    r"(?:;|,)?\s+(?=(?:provided(?:,| that)|except(?: that| as| for)?|unless|"
    r"subject to|notwithstanding|in the event that|for the avoidance of doubt)\b)",
    re.I)
_AND_SPLIT = re.compile(r"(?:;)\s+(?=(?:and|or)\b)")


def split_sentences(text: str, max_parts: int = 5, long_words: int = 45) -> list[str]:
    """两层切分：先按句号/分号切句，再把长句按例外/条件引导词切成 facet。

    为什么需要第二层：ACE 的条款中位数 108 词、平均 167 词，单句往往同时包含
    "主义务 + provided that 例外 + subject to 条件"。不做第二层切分，一个条款
    就只有 1 个节点，多跳搜索无从展开（实测 74% 的条款只会切出 1 段）。
    """
    text = " ".join(str(text).split())
    if not text:
        return [""]

    sents = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    parts: list[str] = []
    for s in sents:
        if len(s.split()) <= long_words:
            parts.append(s)
            continue
        frags = [f.strip() for f in _SUBCLAUSE_SPLIT.split(s) if f.strip()]
        # 片段过短（<8 词）时并回上一段，避免制造噪声节点
        merged: list[str] = []
        for f in frags:
            if merged and len(f.split()) < 8:
                merged[-1] = merged[-1] + " " + f
            else:
                merged.append(f)
        parts.extend(merged)

    if len(parts) <= max_parts:
        return parts
    head = parts[: max_parts - 1]
    head.append(" ".join(parts[max_parts - 1:]))
    return head


def detect_role(sentence: str, is_first: bool) -> str:
    for role, pat in ROLE_CUES:
        if pat.search(sentence):
            return role
    return "OBL" if is_first else "OBL"


def polarity_signals(text: str) -> dict[str, float]:
    """道义极性信号：permissive / prohibitive / 负向词密度，均做长度归一。"""
    words = max(1, len(text.split()))
    per = len(PERMISSIVE_CUES.findall(text)) / math.sqrt(words)
    pro = len(PROHIBITIVE_CUES.findall(text)) / math.sqrt(words)
    neg = len(NEGATION.findall(text)) / math.sqrt(words)
    return {"permissive": per, "prohibitive": pro, "negation": neg}


def scenario_signals(text: str) -> dict[str, float]:
    words = max(1, len(text.split()))
    n = math.sqrt(words)
    return {
        "breach": len(SCENARIO_BREACH_CUES.findall(text)) / n,
        "compliant": len(SCENARIO_COMPLIANT_CUES.findall(text)) / n,
        "na": len(SCENARIO_NA_CUES.findall(text)) / n,
        "len_log": math.log1p(words) / 10.0,
    }


# ---------------------------------------------------------------- 语义后端
class TfidfSemanticScorer:
    """基于 TF-IDF 余弦的相似度。fit 用训练集的文本，避免 test 泄漏。"""

    name = "tfidf"

    def __init__(self, ngram_max: int = 2, min_df: int = 2, max_features: int = 200_000):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vec = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, ngram_max),
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self._fitted = False

    def fit(self, texts: Iterable[str]) -> "TfidfSemanticScorer":
        docs = [self._clean(t) for t in texts if t and t.strip()]
        if len(docs) < 10:
            docs = docs + ["placeholder document text"] * 10
        self._vec.fit(docs)
        self._fitted = True
        return self

    @staticmethod
    def _clean(t: str) -> str:
        return " ".join(str(t).split()).lower()

    def transform(self, texts: Sequence[str]):
        if not self._fitted:
            raise RuntimeError("TfidfSemanticScorer 未 fit")
        return self._vec.transform([self._clean(t) for t in texts])

    def cosine(self, a: str, b: str) -> float:
        import numpy as np

        v = self.transform([a, b])
        x, y = v[0], v[1]
        denom = float((x.multiply(x)).sum()) ** 0.5 * float((y.multiply(y)).sum()) ** 0.5
        if denom <= 0:
            return 0.0
        return float(np.clip((x.multiply(y)).sum() / denom, 0.0, 1.0))

    def pairwise(self, a_vecs, b_vecs):
        """批量化余弦：a_vecs (n,d) 与 b_vecs (m,d) 的内积直接可用（已 L2 归一）。"""
        return (a_vecs @ b_vecs.T).toarray()

    def fit_terms(self, texts: Sequence[str]):
        """返回用于词面重叠的关键词集合（去掉停用词、长度>3）。"""
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        out = []
        for t in texts:
            toks = {w for w in re.findall(r"[a-z]{4,}", str(t).lower())
                    if w not in ENGLISH_STOP_WORDS}
            out.append(toks)
        return out


class SentenceTransformerScorer:
    """可选后端。未安装 sentence-transformers 时由 build_scorer 自动跳过。"""
    name = "st"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # noqa: F401

        self._m = SentenceTransformer(model_name)

    def fit(self, texts):
        return self

    def cosine(self, a: str, b: str) -> float:
        import numpy as np

        v = self._m.encode([a, b], normalize_embeddings=True)
        return float(np.clip(v[0] @ v[1], 0.0, 1.0))


def build_scorer(backend: str = "tfidf", train_texts: Sequence[str] | None = None):
    backend = (backend or "tfidf").lower()
    if backend == "st":
        try:
            return SentenceTransformerScorer()
        except Exception as exc:  # pragma: no cover - 取决于本地环境
            print(f"[features] sentence-transformers 不可用（{type(exc).__name__}），回退到 tfidf 后端")
            backend = "tfidf"
    sc = TfidfSemanticScorer()
    if train_texts:
        sc.fit(train_texts)
    return sc


# 同意/通知类义务：合规审查里最高频的"程序性违反"来源
CLAUSE_CONSENT = re.compile(
    r"\b(consent|approval|authoriz\w+|prior written notice|shall notify|notification)\b", re.I)
SCENARIO_NO_CONSENT = re.compile(
    r"\b(without (?:obtaining |first obtaining )?(?:the )?(?:prior |written |explicit )?"
    r"(?:consent|approval|authorization|authorisation|notice)|"
    r"fail(?:ed|s|ure)? to (?:notify|obtain|seek)|no (?:prior )?(?:written )?notice|"
    r"unnotified|never (?:notified|obtained))\b", re.I)
