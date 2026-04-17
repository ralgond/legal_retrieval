"""
Citation Scoring Pipeline for German Swiss Legal Texts (ZGB/OR/etc.)
=====================================================================
Pipeline:
  1. 对每个 cc 切句，抽取 citation
  2. 基于关键词规则 + 位置对每个 (cc, citation) 打分
  3. 结合 rerank score，对每个 (query, citation) 汇总打分
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional


# ─────────────────────────────────────────────
# 1. 常量：关键词权重表
# ─────────────────────────────────────────────

# 正向关键词 → 权重（越大越重要）
POSITIVE_KEYWORDS: dict[str, float] = {
    # 瑞士特有高权威
    "BGE": 2.0,
    "BGer": 1.8,
    "Botschaft": 1.5,
    "Erwägung": 1.2,
    # 直接引用动词
    "gestützt auf": 1.5,
    "gemäss": 1.2,
    "gemäß": 1.2,
    "nach": 0.6,
    "i.V.m.": 1.3,
    "i.S.v.": 1.2,
    "im Sinne von": 1.2,
    "in Verbindung mit": 1.3,
    # 适用性表达
    "findet Anwendung": 1.5,
    "ist anzuwenden": 1.5,
    "ist massgebend": 1.5,
    "ist massgeblich": 1.4,
    "gilt": 1.0,
    "gelten": 1.0,
    "bestimmt": 0.8,
    "regelt": 0.8,
    "sieht vor": 0.8,
    "vorgesehen": 0.7,
    "verstösst gegen": 1.2,
    "verletzt": 1.0,
    # 强调词
    "namentlich": 1.5,
    "insbesondere": 1.0,
    "ausdrücklich": 1.2,
    "explizit": 1.2,
    "zwingend": 1.5,
    "unmittelbar": 1.3,
    "analog": 1.2,
    "entsprechend": 0.8,
    # 论证结构词
    "daher": 0.7,
    "deshalb": 0.7,
    "folglich": 0.8,
    "somit": 0.8,
    "demnach": 0.8,
    "mithin": 0.9,
    "ergibt sich": 0.8,
    "daraus folgt": 0.9,
    # 直接引用标志
    "vgl.": 0.8,
    "vgl. auch": 0.9,
    "siehe": 0.7,
    "s.": 0.5,
}

# 负向关键词 → 权重（越大扣分越多）
NEGATIVE_KEYWORDS: dict[str, float] = {
    # 异见/争议
    "a.M.": 1.5,
    "anderer Meinung": 1.5,
    "a.A.": 1.5,
    "anderer Ansicht": 1.5,
    "str.": 1.2,
    "strittig": 1.2,
    "umstritten": 1.2,
    "fraglich": 1.2,
    "zweifelhaft": 1.0,
    "offen gelassen": 1.5,
    "dahingestellt": 1.3,
    # 背景性/泛列举
    "u.a.": 0.8,
    "unter anderem": 0.8,
    "z.B.": 0.8,
    "zum Beispiel": 0.8,
    "etwa": 0.5,
    "allgemein": 0.6,
    "allgemeine Meinung": 0.9,
    "herrschende Lehre": 0.9,
    "h.L.": 0.9,
    "herrschende Meinung": 0.9,
    "h.M.": 0.9,
    # 否定语境（citation处于被否定结构中）
    "entgegen": 1.0,
    "abweichend von": 1.0,
    "ungeachtet": 0.8,
    "trotz": 0.6,
    "soweit nicht": 0.8,
}

# citation 类型权重（出现在 citation 字符串本身中）
CITATION_TYPE_BONUS: dict[str, float] = {
    "BGE": 1.5,
    "BGer": 1.3,
    "Art.": 0.5,
    "SR": 0.8,
    "AS": 0.6,
    "BV": 0.7,
    "KV": 0.5,
    "Botschaft": 0.8,
}


# ─────────────────────────────────────────────
# 2. 数据结构
# ─────────────────────────────────────────────

@dataclass
class CitationScore:
    citation: str
    cc_id: str
    sentence_idx: int
    sentence_text: str

    # 关键词得分
    kw_pos_score: float = 0.0
    kw_neg_score: float = 0.0
    kw_pos_count: int = 0
    kw_neg_count: int = 0

    # 位置得分
    position_score: float = 0.0

    # citation 类型加成
    type_bonus: float = 0.0

    # 上下文窗口得分
    window_score: float = 0.0

    # cc 级别得分（综合以上）
    cc_level_score: float = 0.0

    # 最终 query 级别得分（聚合多个 cc）
    query_level_score: float = 0.0


@dataclass
class CCResult:
    """一个召回并 rerank 后的 court consideration"""
    cc_id: str
    text: str
    rerank_score: float  # reranker 给出的相关性分数（越高越相关）
    rank: int            # 在 rerank list 中的排名（从 1 开始）


# ─────────────────────────────────────────────
# 3. 工具函数
# ─────────────────────────────────────────────

# 德语法律文本句子分割（简单版：按句号/分号分割，保留结构）
_SENT_SPLIT_RE = re.compile(
    r'(?<!\bvgl)(?<!\bArt)(?<!\bAbs)(?<!\blit)(?<!\bZiff)(?<!\bS)(?<!\bNr)'
    r'(?<=[.;!?])\s+'
)

def split_sentences(text: str) -> list[str]:
    """
    将 cc 文本切分为句子列表。
    针对德语法律文本做了保护，避免在缩写（Art. Abs. vgl. 等）处误切。
    """
    text = text.strip()
    # 保护常见缩写，替换成占位符
    placeholders = {
        "vgl. auch": "vgl_AUCH",
        "vgl.": "vgl_DOT",
        "Art.": "Art_DOT",
        "Abs.": "Abs_DOT",
        "lit.": "lit_DOT",
        "Ziff.": "Ziff_DOT",
        "Nr.": "Nr_DOT",
        "str.": "str_DOT",
        "a.M.": "aM_DOT",
        "a.A.": "aA_DOT",
        "h.L.": "hL_DOT",
        "h.M.": "hM_DOT",
        "i.V.m.": "iVm_DOT",
        "i.S.v.": "iSv_DOT",
        "s.": "s_DOT",
        "E.": "E_DOT",
        "SR ": "SR_SPACE",
        "AS ": "AS_SPACE",
        "BGE ": "BGE_SPACE",
    }
    protected = text
    for orig, ph in placeholders.items():
        protected = protected.replace(orig, ph)

    sentences = _SENT_SPLIT_RE.split(protected)

    # 恢复占位符
    restored = []
    for s in sentences:
        for orig, ph in placeholders.items():
            s = s.replace(ph, orig)
        s = s.strip()
        if s:
            restored.append(s)
    return restored


# citation 抽取正则
# 覆盖：Art. 123 OR / BGE 145 II 100 / SR 210 / AS 2020 100 / BV Art. 29
_CITATION_RE = re.compile(
    r"""
    (?:
        BGE\s+\d+\s+[IVX]+\s+\d+(?:\s+(?:E\.|Erw\.)\s*\d+(?:\.\d+)*)?  # BGE 145 II 100 E. 3.1
      | BGer[,\s]+\d{1,2}\.\d+\.\d{4}[^\s,;]*                            # BGer 4A_100/2020
      | Art\.\s*\d+(?:[a-z]?)\s*(?:Abs\.\s*\d+)?\s*(?:lit\.\s*[a-z])?\s*(?:Ziff\.\s*\d+)?\s*[A-Z]{2,5}  # Art. 41 Abs. 1 OR
      | Art\.\s*\d+(?:[a-z]?)\s*(?:Abs\.\s*\d+)?\s*(?:lit\.\s*[a-z])?\s*(?:Ziff\.\s*\d+)?               # Art. 41 Abs. 1
      | SR\s+\d{3}(?:\.\d+)+                                              # SR 210.0
      | AS\s+\d{4}\s+\d+                                                  # AS 2020 100
      | Botschaft\s+(?:vom\s+\d+\.\s*\w+\s+\d{4}|BBl\s+\d{4}\s+\d+)     # Botschaft BBl
      | BBl\s+\d{4}\s+\d+                                                 # BBl 2020 1234
    )
    """,
    re.VERBOSE,
)

def extract_citations(text: str) -> list[str]:
    """从文本中提取所有 citation，去重并保持顺序。"""
    found = _CITATION_RE.findall(text)
    seen = set()
    result = []
    for c in found:
        c = c.strip()
        if c and c not in seen:
            seen.add(c)
            result.append(c)
    return result


def find_citation_sentence(citation: str, sentences: list[str]) -> int:
    """返回 citation 所在句子的索引，找不到返回 -1。"""
    for i, s in enumerate(sentences):
        if citation in s:
            return i
    return -1


# ─────────────────────────────────────────────
# 4. 打分函数
# ─────────────────────────────────────────────

def score_keywords(sentence: str, window: str) -> tuple[float, float, int, int, float]:
    """
    计算句子和上下文窗口中的关键词得分。
    返回：(pos_score, neg_score, pos_count, neg_count, window_score)
    """
    s_lower = sentence.lower()
    w_lower = window.lower()

    pos_score, neg_score = 0.0, 0.0
    pos_count, neg_count = 0, 0

    for kw, weight in POSITIVE_KEYWORDS.items():
        if kw.lower() in s_lower:
            pos_score += weight
            pos_count += 1

    for kw, weight in NEGATIVE_KEYWORDS.items():
        if kw.lower() in s_lower:
            neg_score += weight
            neg_count += 1

    # 上下文窗口额外加成（权重折半）
    window_pos = sum(w * 0.5 for kw, w in POSITIVE_KEYWORDS.items() if kw.lower() in w_lower and kw.lower() not in s_lower)
    window_neg = sum(w * 0.5 for kw, w in NEGATIVE_KEYWORDS.items() if kw.lower() in w_lower and kw.lower() not in s_lower)
    window_score = window_pos - window_neg

    return pos_score, neg_score, pos_count, neg_count, window_score


def score_position(sent_idx: int, n_sentences: int) -> float:
    """
    位置得分：法律文本中，靠前的句子（引言/裁判要旨）和靠后（结论）权重更高。
    采用 U 型曲线：头尾高、中间低。
    """
    if n_sentences <= 1:
        return 1.0
    ratio = sent_idx / (n_sentences - 1)  # 0.0 ~ 1.0
    # U 型：score = 1 - 4*(ratio-0.5)^2  → 头尾=1，中间=0
    # 再做 min-max 到 [0.2, 1.0]
    u_score = 1.0 - 4.0 * (ratio - 0.5) ** 2
    return 0.2 + 0.8 * max(0.0, u_score)


def citation_type_bonus(citation: str) -> float:
    """根据 citation 字符串本身的类型给加成。"""
    bonus = 0.0
    for kw, w in CITATION_TYPE_BONUS.items():
        if kw in citation:
            bonus += w
    return bonus


def compute_cc_level_score(
    kw_pos: float,
    kw_neg: float,
    position_score: float,
    type_bonus: float,
    window_score: float,
    weights: Optional[dict[str, float]] = None,
) -> float:
    """
    综合 cc 内各维度得分，计算 cc 级别的 citation 得分。
    默认权重可通过 weights 参数覆盖。
    """
    w = weights or {
        "kw_net": 1.0,
        "position": 0.5,
        "type_bonus": 0.8,
        "window": 0.4,
    }
    kw_net = kw_pos - kw_neg
    score = (
        w["kw_net"] * kw_net
        + w["position"] * position_score
        + w["type_bonus"] * type_bonus
        + w["window"] * window_score
    )
    return score


def aggregate_query_level_score(
    citation_cc_scores: list[tuple[float, float, int]],
    rerank_score_weight: float = 1.0,
    cc_score_weight: float = 1.0,
    top_k_bonus: int = 3,
) -> float:
    """
    将同一 citation 在多个 cc 中的得分聚合为 query 级别得分。

    参数：
        citation_cc_scores: [(cc_level_score, rerank_score, cc_rank), ...]
        rerank_score_weight: rerank score 的权重
        cc_score_weight: cc_level_score 的权重
        top_k_bonus: 前 k 名 cc 中出现的 citation 额外加成

    策略：
        - 加权求和（rerank score × cc_level_score）
        - 出现在前 top_k_bonus 个 cc 中的额外加分
        - 出现频次本身也是信号（但权重较低）
    """
    if not citation_cc_scores:
        return 0.0

    total = 0.0
    freq_bonus = math.log1p(len(citation_cc_scores)) * 0.3  # 频次 log 加成

    for cc_score, rerank_score, cc_rank in citation_cc_scores:
        # rerank score 归一化到合理范围（假设 0~1，若不是请在调用前归一化）
        combined = cc_score_weight * cc_score + rerank_score_weight * rerank_score
        # 前 top_k 额外乘以 1.2
        if cc_rank <= top_k_bonus:
            combined *= 1.2
        total += combined

    return total + freq_bonus


# ─────────────────────────────────────────────
# 5. 主 Pipeline
# ─────────────────────────────────────────────

def score_citations_in_cc(
    cc: CCResult,
    weights: Optional[dict[str, float]] = None,
    window_size: int = 1,
) -> dict[str, CitationScore]:
    """
    对单个 cc 内的所有 citation 进行打分。

    返回：{citation_str: CitationScore}
    """
    sentences = split_sentences(cc.text)
    n = len(sentences)
    all_citations = extract_citations(cc.text)

    scores: dict[str, CitationScore] = {}

    for citation in all_citations:
        sent_idx = find_citation_sentence(citation, sentences)
        if sent_idx == -1:
            # 找不到所在句（罕见），用最后一句兜底
            sent_idx = n - 1

        sentence_text = sentences[sent_idx]

        # 上下文窗口
        win_start = max(0, sent_idx - window_size)
        win_end = min(n, sent_idx + window_size + 1)
        window_text = " ".join(sentences[win_start:win_end])

        # 各维度得分
        pos_s, neg_s, pos_c, neg_c, win_s = score_keywords(sentence_text, window_text)
        pos_score = score_position(sent_idx, n)
        t_bonus = citation_type_bonus(citation)
        cc_score = compute_cc_level_score(pos_s, neg_s, pos_score, t_bonus, win_s, weights)

        scores[citation] = CitationScore(
            citation=citation,
            cc_id=cc.cc_id,
            sentence_idx=sent_idx,
            sentence_text=sentence_text,
            kw_pos_score=round(pos_s, 4),
            kw_neg_score=round(neg_s, 4),
            kw_pos_count=pos_c,
            kw_neg_count=neg_c,
            position_score=round(pos_score, 4),
            type_bonus=round(t_bonus, 4),
            window_score=round(win_s, 4),
            cc_level_score=round(cc_score, 4),
        )

    return scores


def score_citations_for_query(
    cc_list: list[CCResult],
    weights: Optional[dict[str, float]] = None,
    rerank_score_weight: float = 1.0,
    cc_score_weight: float = 1.0,
    top_k_bonus: int = 3,
    window_size: int = 1,
) -> list[dict]:
    """
    对一个 query 的 cc_list 进行完整 citation 打分流程。

    返回：按 query_level_score 降序排列的 citation 列表，每项包含详细信息。
    """
    # citation → [(cc_level_score, rerank_score, cc_rank)]
    citation_appearances: dict[str, list[tuple[float, float, int]]] = defaultdict(list)
    # citation → 所有 CitationScore（用于详情输出）
    citation_detail: dict[str, list[CitationScore]] = defaultdict(list)

    for cc in cc_list:
        cc_scores = score_citations_in_cc(cc, weights=weights, window_size=window_size)
        for citation, cs in cc_scores.items():
            citation_appearances[citation].append(
                (cs.cc_level_score, cc.rerank_score, cc.rank)
            )
            citation_detail[citation].append(cs)

    # 汇总为 query 级别得分
    results = []
    for citation, appearances in citation_appearances.items():
        q_score = aggregate_query_level_score(
            appearances,
            rerank_score_weight=rerank_score_weight,
            cc_score_weight=cc_score_weight,
            top_k_bonus=top_k_bonus,
        )
        # 找最高 cc_level_score 的那条详情作为代表
        best_cs = max(citation_detail[citation], key=lambda x: x.cc_level_score)
        best_cs.query_level_score = round(q_score, 4)

        results.append({
            "citation": citation,
            "query_level_score": round(q_score, 4),
            "appearances": len(appearances),
            "best_cc_id": best_cs.cc_id,
            "best_cc_level_score": best_cs.cc_level_score,
            "best_sentence": best_cs.sentence_text,
            "kw_pos_score": best_cs.kw_pos_score,
            "kw_neg_score": best_cs.kw_neg_score,
            "position_score": best_cs.position_score,
            "type_bonus": best_cs.type_bonus,
            "window_score": best_cs.window_score,
            "all_cc_scores": [
                {"cc_level_score": s, "rerank_score": r, "cc_rank": k}
                for s, r, k in appearances
            ],
        })

    results.sort(key=lambda x: x["query_level_score"], reverse=True)
    return results


# ─────────────────────────────────────────────
# 6. 批量处理多个 query
# ─────────────────────────────────────────────

def process_dataset(
    dataset: list[dict],
    weights: Optional[dict[str, float]] = None,
    rerank_score_weight: float = 1.0,
    cc_score_weight: float = 1.0,
    top_k_bonus: int = 3,
    window_size: int = 1,
) -> list[dict]:
    """
    批量处理数据集。

    dataset 格式：
    [
        {
            "query_id": "q001",
            "query": "...",
            "cc_list": [
                {
                    "cc_id": "cc001",
                    "text": "...",
                    "rerank_score": 0.92,
                    "rank": 1
                },
                ...
            ],
            "gold_citations": ["Art. 41 OR", "BGE 132 III 100"]  # 可选，用于评估
        },
        ...
    ]

    返回：
    [
        {
            "query_id": "q001",
            "query": "...",
            "ranked_citations": [...],   # 按 query_level_score 降序
            "gold_citations": [...],
        },
        ...
    ]
    """
    results = []
    for sample in dataset:
        cc_list = [
            CCResult(
                cc_id=cc["cc_id"],
                text=cc["text"],
                rerank_score=cc["rerank_score"],
                rank=cc["rank"],
            )
            for cc in sample["cc_list"]
        ]

        ranked = score_citations_for_query(
            cc_list,
            weights=weights,
            rerank_score_weight=rerank_score_weight,
            cc_score_weight=cc_score_weight,
            top_k_bonus=top_k_bonus,
            window_size=window_size,
        )

        results.append({
            "query_id": sample.get("query_id", ""),
            "query": sample.get("query", ""),
            "ranked_citations": ranked,
            "gold_citations": sample.get("gold_citations", []),
        })

    return results


# ─────────────────────────────────────────────
# 7. 评估函数
# ─────────────────────────────────────────────

def evaluate(results: list[dict], top_k_list: list[int] = [1, 3, 5, 10]) -> dict:
    """
    计算 Recall@K 和 MAP。
    """
    recall_at_k = {k: [] for k in top_k_list}
    average_precisions = []

    for sample in results:
        gold = set(sample.get("gold_citations", []))
        if not gold:
            continue

        ranked = [r["citation"] for r in sample["ranked_citations"]]

        # Recall@K
        for k in top_k_list:
            top_k_cits = set(ranked[:k])
            hit = len(gold & top_k_cits)
            recall_at_k[k].append(hit / len(gold))

        # Average Precision
        hits, ap = 0, 0.0
        for i, cit in enumerate(ranked, 1):
            if cit in gold:
                hits += 1
                ap += hits / i
        if hits > 0:
            ap /= len(gold)
        average_precisions.append(ap)

    metrics = {f"Recall@{k}": round(sum(v) / len(v), 4) for k, v in recall_at_k.items() if v}
    metrics["MAP"] = round(sum(average_precisions) / len(average_precisions), 4) if average_precisions else 0.0
    return metrics


# ─────────────────────────────────────────────
# 8. 演示
# ─────────────────────────────────────────────

# ── 加载数据 ──────────────────────────────────────────────────────────────────
import pandas as pd
print("Loading data...")
court_consideration_df = pd.read_csv("../data/court_considerations.csv")
court_consideration_d = dict(zip(court_consideration_df['citation'].tolist(), court_consideration_df['text'].tolist()))

import common
train_candidate_d = common.read_candidate("../data/rule_based/raw_train_candidate.pkl", court_consideration_d)
train_df = pd.read_csv("../data/train_rewrite_001.csv")
train_gold_citations_d = { query_id: gold_citations.split(';') for query_id,gold_citations in zip(train_df['query_id'], train_df['gold_citations'])}

def build_dataset():
    
    ret = []
    for query_id, data_d in train_candidate_d.items():
        d = {}
        d['query_id'] = query_id
        d['query'] = query_id

        cc_list = []
        cc_list_base = data_d['rerank']
        for rank, cc in enumerate(cc_list_base,start=1):
            hit, score = cc
            cc_id = hit['citation']
            text = hit['text']
            d2 = {}
            d2["cc_id"] = cc_id
            d2['text'] = text
            d2['rerank_score'] = score
            d2['rank'] = rank
            cc_list.append(d2)

        d['cc_list'] = cc_list
        d['gold_citations'] = train_gold_citations_d.get(d['query_id'])

        ret.append(d)

    return ret


import pprint
import json

if __name__ == "__main__":
    # ── 构造示例数据 ──────────────────────────────

    sample_dataset = build_dataset()

    with open("../data/rule_based/train.json", "w+") as of:
        json.dump(sample_dataset, of)

    pprint.pprint(sample_dataset[0])

    # sample_dataset = [
    #     {
    #         "query_id": "q001",
    #         "query": "Haftung bei Vertragsverletzung im Schweizer Recht",
    #         "cc_list": [
    #             {
    #                 "cc_id": "cc001",
    #                 "text": (
    #                     "Gemäss Art. 97 Abs. 1 OR haftet der Schuldner für den Schaden, "
    #                     "der dem Gläubiger durch die Nichterfüllung entsteht. "
    #                     "Dies gilt namentlich dann, wenn der Schuldner beweisen kann, "
    #                     "dass ihn kein Verschulden trifft. "
    #                     "Vgl. auch BGE 130 III 182 E. 3.2, wonach die Beweislast beim Schuldner liegt. "
    #                     "Allgemein anerkannt ist, dass Art. 41 OR die ausservertragliche Haftung regelt; "
    #                     "es ist jedoch fraglich, ob diese Bestimmung analog angewendet werden kann."
    #                 ),
    #                 "rerank_score": 0.92,
    #                 "rank": 1,
    #             },
    #             {
    #                 "cc_id": "cc002",
    #                 "text": (
    #                     "Das Bundesgericht hat in BGE 133 III 295 festgehalten, "
    #                     "dass Art. 97 OR i.V.m. Art. 101 OR anwendbar ist. "
    #                     "Entgegen der herrschenden Lehre (h.L.) vertritt eine Mindermeinung (a.M.), "
    #                     "dass SR 220 in diesem Kontext nicht massgebend ist. "
    #                     "Somit ergibt sich aus der Rechtsprechung, dass Art. 42 OR unmittelbar findet Anwendung."
    #                 ),
    #                 "rerank_score": 0.85,
    #                 "rank": 2,
    #             },
    #             {
    #                 "cc_id": "cc003",
    #                 "text": (
    #                     "Gestützt auf Art. 41 OR sowie BGE 115 II 440 ist die Schadenersatzpflicht gegeben. "
    #                     "Vgl. z.B. auch BGer 4A_100/2020, wo ähnliche Fragen offen gelassen wurden. "
    #                     "Ausdrücklich vorgesehen ist die Haftung in Art. 41 Abs. 1 OR."
    #                 ),
    #                 "rerank_score": 0.78,
    #                 "rank": 3,
    #             },
    #         ],
    #         "gold_citations": ["Art. 97 Abs. 1 OR", "BGE 130 III 182", "BGE 133 III 295"],
    #     }
    # ]

    # ── 运行 Pipeline ─────────────────────────────

    print("=" * 60)
    print("Citation Scoring Pipeline — Swiss German Legal Text")
    print("=" * 60)

    results = process_dataset(
        sample_dataset,
        rerank_score_weight=1.0,
        cc_score_weight=1.0,
        top_k_bonus=3,
        window_size=1,
    )

    for sample_result in results:
        print(f"\nQuery [{sample_result['query_id']}]: {sample_result['query']}")
        print(f"Gold citations: {sample_result['gold_citations']}")
        print(f"\n{'Rank':<5} {'Citation':<40} {'Q-Score':>8} {'CC-Score':>9} {'Appear':>7} {'Best CC'}")
        print("-" * 85)

        for rank, cit in enumerate(sample_result["ranked_citations"], 1):
            print(
                f"{rank:<5} {cit['citation']:<40} "
                f"{cit['query_level_score']:>8.3f} "
                f"{cit['best_cc_level_score']:>9.3f} "
                f"{cit['appearances']:>7} "
                f"{cit['best_cc_id']}"
            )
            print(f"       Sentence: {cit['best_sentence'][:80]}...")
            print(f"       kw_pos={cit['kw_pos_score']:.2f}  kw_neg={cit['kw_neg_score']:.2f}  "
                  f"pos={cit['position_score']:.2f}  type={cit['type_bonus']:.2f}  win={cit['window_score']:.2f}")
            print()

    # ── 评估 ──────────────────────────────────────

    metrics = evaluate(results, top_k_list=[1, 3, 5, 10, 20])
    print("\n── Evaluation Metrics ──")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")

    # ── 特征矩阵导出（供 LightGBM 使用）────────────

    print("\n── Feature Matrix Preview (for LightGBM) ──")
    feature_rows = []
    for sample_result in results:
        gold = set(sample_result.get("gold_citations", []))
        for cit in sample_result["ranked_citations"]:
            feature_rows.append({
                "query_id": sample_result["query_id"],
                "citation": cit["citation"],
                "label": int(cit["citation"] in gold),
                "query_level_score": cit["query_level_score"],
                "appearances": cit["appearances"],
                "best_cc_level_score": cit["best_cc_level_score"],
                "kw_pos_score": cit["kw_pos_score"],
                "kw_neg_score": cit["kw_neg_score"],
                "position_score": cit["position_score"],
                "type_bonus": cit["type_bonus"],
                "window_score": cit["window_score"],
            })

    print(f"{'citation':<40} {'label':>5} {'q_score':>8} {'appear':>6} {'kw_pos':>6} {'kw_neg':>6}")
    print("-" * 75)
    for row in feature_rows[:10]:
        print(
            f"{row['citation']:<40} {row['label']:>5} "
            f"{row['query_level_score']:>8.3f} {row['appearances']:>6} "
            f"{row['kw_pos_score']:>6.2f} {row['kw_neg_score']:>6.2f}"
        )