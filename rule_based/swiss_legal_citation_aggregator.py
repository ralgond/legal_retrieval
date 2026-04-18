"""
Citation Aggregator
===================
将 query → cc → citation 三层结构聚合为 query 级别的引用分数。

依赖：swiss_legal_citation_analyzer.py（提供 CitationResult 数据结构）
不重复实现 citation_score 计算逻辑。

数据流：
    query
     └── cc_1 (cc_score)  →  [CitationResult, ...]
     └── cc_2 (cc_score)  →  [CitationResult, ...]
     └── ...
           ↓
    {citation_text: CitationQueryScore}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional
import math


# ---------------------------------------------------------------------------
# 1. 输入 / 输出数据结构
# ---------------------------------------------------------------------------

@dataclass
class CCEntry:
    """一个 cc 及其权重，以及该 cc 下所有引用的得分"""
    cc_id: str
    cc_score: float                          # query 下该 cc 的权重（任意正数）
    citation_scores: dict[str, float]        # {citation_text: score}


@dataclass
class CitationQueryScore:
    """一个引用在 query 级别的聚合结果"""
    citation: str
    score: float                             # 核心聚合分数
    confidence: float                        # 0~1，可信度
    consistency: str                         # consistent_positive / consistent_negative / contradictory
    coverage: int                            # 出现在几个 cc 里
    total_cc_weight: float                   # 出现的 cc 的 cc_score 之和
    category: str = "INCIDENTAL"   # ← 新增
    pos_score: Optional[float] = None        # 矛盾时：正向加权均值
    neg_score: Optional[float] = None        # 矛盾时：负向加权均值
    contributing_ccs: list[tuple[str, float, float]] = field(default_factory=list)
    # [(cc_id, cc_score, citation_score), ...]


# ---------------------------------------------------------------------------
# 2. 一致性检查
# ---------------------------------------------------------------------------

def check_consistency(citation_scores: list[float]) -> str:
    """
    检查同一引用在不同 cc 中的符号是否一致。
    忽略得分为 0 的条目。
    """
    nonzero = [s for s in citation_scores if s != 0.0]
    if not nonzero:
        return "neutral"
    positives = sum(1 for s in nonzero if s > 0)
    negatives = sum(1 for s in nonzero if s < 0)
    if negatives == 0:
        return "consistent_positive"
    if positives == 0:
        return "consistent_negative"
    return "contradictory"


# ---------------------------------------------------------------------------
# 3. 加权平均（基础工具）
# ---------------------------------------------------------------------------

def weighted_mean(weights: list[float], values: list[float]) -> float:
    total_w = sum(weights)
    if total_w == 0.0:
        return 0.0
    return sum(w * v for w, v in zip(weights, values)) / total_w


# ---------------------------------------------------------------------------
# 4. 单引用聚合
# ---------------------------------------------------------------------------

def _categorize(score: float) -> str:
    if score >= 2.0:  return "CORE"
    if score <= -2.0: return "NEGATIVE"
    return "INCIDENTAL"

def _aggregate_one(
    citation: str,
    entries: list[tuple[str, float, float]],   # [(cc_id, cc_score, citation_score)]
    max_possible_cc_weight: float,
) -> CitationQueryScore:
    """
    对一个引用跨所有 cc 的观测值进行聚合。

    策略：
      - consistent_positive / consistent_negative：加权平均
      - contradictory：正负分别加权平均，再按各自 cc_score 总量合并
      - 附加置信度 = cc覆盖度 × 一致性系数
    """
    cc_ids      = [e[0] for e in entries]
    cc_scores   = [e[1] for e in entries]
    cite_scores = [e[2] for e in entries]

    consistency = check_consistency(cite_scores)
    coverage    = len(entries)
    total_w     = sum(cc_scores)

    if consistency in ("consistent_positive", "consistent_negative", "neutral"):
        score     = weighted_mean(cc_scores, cite_scores)
        pos_score = None
        neg_score = None

    else:  # contradictory
        pos_pairs = [(cw, cs) for cw, cs in zip(cc_scores, cite_scores) if cs > 0]
        neg_pairs = [(cw, cs) for cw, cs in zip(cc_scores, cite_scores) if cs < 0]

        pos_w     = sum(p[0] for p in pos_pairs)
        neg_w     = sum(p[0] for p in neg_pairs)

        pos_score = weighted_mean([p[0] for p in pos_pairs], [p[1] for p in pos_pairs])
        neg_score = weighted_mean([p[0] for p in neg_pairs], [p[1] for p in neg_pairs])

        # 按正负各自的 cc_score 总量做最终合并
        denom = pos_w + neg_w
        score = (pos_score * pos_w + neg_score * neg_w) / denom if denom else 0.0

    # 置信度
    coverage_ratio     = total_w / max_possible_cc_weight if max_possible_cc_weight > 0 else 0.0
    consistency_factor = 1.0 if consistency != "contradictory" else 0.6
    confidence         = min(1.0, coverage_ratio * consistency_factor)

    return CitationQueryScore(
        citation          = citation,
        score             = round(score, 4),
        confidence        = round(confidence, 4),
        consistency       = consistency,
        coverage          = coverage,
        total_cc_weight   = round(total_w, 4),
        pos_score         = round(pos_score, 4) if pos_score is not None else None,
        neg_score         = round(neg_score, 4) if neg_score is not None else None,
        contributing_ccs  = entries,
        category          = _categorize(score),   # ← 新增
    )


# ---------------------------------------------------------------------------
# 5. 主聚合函数
# ---------------------------------------------------------------------------

def aggregate(
    cc_entries: list[CCEntry],
    score_threshold: Optional[float] = None,
    top_k: Optional[int] = None,
    normalize_cc_scores: bool = True,
) -> list[CitationQueryScore]:
    """
    将 query 下所有 cc 的引用得分聚合为 query 级别排序列表。

    参数
    ----
    cc_entries          : 每个 cc 的 CCEntry 列表
    score_threshold     : 过滤掉 |score| < threshold 的引用（可选）
    top_k               : 只返回得分最高的 k 个引用（可选）
    normalize_cc_scores : 是否先将 cc_score 归一化为概率（默认 True）

    返回
    ----
    按 score 降序排列的 CitationQueryScore 列表
    """
    if not cc_entries:
        return []

    # ── 归一化 cc_score ──────────────────────────────────────────────────────
    raw_weights = [e.cc_score for e in cc_entries]
    total_raw   = sum(raw_weights)

    if normalize_cc_scores and total_raw > 0:
        norm_weights = [w / total_raw for w in raw_weights]
    else:
        norm_weights = raw_weights

    max_possible = sum(norm_weights)   # 归一化后为 1.0，否则为原始总和

    # ── 收集每个引用的所有观测 ───────────────────────────────────────────────
    # {citation_text: [(cc_id, cc_score_norm, citation_score), ...]}
    citation_obs: dict[str, list[tuple[str, float, float]]] = defaultdict(list)

    for entry, norm_w in zip(cc_entries, norm_weights):
        for cite_text, cite_score in entry.citation_scores.items():
            citation_obs[cite_text].append((entry.cc_id, norm_w, cite_score))

    # ── 逐引用聚合 ───────────────────────────────────────────────────────────
    results: list[CitationQueryScore] = []
    for citation, obs in citation_obs.items():
        result = _aggregate_one(citation, obs, max_possible)
        results.append(result)

    # ── 过滤 + 排序 ──────────────────────────────────────────────────────────
    if score_threshold is not None:
        results = [r for r in results if abs(r.score) >= score_threshold]

    results.sort(key=lambda r: r.score, reverse=True)

    if top_k is not None:
        results = results[:top_k]

    return results


# ---------------------------------------------------------------------------
# 6. 与 swiss_legal_citation_analyzer 的对接适配器
# ---------------------------------------------------------------------------

def from_analyzer_results(
    cc_id: str,
    cc_score: float,
    citation_results: list,          # list[CitationResult] from analyzer
) -> CCEntry:
    """
    将 swiss_legal_citation_analyzer.analyze_text() 返回的
    CitationResult 列表转换为 CCEntry。

    用法：
        from swiss_legal_citation_analyzer import analyze_text
        doc, results = analyze_text(cc_text, nlp)
        entry = from_analyzer_results("cc_001", 0.8, results)
    """
    return CCEntry(
        cc_id            = cc_id,
        cc_score         = cc_score,
        citation_scores  = {r.citation.original: r.score for r in citation_results},
    )


# ---------------------------------------------------------------------------
# 7. 报告输出
# ---------------------------------------------------------------------------

_CATEGORY_COLOR = {
    "consistent_positive":  "\033[34m",   # 蓝
    "consistent_negative":  "\033[31m",   # 红
    "contradictory":        "\033[33m",   # 黄
    "neutral":              "\033[37m",   # 灰
}
_RESET = "\033[0m"

_CONSISTENCY_LABEL = {
    "consistent_positive": "一致正向",
    "consistent_negative": "一致负向",
    "contradictory":       "矛盾",
    "neutral":             "中性",
}


def print_aggregation_report(
    results: list[CitationQueryScore],
    use_color: bool = True,
) -> None:
    print("\n" + "=" * 70)
    print("  Query 级别引用聚合报告")
    print("=" * 70)

    for rank, r in enumerate(results, 1):
        color = _CATEGORY_COLOR.get(r.consistency, "") if use_color else ""
        reset = _RESET if use_color else ""
        cons_label = _CONSISTENCY_LABEL.get(r.consistency, r.consistency)

        print(f"\n#{rank:02d}  {color}{r.citation}{reset}")
        print(f"      得分       : {r.score:+.4f}")
        print(f"      置信度     : {r.confidence:.4f}")
        print(f"      一致性     : {cons_label}")
        print(f"      出现 cc 数 : {r.coverage}  (cc_score 合计 {r.total_cc_weight:.4f})")

        if r.consistency == "contradictory":
            print(f"      正向均值   : {r.pos_score:+.4f}")
            print(f"      负向均值   : {r.neg_score:+.4f}")

        print("      贡献明细   :")
        for cc_id, cc_w, cs in r.contributing_ccs:
            bar = "▲" if cs >= 0 else "▼"
            print(f"        {bar} {cc_id:15s}  cc_score={cc_w:.4f}  cite_score={cs:+.2f}")

    print("\n" + "─" * 70)
    core = sum(1 for r in results if r.score >= 2.0)
    incid = sum(1 for r in results if -2.0 < r.score < 2.0)
    neg = sum(1 for r in results if r.score <= -2.0)
    print(f"  汇总：核心 {core}  附带 {incid}  负面 {neg}  共 {len(results)} 个")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# 8. 演示入口
# ---------------------------------------------------------------------------

def main() -> None:
    """
    最小可运行演示，不依赖 spaCy / swiss_legal_citation_analyzer，
    直接手动构造 CCEntry 展示聚合逻辑。
    """

    cc_entries = [
        CCEntry(
            cc_id        = "cc_001",
            cc_score     = 0.8,
            citation_scores = {
                "Art. 41 Abs. 1 OR": +3.0,
                "Art. 44 OR":        -1.0,
                "BGE 140 III 115":   +2.5,
            },
        ),
        CCEntry(
            cc_id        = "cc_002",
            cc_score     = 0.5,
            citation_scores = {
                "Art. 41 Abs. 1 OR": +1.5,   # 同一引用，仍正向但较弱
                "Art. 28a ZGB":      +3.0,
                "BGE 140 III 115":   -2.0,   # ← 与 cc_001 矛盾！
            },
        ),
        CCEntry(
            cc_id        = "cc_003",
            cc_score     = 0.3,
            citation_scores = {
                "Art. 43 Abs. 2 OR": -3.5,
                "Art. 44 OR":        -0.5,
                "Art. 28a ZGB":      +2.0,
            },
        ),
    ]

    print("输入数据：")
    for e in cc_entries:
        print(f"  [{e.cc_id}] cc_score={e.cc_score}")
        for cite, score in e.citation_scores.items():
            print(f"    {score:+.1f}  {cite}")

    results = aggregate(
        cc_entries,
        normalize_cc_scores = True,
        score_threshold     = None,
        top_k               = None,
    )

    print_aggregation_report(results)


if __name__ == "__main__":
    main()