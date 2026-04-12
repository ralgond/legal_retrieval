"""
build_dataset.py
构造 LightGBM Learning-to-Rank 训练集 / 验证集。

输出：
  - train_features.npy   (N_train_pairs, N_feats)
  - train_labels.npy     (N_train_pairs,)   0/1 relevance
  - train_groups.npy     (N_train_queries,) 每个 query 的候选数
  - valid_features.npy / valid_labels.npy / valid_groups.npy  同上
"""

from __future__ import annotations
import math
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

src_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

import reranker_utils
import hits_utils
import citation_utils

# ── 超参数 ────────────────────────────────────────────────────────────────────
TOP_HIT_THRESHOLD = 10   # top-hit bonus 阈值
RETRIEVE_TOP_K    = 100  # 检索召回数
OUTPUT_DIR        = "../data/ml2/lgbm_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 加载数据 ──────────────────────────────────────────────────────────────────
print("Loading data...")
court_consideration_df = pd.read_csv("../data/court_considerations.csv")
court_consideration_d = dict(zip(court_consideration_df['citation'].tolist(), court_consideration_df['text'].tolist()))

court_doc = [{'citation':citation, 'text':text} for citation,text in zip(court_consideration_df['citation'].tolist(), court_consideration_df['text'].tolist())]

# ── 特征名（顺序固定，训练/推理必须一致） ─────────────────────────────────────
FEATURE_NAMES = [
    # ── 检索信号特征 ──────────────────────────────────────────────────────────
    "log_pos_decay",          # reranker_score * 1/log(2+idx)
    # "hit_rank_decay",         # reranker_score * 1/log(2+hit_rank)
    "rr_pos_decay",           # reranker_score * 1/(1+idx)
    "dense_log_pos",          # dense_score    * 1/log(2+idx)
    "sparse_log_pos",         # sparse_score   * 1/log(2+idx)
    # "cite_freq",              # 出现次数（raw）
    # "log_cite_freq",          # log(1+freq)
    "top_hit_bonus",          # 仅 top-N hit 贡献的 log_pos_decay
    "max_score_pos",          # 单次最大 score*log_pos
    "score_sq_log_pos",       # score^2 * 1/log(2+idx)
    "mean_reranker_score",    # 贡献 hit 的平均 reranker 分
    "max_reranker_score",     # 贡献 hit 的最大 reranker 分
    "min_hit_rank",           # 最早出现的 hit 排名
    "sum_dense_score",        # dense 分累计
    "sum_sparse_score",       # sparse 分累计
    "dense_sparse_ratio",     # sum_dense / (sum_sparse + 1e-9)
    "hit_coverage",           # 出现在多少个不同 hit 中
    "weighted_hit_coverage",  # hit_coverage 按 reranker_score 加权
    # ── 句子级文本特征（citation 所在句子）────────────────────────────────────
    "mean_sent_char_len",     # 所在句子的平均字符长度
    "max_sent_char_len",      # 所在句子的最大字符长度
    "mean_sent_word_len",     # 所在句子的平均词数
    "mean_sent_pos_ratio",    # 句子在段落中的平均相对位置（0=开头, 1=结尾）
    "min_sent_pos_ratio",     # 句子在段落中的最早出现位置
    # ── Query-Citation 文本匹配特征 ───────────────────────────────────────────
    "mean_query_term_overlap", # query 词覆盖率：命中的 query 词 / query 总词数（均值）
    "max_query_term_overlap",  # query 词覆盖率最大值
    "mean_char_bigram_overlap", # query 与句子的字符级 bigram overlap（均值）
    # ── Citation 文档级全局特征 ───────────────────────────────────────────────
    "doc_char_len",            # citation 对应文档的字符长度
    "doc_word_len",            # citation 对应文档的词数
    "doc_total_citations",     # 文档内包含的 citation 总数
]
N_FEATS = len(FEATURE_NAMES)
print(f"Feature dim: {N_FEATS}  {FEATURE_NAMES}")

import common
train_candidate_d = common.read_candidate("../data/ml2/raw_train_candidate.pkl", court_consideration_d)
valid_candidate_d = common.read_candidate("../data/ml2/raw_valid_candidate.pkl", court_consideration_d)
test_candidate_d = common.read_candidate("../data/ml2/raw_test_candidate.pkl", court_consideration_d)


def _char_bigram_overlap(a: str, b: str) -> float:
    """字符级 bigram 的 Jaccard overlap，衡量两段文本的字面相似度。"""
    def bigrams(s: str) -> set:
        s = s.lower()
        return {s[i:i+2] for i in range(len(s) - 1)} if len(s) >= 2 else set()
    bg_a, bg_b = bigrams(a), bigrams(b)
    if not bg_a or not bg_b:
        return 0.0
    return len(bg_a & bg_b) / len(bg_a | bg_b)


def _query_term_overlap(query_tokens: set[str], sentence: str) -> float:
    """query 词在句子中的覆盖率（命中词数 / query 总词数）。"""
    if not query_tokens:
        return 0.0
    sent_tokens = set(sentence.lower().split())
    return len(query_tokens & sent_tokens) / len(query_tokens)


def extract_features_for_query(
        query_id: str, query: str,
        retrieve_top_k: int = RETRIEVE_TOP_K,
        top_hit_threshold: int = TOP_HIT_THRESHOLD,
) -> dict[str, np.ndarray]:
    """
    对单个 query 做检索+rerank，返回
      { citation_id: np.ndarray(N_FEATS,) }
    """
    if query_id in train_candidate_d:
        hits1 = train_candidate_d[query_id]['dense']
        hits2 = train_candidate_d[query_id]['sparse']
        hits3 = train_candidate_d[query_id]['rerank']
    elif query_id in valid_candidate_d:
        hits1 = valid_candidate_d[query_id]['dense']
        hits2 = valid_candidate_d[query_id]['sparse']
        hits3 = valid_candidate_d[query_id]['rerank']
    elif query_id in test_candidate_d:
        hits1 = test_candidate_d[query_id]['dense']
        hits2 = test_candidate_d[query_id]['sparse']
        hits3 = test_candidate_d[query_id]['rerank']
    else:
        return {}


    dense_score_d  = {hit['citation']: score for hit, score in hits1}
    sparse_score_d = {hit['citation']: score for hit, score in hits2}
    hit_with_score_l = hits3

    # query 预处理（用于文本匹配特征）
    query_tokens = set(query.lower().split())

    # 累积结构
    accum: dict[str, dict] = defaultdict(lambda: {
        # 检索信号
        "log_pos_decay":       0.0,
        # "hit_rank_decay":      0.0,
        "rr_pos_decay":        0.0,
        "dense_log_pos":       0.0,
        "sparse_log_pos":      0.0,
        # "cite_freq":           0,
        "top_hit_bonus":       0.0,
        "max_score_pos":       0.0,
        "score_sq_log_pos":    0.0,
        "reranker_scores":     [],
        "hit_ranks":           [],
        "sum_dense_score":     0.0,
        "sum_sparse_score":    0.0,
        "hit_ids":             set(),
        "weighted_hit_coverage": 0.0,
        # 句子文本特征（每次 citation 出现对应一条句子记录）
        "sent_char_lens":      [],   # 句子字符长度列表
        "sent_word_lens":      [],   # 句子词数列表
        "sent_pos_ratios":     [],   # 句子在段落中的相对位置
        # Query-Citation 文本匹配
        "query_term_overlaps": [],   # 每条句子的 query 词覆盖率
        "char_bigram_overlaps":[],   # 每条句子的字符 bigram overlap
    })

    for hit_rank, (hit, reranker_score) in enumerate(hit_with_score_l):
        parsed_cc  = citation_utils.parse_cc_output_citations_and_sentences(hit['text'])
        hit_doc_id = hit['citation']
        dense_score  = dense_score_d.get(hit_doc_id, 0.0)
        sparse_score = sparse_score_d.get(hit_doc_id, 0.0)

        # 段落总句数，用于计算句子相对位置
        sentences = parsed_cc.get('sentences', [])
        total_sents = max(len(sentences), 1)

        for cid, idx in parsed_cc['citations']:
            a = accum[cid]
            log_pos   = 1.0 / math.log(2 + idx)
            rr_pos    = 1.0 / (1 + idx)
            hit_decay = 1.0 / math.log(2 + hit_rank)

            # ── 检索信号 ──────────────────────────────────────────────────────
            a["log_pos_decay"]    += reranker_score * log_pos
            # a["hit_rank_decay"]   += reranker_score * hit_decay
            a["rr_pos_decay"]     += reranker_score * rr_pos
            a["dense_log_pos"]    += dense_score * log_pos
            a["sparse_log_pos"]   += sparse_score * log_pos
            # a["cite_freq"]        += 1
            a["score_sq_log_pos"] += (reranker_score ** 2) * log_pos
            a["sum_dense_score"]  += dense_score
            a["sum_sparse_score"] += sparse_score

            val = reranker_score * log_pos
            if val > a["max_score_pos"]:
                a["max_score_pos"] = val
            if hit_rank < top_hit_threshold:
                a["top_hit_bonus"] += val

            a["reranker_scores"].append(reranker_score)
            a["hit_ranks"].append(hit_rank)
            a["hit_ids"].add(hit_doc_id)
            a["weighted_hit_coverage"] += reranker_score

            # ── 文本特征：取 citation 所在句子（idx 即句子下标）─────────────
            sent_text = sentences[idx] if idx < len(sentences) else ""
            char_len  = len(sent_text)
            word_len  = len(sent_text.split())
            pos_ratio = idx / total_sents          # 0=段落开头, 趋近1=结尾

            a["sent_char_lens"].append(char_len)
            a["sent_word_lens"].append(word_len)
            a["sent_pos_ratios"].append(pos_ratio)

            # Query-Citation 匹配
            a["query_term_overlaps"].append(_query_term_overlap(query_tokens, sent_text))
            a["char_bigram_overlaps"].append(_char_bigram_overlap(query, sent_text))

    # ── 文档级全局特征（与 query 无关，每个 cid 只算一次）────────────────────
    # court_consideration_d: { citation -> full_doc_text }，在模块顶层加载
    def _doc_feats(cid: str) -> tuple[float, float, float]:
        doc_text = court_consideration_d.get(cid, "")
        char_len  = float(len(doc_text))
        word_len  = float(len(doc_text.split()))
        # 文档中 citation 标记的粗略计数（依赖具体格式，可按实际调整）
        total_cit = float(doc_text.count("[") )   # 假设 citation 用 [xx] 标注
        return char_len, word_len, total_cit

    # 整理为特征向量
    cid_feat_d: dict[str, np.ndarray] = {}

    for cid, a in accum.items():
        # freq = a["cite_freq"]
        rs   = a["reranker_scores"]
        scl  = a["sent_char_lens"]   or [0]
        swl  = a["sent_word_lens"]   or [0]
        spr  = a["sent_pos_ratios"]  or [0.0]
        qto  = a["query_term_overlaps"]  or [0.0]
        cbo  = a["char_bigram_overlaps"] or [0.0]
        doc_char, doc_word, doc_ncit = _doc_feats(cid)

        feat_vec = np.array([
            # 检索信号（18 维）
            a["log_pos_decay"],
            # a["hit_rank_decay"],
            a["rr_pos_decay"],
            a["dense_log_pos"],
            a["sparse_log_pos"],
            # float(freq),
            # math.log(1 + freq),
            a["top_hit_bonus"],
            a["max_score_pos"],
            a["score_sq_log_pos"],
            float(np.mean(rs)),
            float(np.max(rs)),
            float(min(a["hit_ranks"])),
            a["sum_dense_score"],
            a["sum_sparse_score"],
            a["sum_dense_score"] / (a["sum_sparse_score"] + 1e-9),
            float(len(a["hit_ids"])),
            a["weighted_hit_coverage"],
            # 句子级文本特征（5 维）
            float(np.mean(scl)),
            float(np.max(scl)),
            float(np.mean(swl)),
            float(np.mean(spr)),
            float(np.min(spr)),
            # Query-Citation 匹配特征（3 维）
            float(np.mean(qto)),
            float(np.max(qto)),
            float(np.mean(cbo)),
            # 文档级特征（3 维）
            doc_char,
            doc_word,
            doc_ncit,
        ], dtype=np.float32)
        assert len(feat_vec) == N_FEATS, \
            f"Feature dim mismatch: {len(feat_vec)} vs {N_FEATS}  cid={cid}"
        cid_feat_d[cid] = feat_vec

    return cid_feat_d


def build_split(df: pd.DataFrame, split_name: str,
                min_pos_in_candidate: int = 1,
                only_filter_train:    bool = True,
                neg_pos_ratio:        int  = 10):
    """
    对一个 split（train 或 valid）构造特征矩阵和标签。
    每个 query 的所有候选 citation（由检索+rerank 产生）全部参与特征提取；
    训练集对负样本做随机下采样以缓解类别不平衡，valid 集保持完整。

    Parameters
    ----------
    min_pos_in_candidate : int
        候选集中至少需要包含多少个正样本，不足则跳过该 query。
        仅在 only_filter_train=True 且 split_name=="train" 时生效，
        或 only_filter_train=False 时对所有 split 生效。
    only_filter_train : bool
        True  → 只对 train split 做过滤和负采样（valid 保持完整，便于公平评估）
        False → 所有 split 都过滤和负采样
    neg_pos_ratio : int
        训练时每个正样本保留多少个负样本（默认 10，即 1:10）。
        设为 -1 则不做负采样（保留全部负样本）。
        val split 始终不做负采样。
    """
    do_filter     = (split_name == "train") if only_filter_train else True
    do_neg_sample = (split_name == "train") if only_filter_train else True

    rng = np.random.default_rng(seed=42)

    all_feats, all_labels, groups = [], [], []
    skipped, total = 0, 0

    for i, (query_id, query, gold_citations) in enumerate(
            zip(df['query_id'], df['query2'], df['gold_citations'])):
        gold_set = set(gold_citations.split(';'))
        total += 1
        print(f"[{split_name}] {i+1}/{len(df)}  query_id={query_id}")

        cid_feat_d = extract_features_for_query(query_id, query)

        # 全部候选，分为正负两组
        all_cids = list(cid_feat_d.keys())
        pos_cids = [c for c in all_cids if c in gold_set]
        neg_cids = [c for c in all_cids if c not in gold_set]

        # ── 正样本过滤 ────────────────────────────────────────────────────────
        if do_filter:
            if len(pos_cids) < min_pos_in_candidate:
                skipped += 1
                print(f"  ↳ SKIP  pos={len(pos_cids)} "
                      f"< min_pos_in_candidate={min_pos_in_candidate}  "
                      f"(gold_size={len(gold_set)})")
                continue
        # ─────────────────────────────────────────────────────────────────────

        # ── 负采样（仅 train，val 保持完整） ──────────────────────────────────
        if do_neg_sample and neg_pos_ratio >= 0:
            max_neg = len(pos_cids) * neg_pos_ratio
            if len(neg_cids) > max_neg:
                neg_cids = list(rng.choice(neg_cids, size=max_neg, replace=False))
        # ─────────────────────────────────────────────────────────────────────

        sampled_cids = pos_cids + neg_cids
        print(f"  pos={len(pos_cids)}  neg={len(neg_cids)}  "
              f"ratio=1:{len(neg_cids)//max(len(pos_cids),1)}")

        for cid in sampled_cids:
            all_feats.append(cid_feat_d[cid])
            all_labels.append(1 if cid in gold_set else 0)

        groups.append(len(sampled_cids))

    kept = total - skipped
    print(f"\n[{split_name}] kept={kept}/{total}  skipped={skipped}  "
          f"(filter={'on' if do_filter else 'off'}, "
          f"min_pos_in_candidate={min_pos_in_candidate})")

    X = np.stack(all_feats, axis=0)          # (N, N_FEATS)
    y = np.array(all_labels, dtype=np.int32) # (N,)
    g = np.array(groups,     dtype=np.int32) # (n_queries,)
    print(f"[{split_name}] X={X.shape}  pos_rate={y.mean():.4f}")

    np.save(f"{OUTPUT_DIR}/{split_name}_features.npy", X)
    np.save(f"{OUTPUT_DIR}/{split_name}_labels.npy",   y)
    np.save(f"{OUTPUT_DIR}/{split_name}_groups.npy",   g)
    print(f"[{split_name}] Saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min_pos", type=int, default=1,
        help="候选集中至少包含多少个正样本才保留该 query（仅对 train 生效）。"
             "默认 1，即丢弃候选集中 0 个正样本的 query。"
             "设为 2 则更严格，只保留至少命中 2 个 gold 的 query。")
    parser.add_argument(
        "--neg_pos_ratio", type=int, default=10,
        help="训练集每个正样本保留多少负样本（默认 10）。-1 表示不采样（保留全部）。")
    parser.add_argument(
        "--filter_valid", action="store_true",
        help="同时对 valid 集做相同过滤（默认只过滤 train）")
    args = parser.parse_args()

    # 保存特征名供后续使用
    with open(f"{OUTPUT_DIR}/feature_names.txt", "w") as f:
        f.write("\n".join(FEATURE_NAMES))

    train_df = pd.read_csv("../data/train_rewrite_001.csv")
    valid_df  = pd.read_csv("../data/valid_rewrite_001.csv")

    build_split(train_df, "train",
                min_pos_in_candidate=args.min_pos,
                only_filter_train=not args.filter_valid,
                neg_pos_ratio=args.neg_pos_ratio)
    build_split(valid_df, "valid",
                min_pos_in_candidate=args.min_pos,
                only_filter_train=not args.filter_valid,
                neg_pos_ratio=args.neg_pos_ratio)
