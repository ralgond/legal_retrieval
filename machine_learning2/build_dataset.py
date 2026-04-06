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

from FlagEmbedding import FlagReranker, BGEM3FlagModel
from dense_index_bge import DenseIndex
from sparse_index import SparseIndex
import reranker_utils
import hits_utils
import citation_utils

# ── 超参数 ────────────────────────────────────────────────────────────────────
TOP_HIT_THRESHOLD = 10   # top-hit bonus 阈值
RETRIEVE_TOP_K    = 100  # 检索召回数
CANDIDATE_TOP_K   = 50   # 每个 query 保留的候选 citation 数（送入 LGB 排序）
OUTPUT_DIR        = "./lgbm_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 加载模型 ──────────────────────────────────────────────────────────────────
print("Loading models...")
court_consideration_df = pd.read_csv("../data/court_considerations.csv")
court_consideration_d = dict(zip(court_consideration_df['citation'].tolist(), court_consideration_df['text'].tolist()))

court_doc = [{'citation':citation, 'text':text} for citation,text in zip(court_consideration_df['citation'].tolist(), court_consideration_df['text'].tolist())]

# -- load index and reranker --
model = BGEM3FlagModel('/root/.cache/modelscope/hub/models/BAAI/bge-m3', use_fp16=True, show_progress_bar=False)
dense_index = DenseIndex(model, "/root/autodl-fs/bge-processed/_dense_sparse_court/", court_doc)
sparse_index = SparseIndex(model, "/root/autodl-fs/bge-processed/_dense_sparse_court/", court_doc)
reranker = FlagReranker('/root/.cache/modelscope/hub/models/BAAI/bge-reranker-v2-m3', use_fp16=True, normalize=True)

print("Models loaded.")

# ── 特征名（顺序固定，训练/推理必须一致） ─────────────────────────────────────
FEATURE_NAMES = [
    # 原始特征
    "log_pos_decay",       # reranker_score * 1/log(2+idx)          ← 保留原始
    # 新增特征
    "hit_rank_decay",      # reranker_score * 1/log(2+hit_rank)
    "rr_pos_decay",        # reranker_score * 1/(1+idx)
    "dense_log_pos",       # dense_score    * 1/log(2+idx)
    "sparse_log_pos",      # sparse_score   * 1/log(2+idx)
    "cite_freq",           # 出现次数（raw）
    "log_cite_freq",       # log(1+freq)
    "top_hit_bonus",       # 仅 top-N hit 贡献的 log_pos_decay
    "max_score_pos",       # 单次最大 score*log_pos
    "score_sq_log_pos",    # score^2 * 1/log(2+idx)
    # 额外统计特征
    "mean_reranker_score", # citation 贡献 hit 的平均 reranker 分
    "max_reranker_score",  # citation 贡献 hit 的最大 reranker 分
    "min_hit_rank",        # citation 最早出现的 hit 排名（越小越好）
    "sum_dense_score",     # dense 分累计
    "sum_sparse_score",    # sparse 分累计
    "dense_sparse_ratio",  # sum_dense / (sum_sparse + 1e-9)
    "hit_coverage",        # 该 citation 出现在多少个不同 hit 中
    "weighted_hit_coverage",  # hit_coverage 按 reranker_score 加权
]
N_FEATS = len(FEATURE_NAMES)
print(f"Feature dim: {N_FEATS}  {FEATURE_NAMES}")
 
 
def extract_features_for_query(
        query: str,
        dense_index, sparse_index, reranker,
        retrieve_top_k: int = RETRIEVE_TOP_K,
        top_hit_threshold: int = TOP_HIT_THRESHOLD,
) -> dict[str, list]:
    """
    对单个 query 做检索+rerank，返回
      { citation_id: np.ndarray(N_FEATS,) }
    """
    hits1 = dense_index.search_with_score(query, top_k=retrieve_top_k)
    hits2 = sparse_index.search_with_score(query, top_k=retrieve_top_k)
 
    dense_score_d  = {hit['citation']: score for hit, score in hits1}
    sparse_score_d = {hit['citation']: score for hit, score in hits2}
 
    _hits = hits_utils.merge_hits_with_score_l_by_max(hits1, hits2)
    hit_with_score_l = reranker_utils.rerank_by_batch_chunked2(
        reranker, query, [hit for hit, _ in _hits])
 
    # 累积结构
    accum: dict[str, dict] = defaultdict(lambda: {
        "log_pos_decay":       0.0,
        "hit_rank_decay":      0.0,
        "rr_pos_decay":        0.0,
        "dense_log_pos":       0.0,
        "sparse_log_pos":      0.0,
        "cite_freq":           0,
        "top_hit_bonus":       0.0,
        "max_score_pos":       0.0,
        "score_sq_log_pos":    0.0,
        "reranker_scores":     [],   # 用于计算 mean/max
        "hit_ranks":           [],   # 用于计算 min_hit_rank
        "sum_dense_score":     0.0,
        "sum_sparse_score":    0.0,
        "hit_ids":             set(),  # 用于 hit_coverage
        "weighted_hit_coverage": 0.0,
    })
 
    for hit_rank, (hit, reranker_score) in enumerate(hit_with_score_l):
        parsed_cc  = citation_utils.parse_cc_output_citations_and_sentences(hit['text'])
        hit_doc_id = hit['citation']
        dense_score  = dense_score_d.get(hit_doc_id, 0.0)
        sparse_score = sparse_score_d.get(hit_doc_id, 0.0)
 
        for cid, idx in parsed_cc['citations']:
            a = accum[cid]
            log_pos   = 1.0 / math.log(2 + idx)
            rr_pos    = 1.0 / (1 + idx)
            hit_decay = 1.0 / math.log(2 + hit_rank)
 
            a["log_pos_decay"]    += reranker_score * log_pos
            a["hit_rank_decay"]   += reranker_score * hit_decay
            a["rr_pos_decay"]     += reranker_score * rr_pos
            a["dense_log_pos"]    += dense_score * log_pos
            a["sparse_log_pos"]   += sparse_score * log_pos
            a["cite_freq"]        += 1
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
 
    # 整理为特征向量
    cid_feat_d: dict[str, np.ndarray] = {}
    total_freq = sum(v["cite_freq"] for v in accum.values()) + 1e-9
 
    for cid, a in accum.items():
        freq = a["cite_freq"]
        rs   = a["reranker_scores"]
        feat_vec = np.array([
            a["log_pos_decay"],
            a["hit_rank_decay"],
            a["rr_pos_decay"],
            a["dense_log_pos"],
            a["sparse_log_pos"],
            float(freq),
            math.log(1 + freq),
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
        ], dtype=np.float32)
        assert len(feat_vec) == N_FEATS, f"Feature dim mismatch: {len(feat_vec)} vs {N_FEATS}"
        cid_feat_d[cid] = feat_vec
 
    return cid_feat_d
 
 
def build_split(df: pd.DataFrame, split_name: str,
                min_pos_in_candidate: int = 1,
                only_filter_train:    bool = True):
    """
    对一个 split（train 或 valid）构造特征矩阵和标签。
    每个 query 对应 CANDIDATE_TOP_K 个候选，按 log_pos_decay 排序后取 top。
 
    Parameters
    ----------
    min_pos_in_candidate : int
        候选集中至少需要包含多少个正样本，不足则跳过该 query。
        仅在 only_filter_train=True 且 split_name=="train" 时生效，
        或 only_filter_train=False 时对所有 split 生效。
    only_filter_train : bool
        True  → 只对 train split 做过滤（valid 保持完整，便于公平评估）
        False → 所有 split 都过滤
    """
    do_filter = (split_name == "train") if only_filter_train else True
 
    all_feats, all_labels, groups = [], [], []
    skipped, total = 0, 0
 
    for i, (query_id, query, gold_citations) in enumerate(
            zip(df['query_id'], df['query2'], df['gold_citations'])):
        gold_set = set(gold_citations.split(';'))
        total += 1
        print(f"[{split_name}] {i+1}/{len(df)}  query_id={query_id}")
 
        cid_feat_d = extract_features_for_query(
            query, dense_index, sparse_index, reranker)
 
        # 按原始特征 log_pos_decay 粗排，取 top CANDIDATE_TOP_K
        sorted_cids = sorted(
            cid_feat_d.keys(),
            key=lambda c: cid_feat_d[c][FEATURE_NAMES.index("log_pos_decay")],
            reverse=True
        )[:CANDIDATE_TOP_K]
 
        # ── 正样本过滤 ────────────────────────────────────────────────────────
        if do_filter:
            pos_in_candidate = sum(1 for c in sorted_cids if c in gold_set)
            if pos_in_candidate < min_pos_in_candidate:
                skipped += 1
                print(f"  ↳ SKIP  pos_in_candidate={pos_in_candidate} "
                      f"< min_pos_in_candidate={min_pos_in_candidate}  "
                      f"(gold_size={len(gold_set)})")
                continue
        # ─────────────────────────────────────────────────────────────────────
 
        for cid in sorted_cids:
            all_feats.append(cid_feat_d[cid])
            all_labels.append(1 if cid in gold_set else 0)
 
        groups.append(len(sorted_cids))
 
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
        "--min_pos", type=int, default=2,
        help="候选集中至少包含多少个正样本才保留该 query（仅对 train 生效）。"
             "默认 1，即丢弃候选集中 0 个正样本的 query。"
             "设为 2 则更严格，只保留至少命中 2 个 gold 的 query。")
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
                only_filter_train=not args.filter_valid)
    build_split(valid_df, "valid",
                min_pos_in_candidate=args.min_pos,
                only_filter_train=not args.filter_valid)