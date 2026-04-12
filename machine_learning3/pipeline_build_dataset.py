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

OUTPUT_DIR = "../data/ml3/lgbm_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 加载数据 ──────────────────────────────────────────────────────────────────
print("Loading data...")
court_consideration_df = pd.read_csv("../data/court_considerations.csv")
court_consideration_d = dict(zip(court_consideration_df['citation'].tolist(), court_consideration_df['text'].tolist()))

court_doc = [{'citation':citation, 'text':text} for citation,text in zip(court_consideration_df['citation'].tolist(), court_consideration_df['text'].tolist())]

import common
train_candidate_d = common.read_candidate("../data/ml2/raw_train_candidate.pkl", court_consideration_d)
valid_candidate_d = common.read_candidate("../data/ml2/raw_valid_candidate.pkl", court_consideration_d)
test_candidate_d = common.read_candidate("../data/ml2/raw_test_candidate.pkl", court_consideration_d)

def _maxmin_normalize_hits(hits):
    max_value = hits[0][1]
    min_value = hits[0][1]
    for i in range(1, len(hits)):
        max_value = max(max_value, hits[i][1])
        min_value = min(min_value, hits[i][1])
    span = max_value - min_value

    ret = [hit.copy() for hit in hits]
    for hit in ret:
        hit[1] = (hit[1] - min_value) * 1. / span

    return ret

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

    norm_hits1 = _maxmin_normalize_hits(hits1)
    norm_hits2 = _maxmin_normalize_hits(hits2)
    norm_hits3 = _maxmin_normalize_hits(hits3)

    dense_score_d  = {hit['citation']: score for hit, score in hits1} # Dict[cc_id->score]
    sparse_score_d = {hit['citation']: score for hit, score in hits2}
    rerank_score_d = {hit['citation']: score for hit, score in hits3}
    norm_dense_score_d  = {hit['citation']: score for hit, score in norm_hits1}
    norm_sparse_score_d = {hit['citation']: score for hit, score in norm_hits2}
    norm_rerank_score_d = {hit['citation']: score for hit, score in norm_hits3}
    
    accum: dict[str, dict] = defaultdict(lambda: {
        norm_dense_score: []
        norm_sparse_score: []
        norm_rerank_score: []
    })

    for hit_rank, (hit, rerank_score) in enumerate(norm_hits3):
        cc_id = hit['citation']
        accum.append(dense_score_d.get(cc_id, 0.))
        accum.append(sparse_score_d.get(cc_id, 0.))
        accum.append(rerank_score_d.get(cc_id, 0.))
        
    
    # 整理为特征向量
    cid_feat_d: dict[str, np.ndarray] = {}

    for cid, a in accum.items():
        # freq = a["cite_freq"]

        feat_vec = np.array([
            # 检索信号（18 维）
            
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

    

    