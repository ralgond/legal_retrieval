"""
inference_lgbm.py
用训练好的 LightGBM 模型对 valid 集推理并评估 Recall / Precision。
"""

from __future__ import annotations
import math
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
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
import metric_utils

# ── 超参数 ────────────────────────────────────────────────────────────────────
TOP_HIT_THRESHOLD = 10
RETRIEVE_TOP_K    = 100
CANDIDATE_TOP_K   = 50
TOP_K             = 40
MODEL_PATH        = "./lgbm_model/lgbm_ranker.txt"
DATA_DIR          = "./lgbm_data"

# ── 特征名 ────────────────────────────────────────────────────────────────────
with open(f"{DATA_DIR}/feature_names.txt") as f:
    FEATURE_NAMES = [l.strip() for l in f.readlines()]
N_FEATS = len(FEATURE_NAMES)

# ── 加载 LightGBM 模型 ────────────────────────────────────────────────────────
print(f"Loading LightGBM model from {MODEL_PATH} ...")
booster = lgb.Booster(model_file=MODEL_PATH)
print(f"LightGBM model loaded. num_trees={booster.num_trees()}")

# ── 加载检索模型 ──────────────────────────────────────────────────────────────
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


# ── 特征提取 ──────────────────────────────────────────────────────────────────
def extract_features_for_query(query: str) -> dict[str, np.ndarray]:
    hits1 = dense_index.search_with_score(query, top_k=RETRIEVE_TOP_K)
    hits2 = sparse_index.search_with_score(query, top_k=RETRIEVE_TOP_K)

    dense_score_d  = {hit['citation']: score for hit, score in hits1}
    sparse_score_d = {hit['citation']: score for hit, score in hits2}

    _hits = hits_utils.merge_hits_with_score_l_by_max(hits1, hits2)
    hit_with_score_l = reranker_utils.rerank_by_batch_chunked2(
        reranker, query, [hit for hit, _ in _hits])

    accum: dict[str, dict] = defaultdict(lambda: {
        "log_pos_decay":         0.0,
        "hit_rank_decay":        0.0,
        "rr_pos_decay":          0.0,
        "dense_log_pos":         0.0,
        "sparse_log_pos":        0.0,
        "cite_freq":             0,
        "top_hit_bonus":         0.0,
        "max_score_pos":         0.0,
        "score_sq_log_pos":      0.0,
        "reranker_scores":       [],
        "hit_ranks":             [],
        "sum_dense_score":       0.0,
        "sum_sparse_score":      0.0,
        "hit_ids":               set(),
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
            if hit_rank < TOP_HIT_THRESHOLD:
                a["top_hit_bonus"] += val

            a["reranker_scores"].append(reranker_score)
            a["hit_ranks"].append(hit_rank)
            a["hit_ids"].add(hit_doc_id)
            a["weighted_hit_coverage"] += reranker_score

    cid_feat_d: dict[str, np.ndarray] = {}
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
        assert len(feat_vec) == N_FEATS, \
            f"Feature dim mismatch: {len(feat_vec)} vs {N_FEATS}"
        cid_feat_d[cid] = feat_vec

    return cid_feat_d


# ── 推理主循环 ────────────────────────────────────────────────────────────────
valid_df  = pd.read_csv("../data/valid_rewrite_001.csv")
result_l  = []
gold_l    = []

for i, (query_id, query, gold_citations) in enumerate(
        zip(valid_df['query_id'], valid_df['query2'], valid_df['gold_citations'])):

    gold_set = set(gold_citations.split(';'))
    gold_l.append(list(gold_set))
    print(f"[{i+1}/{len(valid_df)}] query_id={query_id}")

    # 1. 特征提取
    cid_feat_d = extract_features_for_query(query)

    # 2. 粗排，取候选
    sorted_cids = sorted(
        cid_feat_d.keys(),
        key=lambda c: cid_feat_d[c][FEATURE_NAMES.index("log_pos_decay")],
        reverse=True,
    )[:CANDIDATE_TOP_K]

    if not sorted_cids:
        result_l.append([])
        continue

    # 3. 构造特征矩阵
    X = np.stack([cid_feat_d[c] for c in sorted_cids], axis=0)  # (n_cand, N_FEATS)

    # 4. LightGBM 打分
    scores = booster.predict(X)  # (n_cand,)

    # 5. 按分数降序，取 TOP_K
    order    = np.argsort(scores)[::-1]
    top_cids = [sorted_cids[idx] for idx in order[:TOP_K]]
    result_l.append(top_cids)

    

# ── 评估 ──────────────────────────────────────────────────────────────────────
for limit in [5,7,10,12,15,17,20,22,25,27,30,32,35,37,40]:
    result_l2 = [r[:limit] for r in result_l]
    recall    = metric_utils.cal_recall(result_l2, gold_l)
    precision = metric_utils.cal_precision(result_l2, gold_l)
    print(f"[{limit}] Recall:{recall:.4f}, Precision:{precision:.4f}, F1:{metric_utils.cal_f1(recall, precision):.4f}")


# ── 预测 ──────────────────────────────────────────────────────────────────────
test_df  = pd.read_csv("../data/test_rewrite_001.csv")
result_l  = []

for i, (query_id, query) in enumerate(zip(test_df['query_id'], test_df['query'])):

    print(f"[{i+1}/{len(test_df)}] query_id={query_id}")

    # 1. 特征提取
    cid_feat_d = extract_features_for_query(query)

    # 2. 粗排，取候选
    sorted_cids = sorted(
        cid_feat_d.keys(),
        key=lambda c: cid_feat_d[c][FEATURE_NAMES.index("log_pos_decay")],
        reverse=True,
    )[:CANDIDATE_TOP_K]

    if not sorted_cids:
        result_l.append([])
        continue

    # 3. 构造特征矩阵
    X = np.stack([cid_feat_d[c] for c in sorted_cids], axis=0)  # (n_cand, N_FEATS)

    # 4. LightGBM 打分
    scores = booster.predict(X)  # (n_cand,)

    # 5. 按分数降序，取 TOP_K
    order    = np.argsort(scores)[::-1]
    top_cids = [sorted_cids[idx] for idx in order[:TOP_K]]
    result_l.append(top_cids)

query_l = []
predicted_citations_l = []
for query_id, result in zip(test_df['query_id'], result_l):
    query_l.append(query_id)
    predicted_citations_l.append(';'.join(result))

result_df = pd.DataFrame({'query_id':query_l, 'predicted_citations':predicted_citations_l})
result_df.to_csv("../data/prediction.csv", index=False)