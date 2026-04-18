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

import reranker_utils
import hits_utils
import citation_utils
import metric_utils

from pipeline_common import _maxmin_normalize_hits, CC, Citation, Query, extract_features_for_query

# ── 超参数 ────────────────────────────────────────────────────────────────────
TOP_HIT_THRESHOLD = 10
RETRIEVE_TOP_K    = 100
TOP_K             = 20
MODEL_PATH        = "../data/ml6/lgbm_model/lgbm_ranker.txt"
DATA_DIR          = "../data/ml6/lgbm_data"

# ── 特征名 ────────────────────────────────────────────────────────────────────
with open(f"{DATA_DIR}/feature_names.txt") as f:
    FEATURE_NAMES = [l.strip() for l in f.readlines()]
N_FEATS = len(FEATURE_NAMES)

# ── 加载 LightGBM 模型 ────────────────────────────────────────────────────────
print(f"Loading LightGBM model from {MODEL_PATH} ...")
booster = lgb.Booster(model_file=MODEL_PATH)
print(f"LightGBM model loaded. num_trees={booster.num_trees()}")

# ── 加载检索模型 ──────────────────────────────────────────────────────────────
court_consideration_df = pd.read_csv("../data/court_considerations.csv")
court_consideration_d = dict(zip(court_consideration_df['citation'].tolist(), court_consideration_df['text'].tolist()))

print("Loaded cc...")

import common
train_candidate_d = common.read_candidate("../data/ml6/raw_train_candidate.pkl", court_consideration_d)
valid_candidate_d = common.read_candidate("../data/ml6/raw_valid_candidate.pkl", court_consideration_d)
test_candidate_d = common.read_candidate("../data/ml6/raw_test_candidate.pkl", court_consideration_d)

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
    cid_feat_d = extract_features_for_query(query_id, query, valid_candidate_d, gold_set)

    if not cid_feat_d:
        result_l.append([])
        continue

    # 全部候选，不做截断
    all_cids = list(cid_feat_d.keys())

    # 2. 构造特征矩阵
    X = np.stack([cid_feat_d[c] for c in all_cids], axis=0)  # (n_cand, N_FEATS)

    # 3. LightGBM 打分
    scores = booster.predict(X)  # (n_cand,)

    # 4. 按分数降序，取 TOP_K
    order    = np.argsort(scores)[::-1]
    # top_cids = [all_cids[idx] for idx in order[:TOP_K]]
    top_cids = [all_cids[idx] for idx in order]
    result_l.append(top_cids)

# ── 评估 ──────────────────────────────────────────────────────────────────────
for TOP_K in [5,7,10,12,15,17,20,22,25,27,30,33,35,37,40]:
    result_l2 = [r[:TOP_K] for r in result_l]
    recall    = metric_utils.cal_recall(result_l2, gold_l)
    precision = metric_utils.cal_precision(result_l2, gold_l)
    print(f"[{TOP_K}] Recall@{TOP_K}:{recall:.4f}, Precision:{precision:.4f}, F1:{2*recall*precision/(recall+precision):.4f}")
