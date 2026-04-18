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
test_candidate_d = common.read_candidate("../data/ml6/raw_test_candidate.pkl", court_consideration_d)

# ── 推理主循环 ────────────────────────────────────────────────────────────────
test_df  = pd.read_csv("../data/test_rewrite_001.csv")
result_l  = []
gold_l    = []

query_id_l = []
for i, (query_id, query) in enumerate(zip(test_df['query_id'], test_df['query'])):

    print(f"[{i+1}/{len(test_df)}] query_id={query_id}")
    query_id_l.append(query_id)

    # 1. 特征提取
    cid_feat_d = extract_features_for_query(query_id, query, test_candidate_d, {})

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
    top_cids = [all_cids[idx] for idx in order[:TOP_K]]
    # top_cids = [all_cids[idx] for idx in order]
    result_l.append(';'.join(top_cids))

df = pd.DataFrame({"query_id":query_id_l, "predicted_citations":result_l})
df.to_csv("../data/ml6/prediction.csv", index=False)