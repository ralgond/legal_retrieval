from __future__ import annotations
import os
import json
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

DATA_DIR   = "../data/ml5/lgbm_data"
OUTPUT_DIR = "../data/ml5/lgbm_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 加载数据 ──────────────────────────────────────────────────────────────────
X_tr = np.load(f"{DATA_DIR}/train_features.npy")
y_tr = np.load(f"{DATA_DIR}/train_labels.npy")
g_tr = np.load(f"{DATA_DIR}/train_groups.npy")

with open(f"{DATA_DIR}/feature_names.txt") as f:
    feature_names = [l.strip() for l in f.readlines()]

train_ds = lgb.Dataset(X_tr, label=y_tr, group=g_tr, feature_name=feature_names)

params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [1, 5, 10, 20],
    'learning_rate': 0.01,
    'num_leaves': 32,
    'max_depth': 6,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 2,
    'seed': 42,

    'boosting_type' : "gbdt",
    'label_gain'    : [0, 1],

    "lambda_l1":        0.1,
    "lambda_l2":        0.1,

    "min_data_in_leaf": 1, 
    "lambdarank_truncation_level": 200, 

    "feature_pre_filter" : False,
}


booster = lgb.train(
        params,
        train_ds,
        num_boost_round = 1
    )

model_path = f"{OUTPUT_DIR}/lgbm_ranker_all.txt"
booster.save_model(model_path)
print(f"Model saved → {model_path}")