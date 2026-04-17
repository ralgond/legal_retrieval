# 分析每个gold citation所在cc的句子的特征

from __future__ import annotations
import math
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import random

src_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

import citation_utils

OUTPUT_DIR = "../data/rule_based/lgbm_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 加载数据 ──────────────────────────────────────────────────────────────────
print("Loading data...")
court_consideration_df = pd.read_csv("../data/court_considerations.csv")
court_consideration_d = dict(zip(court_consideration_df['citation'].tolist(), court_consideration_df['text'].tolist()))

import common
train_candidate_d = common.read_candidate("../data/rule_based/raw_train_candidate.pkl", court_consideration_d)
valid_candidate_d = common.read_candidate("../data/rule_based/raw_valid_candidate.pkl", court_consideration_d)
test_candidate_d = common.read_candidate("../data/rule_based/raw_test_candidate.pkl", court_consideration_d)


def build_split(df: pd.DataFrame, split_name: str,
                min_pos_in_candidate: int = 1,
                only_filter_train:    bool = True,
                neg_pos_ratio:        int  = 10)

    for i, (query_id, query, gold_citations) in enumerate(
            zip(df['query_id'], df['query2'], df['gold_citations'])):
        
    