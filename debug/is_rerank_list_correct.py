
import math
import os
import sys
import numpy as np
import pandas as pd

# ── 加载数据 ──────────────────────────────────────────────────────────────────
print("Loading data...")
court_consideration_df = pd.read_csv("../data/court_considerations.csv")
court_consideration_d = dict(zip(court_consideration_df['citation'].tolist(), court_consideration_df['text'].tolist()))


import common

train_candidate_d = common.read_candidate("../data/ml5/raw_train_candidate.pkl", court_consideration_d)
valid_candidate_d = common.read_candidate("../data/ml5/raw_valid_candidate.pkl", court_consideration_d)
test_candidate_d = common.read_candidate("../data/ml5/raw_test_candidate.pkl", court_consideration_d)

# print(valid_candidate_d.keys())

# print(valid_candidate_d['val_001']['rerank'])


print(test_candidate_d.keys())

print(test_candidate_d['test_001']['rerank'])
