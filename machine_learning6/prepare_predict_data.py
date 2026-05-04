import common
import re
import pandas as pd
import math
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import random
import json

src_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

import citation_utils

# ── 加载数据 ──────────────────────────────────────────────────────────────────
print("Loading data...")
court_consideration_df = pd.read_csv("../data/court_considerations.csv")
court_consideration_d = dict(zip(court_consideration_df['citation'].tolist(), court_consideration_df['text'].tolist()))

train_candidate_d = common.read_candidate("../data/ml6/raw_train_candidate.pkl", court_consideration_d)
valid_candidate_d = common.read_candidate("../data/ml6/raw_valid_candidate.pkl", court_consideration_d)
test_candidate_d = common.read_candidate("../data/ml6/raw_test_candidate.pkl", court_consideration_d)


test_df = pd.read_csv("../data/test_rewrite_001.csv")
test_qid_2_query = {}
for query_id, query in zip(test_df['query_id'], test_df['query']):
    test_qid_2_query[query_id] = query

OUTPUT="../data/ml6"
# ── 生成数据 ──────────────────────────────────────────────────────────────────
l = []
for query_id, _d in test_candidate_d.items():
    d = {}
    d['query_id'] = query_id
    d['query'] = test_qid_2_query[query_id]
    court_considerations = []
    for cc, rerank_score in _d['rerank']:
        cc_d = {}
        cc_d['cc_id'] = cc['citation']
        cc_d['cc_text'] = cc['text']
        court_considerations.append(cc_d)
    d['cc_list'] = court_considerations
    l.append(d)

with open(f"{OUTPUT}/predict.jsonl", "w+", encoding="utf-8") as of:
    for d in l:
        of.write(json.dumps(d, ensure_ascii=False)+"\n")