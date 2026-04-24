
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

court_consideration_df = pd.read_csv("../data/court_considerations.csv")
court_consideration_d = dict(zip(court_consideration_df['citation'], court_consideration_df['text']))
print("data loaded.")

import common
train_candidate_d = common.read_candidate("../data/ml4/raw_train_candidate.pkl", court_consideration_d)
valid_candidate_d = common.read_candidate("../data/ml4/raw_valid_candidate.pkl", court_consideration_d)
test_candidate_d = common.read_candidate("../data/ml4/raw_test_candidate.pkl", court_consideration_d)

import citation_utils

train_query_id_2_gold_citation_l = []
train_query_id_2_citation_l = []
train_df = pd.read_csv("../data/train_rewrite_001.csv")
for query_id, gold_citations in zip(train_df['query_id'], train_df['gold_citations']):
    train_query_id_2_gold_citation_l.append(gold_citations.split(';'))
    
    hits_l = train_candidate_d[query_id]['rerank']
    citation_set = set()
    for hit, score in hits_l:
        cids = citation_utils.extract_citations_from_text(hit['text'])
        for cid in cids:
            citation_set.add(cid)
    train_query_id_2_citation_l.append(list(citation_set))

import metric_utils
recall = metric_utils.cal_recall(train_query_id_2_citation_l, train_query_id_2_gold_citation_l)

print(recall)