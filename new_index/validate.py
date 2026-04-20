from __future__ import annotations
import math
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm
from pathlib import Path
from more_itertools import chunked
from dense_index import DenseIndex

model = BGEM3FlagModel('/root/.cache/modelscope/hub/models/BAAI/bge-m3', use_fp16=True)
path_str = '../data/processed/new_index_fact'

cc_df = pd.read_csv("../data/court_considerations.csv")
print("cc_df loaded.")

court_doc = [{'citation':citation, 'text':text} for citation,text in zip(cc_df['citation'], cc_df['text'])]

index = DenseIndex(model, path_str, court_doc)

valid_df = pd.read_csv("../data/valid_rewrite_split_question_001.csv")
v_qid_2_fact = {}
for query_id, query in zip(valid_df['query_id'], valid_df['query']):
    v_qid_2_fact[query_id] = query.rsplit("\n\n", 1)[0]

valid_df = pd.read_csv("../data/valid_rewrite_001.csv")
v_qid_2_gold = {}
for query_id, golds in zip(valid_df['query_id'], valid_df['gold_citations']):
    v_qid_2_gold[query_id] = set(golds.split(";"))

src_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)
import citation_utils
import metric_utils

v_qid_2_predict = {}
for query_id, fact in v_qid_2_fact.items():
    cid_set = set()
    results = index.search(fact, 200)
    for r in results:
        text = r['text']
        cids = citation_utils.extract_citations_from_text(text)
        for cid in cids:
            cid_set.add(cid)
    v_qid_2_predict[query_id] = cid_set

result_l = []
gold_l = []
for query_id, predict in v_qid_2_predict.items():
    result_l.append(list(predict))
    gold_l.append(list(v_qid_2_gold[query_id]))

recall = metric_utils.cal_recall(result_l, gold_l)

print(recall)
