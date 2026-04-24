from __future__ import annotations
import math
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import bm25s
import Stemmer  # optional: for stemming

cc_df = pd.read_csv("../data/court_considerations.csv")
print("cc_df loaded.")

court_doc = [{'citation':citation, 'text':text} for citation,text in zip(cc_df['citation'], cc_df['text'])]

retriever = bm25s.BM25.load("../data/bm25/cc", load_corpus=True)

valid_df = pd.read_csv("../data/valid_rewrite_001.csv")
v_qid_2_query = {}
for query_id, query in zip(valid_df['query_id'], valid_df['query2']):
    v_qid_2_query[query_id] = query

valid_df = pd.read_csv("../data/valid_rewrite_001.csv")
v_qid_2_gold = {}
for query_id, golds in zip(valid_df['query_id'], valid_df['gold_citations']):
    v_qid_2_gold[query_id] = set(golds.split(";"))

src_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)
import citation_utils
import metric_utils

# optional: create a stemmer
stemmer = Stemmer.Stemmer("german")

v_qid_2_query_list = defaultdict(list)
valid_df2 = pd.read_csv("../data/valid_10variants_2.csv")
for query_id, query in zip(valid_df2['query_id'], valid_df2['query']):
    v_qid_2_query_list[query_id].append(query)

# for query_id, query in v_qid_2_query.items():
#     v_qid_2_query_list[query_id].append(query)

v_qid_2_predict = {}
for query_id, query_list in v_qid_2_query_list.items():
    cid_set = set()
    
    for query in query_list:
        query_tokens = bm25s.tokenize(query, stemmer=stemmer)

        results, scores = retriever.retrieve(query_tokens, k=500)

        for i in range(results.shape[1]):
            idx, score = results[0, i], scores[0, i]
            # print(f"Rank {i+1} (score: {score:.2f}): {doc}")
            text = court_doc[idx]['text']
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
