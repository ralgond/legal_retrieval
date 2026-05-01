from __future__ import annotations
import math
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import Stemmer
import bm25s

src_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from FlagEmbedding import FlagReranker, BGEM3FlagModel
from dense_index_bge import DenseIndex
from sparse_index import SparseIndex
import reranker_utils
import hits_utils
import citation_utils
import pickle
from tqdm import tqdm
import metric_utils

# ── 加载模型 ──────────────────────────────────────────────────────────────────
print("Loading models...")
court_consideration_df = pd.read_csv("../data/court_considerations.csv")
court_consideration_d = dict(zip(court_consideration_df['citation'].tolist(), court_consideration_df['text'].tolist()))

court_doc = [{'citation':citation, 'text':text} for citation,text in zip(court_consideration_df['citation'].tolist(), court_consideration_df['text'].tolist())]

print("Models loaded.")


recall_citation_l_l_bm25 = []
recall_cc_l_l_bm25 = []
gold_citation_l_l = []

def extract_all_citation_from_cc_l(cc_l):
    ret_set = set()
    for cc in cc_l:
        cl = citation_utils.extract_citations_from_text(cc)
        for c in cl:
            ret_set.add(c)
    return list(ret_set)

valid_df = pd.read_csv("../data/valid_rewrite_001.csv")
valid_gold_d = {}
for query_id, gold_citations in zip(valid_df['query_id'], valid_df['gold_citations']):
    valid_gold_d[query_id] = gold_citations.split(";")

valid_df = pd.read_csv("../data/valid_rewrite_003_10v.csv")
valid_d = defaultdict(list)
for query_id, query2 in zip(valid_df['query_id'], valid_df['query2']):
    valid_d[query_id].append(query2)

# optional: create a stemmer
stemmer = Stemmer.Stemmer("german")
retriever = bm25s.BM25.load("../data/bm25/cc", load_corpus=True)

for query_id, query_list in valid_d.items():
    gold_citations = valid_gold_d[query_id]
    gold_citation_l_l.append(gold_citations)
    
    cid_set = set()
    ccid_set = set()
    for query in query_list:
        query_tokens = bm25s.tokenize(query, stemmer=stemmer)
        results, scores = retriever.retrieve(query_tokens, k=200)
        
        for i in range(results.shape[1]):
            idx, score = results[0, i], scores[0, i]
            # print(f"Rank {i+1} (score: {score:.2f}): {doc}")
            citation = court_doc[idx]['citation']
            ccid_set.add(citation)
            text = court_doc[idx]['text']
            cids = citation_utils.extract_citations_from_text(text)
            for cid in cids:
                cid_set.add(cid)
                
    recall_citation_l_l_bm25.append(list(cid_set))
    recall_cc_l_l_bm25.append(list(ccid_set))

recall_bm25 = metric_utils.cal_recall(recall_citation_l_l_bm25, gold_citation_l_l)

print("recall_bm25:", recall_bm25)

for l in recall_cc_l_l_bm25:
    print("l.len:", len(l))



