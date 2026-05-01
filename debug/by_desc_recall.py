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

# -- load index and reranker --
model = BGEM3FlagModel('/root/.cache/modelscope/hub/models/BAAI/bge-m3', use_fp16=True, show_progress_bar=False)
dense_index = DenseIndex(model, "/root/autodl-fs/bge-processed/_dense_sparse_court/", court_doc)
sparse_index = SparseIndex(model, "/root/autodl-fs/bge-processed/_dense_sparse_court/", court_doc)
reranker = FlagReranker('/root/.cache/modelscope/hub/models/BAAI/bge-reranker-v2-m3', use_fp16=True, normalize=True)

print("Models loaded.")


recall_citation_l_l_direct = []
recall_citation_l_l_2stage = []
recall_citation_l_l_bm25 = []
gold_citation_l_l = []

def extract_all_citation_from_cc_l(cc_l):
    ret_set = set()
    for cc in cc_l:
        cl = citation_utils.extract_citations_from_text(cc)
        for c in cl:
            ret_set.add(c)
    return list(ret_set)

valid_df = pd.read_csv("../data/valid_rewrite_001.csv")

valid_multiq_df = pd.read_csv("../data/valid_rewrite_split_question_001.csv")
valid_multiq_d = defaultdict(list)
for query_id, query in zip(valid_multiq_df['query_id'], valid_multiq_df['query']):
    valid_multiq_d[query_id].append(query)

# optional: create a stemmer
stemmer = Stemmer.Stemmer("german")
retriever = bm25s.BM25.load("../data/bm25/cc", load_corpus=True)

for query_id, query, gold_citations in zip(valid_df['query_id'], valid_df['query2'], valid_df['gold_citations']):
    multiq = valid_multiq_d[query_id]

    desc = multiq[0].rsplit('\n\n', 1)[0]
    
    gold_citation_l_l.append(gold_citations.split(';'))
    
    l = [cc['text'] for cc,_ in dense_index.search_with_score(desc, top_k=500)]

    recall_citation_l_l_direct.append(extract_all_citation_from_cc_l(l))


recall_direct_500 = metric_utils.cal_recall(recall_citation_l_l_direct, gold_citation_l_l)
# recall_2stage_500 = metric_utils.cal_recall(recall_citation_l_l_2stage, gold_citation_l_l)
# recall_bm25_500 = metric_utils.cal_recall(recall_citation_l_l_bm25, gold_citation_l_l)

print("recall_direct_500:", recall_direct_500)
# print("recall_2stage_500:", recall_2stage_500)
# print("recall_bm25_500:", recall_bm25_500)



