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

# optional: create a stemmer
stemmer = Stemmer.Stemmer("german")
retriever = bm25s.BM25.load("../data/bm25/cc", load_corpus=True)

for query_id, query, gold_citations in zip(valid_df['query_id'], valid_df['query2'], valid_df['gold_citations']):
    gold_citation_l_l.append(gold_citations.split(';'))
    # direct_500 = [cc['text'] for cc,_ in dense_index.search_with_score(query, top_k=400)]
    # direct_500_2 = [cc['text'] for cc,_ in sparse_index.search_with_score(query, top_k=100)]
    # direct_500.extend(direct_500_2)
    # recall_citation_l_l_direct.append(extract_all_citation_from_cc_l(direct_500))

    two_step_500 = []
    seed_50 = [cc['text'] for cc,_ in dense_index.search_with_score(query, top_k=400)]
    for _q in seed_50:
        two_step_500.extend([cc['text'] for cc,_ in sparse_index.search_with_score(_q, top_k=100)])
    two_step_500.extend(seed_50)
    recall_citation_l_l_2stage.append(extract_all_citation_from_cc_l(two_step_500))

    # cc_l_5_x_100 = []
    # seed_100 = [cc for cc,_ in dense_index.search_with_score(query, top_k=100)]
    # seed_5 = sorted(reranker_utils.rerank_by_batch_chunked2(reranker, query, seed_100), key=lambda x:x[1], reverse=True)[:10]
    # for cc,_ in seed_5:
    #     l = [c0['text'] for c0,_ in sparse_index.search_with_score(cc['text'], 50)]
    #     # l = [c0['text'] for c0,_ in dense_index.search_with_score(cc['text'], 100)]
    #     cc_l_5_x_100.extend(l)
    # recall_citation_l_l_2stage.append(extract_all_citation_from_cc_l(cc_l_5_x_100))

    # query_tokens = bm25s.tokenize(query, stemmer=stemmer)
    # results, scores = retriever.retrieve(query_tokens, k=100)
    # cid_set = set()
    # for i in range(results.shape[1]):
    #     idx, score = results[0, i], scores[0, i]
    #     # print(f"Rank {i+1} (score: {score:.2f}): {doc}")
    #     text = court_doc[idx]['text']
    #     cids = citation_utils.extract_citations_from_text(text)
    #     for cid in cids:
    #         cid_set.add(cid)
    # recall_citation_l_l_bm25.append(list(cid_set))

    # recall_citation_l_l_direct[-1].extend(list(cid_set))

    # # recall_citation_l_l_direct[-1].extend(extract_all_citation_from_cc_l(cc_l_5_x_100))
    


# recall_direct_500 = metric_utils.cal_recall(recall_citation_l_l_direct, gold_citation_l_l)
recall_2stage_500 = metric_utils.cal_recall(recall_citation_l_l_2stage, gold_citation_l_l)
# recall_bm25_500 = metric_utils.cal_recall(recall_citation_l_l_bm25, gold_citation_l_l)

# print("recall_direct_500:", recall_direct_500)
print("recall_2stage_500:", recall_2stage_500)
# print("recall_bm25_500:", recall_bm25_500)



