"""
build_dataset.py
构造 LightGBM Learning-to-Rank candidate list。

输出：
../data/ml3/raw_train_candidate.pkl
../data/ml3/raw_valid_candidate.pkl
../data/ml3/raw_test_candidate.pkl

三个文件中都包含下面数据结构：
(query_id, List[(cc_id, dense_score), ...], List[(cc_id, sparse_score), ...], List[(cc_id, rerank_score), ...])

有了这三个文件，即可在没有GPU的环境下也能调试算法
"""

from __future__ import annotations
import math
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

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

retrieve_top_k = 100

train_df = pd.read_csv("../data/train_rewrite_001.csv")
valid_df = pd.read_csv("../data/valid_rewrite_001.csv")
test_df = pd.read_csv("../data/test_rewrite_001.csv")

def generate_dataset(df, query_col_name):
    count=0
    _l = []
    for query_id, query in tqdm(zip(df['query_id'].tolist(), df[query_col_name].tolist()), total=len(df)):
        hits1 = dense_index.search_with_score(query, top_k=retrieve_top_k)
        hits2 = sparse_index.search_with_score(query, top_k=retrieve_top_k)
     
        _hits = hits_utils.merge_hits_with_score_l_by_max(hits1, hits2)
        hit_with_score_l = reranker_utils.rerank_by_batch_chunked2(reranker, query, [hit for hit, _ in _hits])

        hits3 = hit_with_score_l
    
        hits1_strip_text = [({'citation':hit['citation']}, dense_score) for hit, dense_score in hits1] # 不需要保存text
        hits2_strip_text = [({'citation':hit['citation']}, sparse_score) for hit, sparse_score in hits2]
        hits3_strip_text = [({'citation':hit['citation']}, rerank_score) for hit, rerank_score in hits3]
    
        term = (query_id, hits1_strip_text, hits2_strip_text, hits3_strip_text)
        _l.append(term)

        count += 1
        if count >= 100:
            pass
            
    return _l

valid_l = generate_dataset(valid_df, 'query2')
with open("../data/ml3/raw_valid_candidate.pkl", "wb+") as of:
    pickle.dump(valid_l, of)
    
train_l = generate_dataset(train_df, 'query2')
with open("../data/ml3/raw_train_candidate.pkl", "wb+") as of:
    pickle.dump(train_l, of)
    
test_l = generate_dataset(test_df, 'query')
with open("../data/ml3/raw_test_candidate.pkl", "wb+") as of:
    pickle.dump(test_l, of)