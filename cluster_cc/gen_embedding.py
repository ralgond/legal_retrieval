from collections import Counter, defaultdict
import os
import os.path
import sys
import pickle
import pandas as pd
import numpy as np

src_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

import citation_utils
from embedding_utils import BGEEmbedder

embedder = BGEEmbedder('/root/.cache/modelscope/hub/models/BAAI/bge-m3')

valid_emb_d = {}
valid_df = pd.read_csv("../data/valid_rewrite_001.csv")
for query_id, query_text, gold_citations in zip(valid_df['query_id'], valid_df['query2'], valid_df['gold_citations']):
    valid_emb_d[query_id] = embedder.encode(query_text).astype(np.float32)
with open("../data/valid_rewrite_001_emb.pkl", "wb+") as of:
    pickle.dump(valid_emb_d, of)

test_emb_d = {}
test_df = pd.read_csv("../data/test_rewrite_001.csv")
for query_id, query_text in zip(test_df['query_id'], test_df['query']):
    test_emb_d[query_id] = embedder.encode(query_text).astype(np.float32)
with open("../data/test_rewrite_001_emb.pkl", "wb+") as of:
    pickle.dump(test_emb_d, of)

train_emb_d = {}
train_df = pd.read_csv("../data/train_rewrite_001.csv")
for query_id, query_text, gold_citations in zip(train_df['query_id'], train_df['query2'], train_df['gold_citations']):
    train_emb_d[query_id] = embedder.encode(query_text).astype(np.float32)
with open("../data/train_rewrite_001_emb.pkl", "wb+") as of:
    pickle.dump(train_emb_d, of)