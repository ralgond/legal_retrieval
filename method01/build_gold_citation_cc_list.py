from __future__ import annotations
import math
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import random
from tqdm import tqdm
import pickle

src_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

import citation_utils

OUTPUT_DIR = "../data/method01/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

cc_df = pd.read_csv("../data/court_considerations.csv")
cc_d = {citation:text for citation,text in zip(cc_df['citation'], cc_df['text'])}
print("cc_df loaded.")

cid_2_ccid_d = {}

# 处理所有的cc
if os.path.exists(f"{OUTPUT_DIR}/cid_2_ccid_d.pkl"):
    with open(f"{OUTPUT_DIR}/cid_2_ccid_d.pkl", "rb") as inf:
        cid_2_ccid_d = pickle.load(inf)
else:
    for ccid, text in tqdm(zip(cc_df['citation'], cc_df['text']), total=len(cc_df)):
        cid_l = citation_utils.extract_citations_from_text(text)
        for cid in cid_l:
            if cid not in cid_2_ccid_d:
                cid_2_ccid_d[cid] = set()
            cid_2_ccid_d[cid].add(ccid)

    with open(f"{OUTPUT_DIR}/cid_2_ccid_d.pkl", "wb+") as of:
        pickle.dump(cid_2_ccid_d, of)


# -------------------------
# 2. 构建 BM25 检索器
# -------------------------
from rank_bm25 import BM25Okapi

import spacy
nlp = spacy.load("de_core_news_lg")

def tokenize(text):
    return [t.lemma_ for t in nlp(text) if not t.is_stop and not t.is_punct]

class BM25Retriever:
    def __init__(self, texts):
        """
        texts: List[str]
        """
        self.texts = texts
        self.tokenized_texts = [tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.tokenized_texts)

    def search(self, query, top_k=200):
        """
        return top-k most relevant texts
        """
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 排序
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = [
            {
                "text": self.texts[i],
                "score": float(scores[i])
            }
            for i in top_idx
        ]

        return results


class Query:
    def __init__(self, id, query, gold_citation):
        self.id = id
        self.query = query
        self.gold_citation = gold_citation
        self.ccid_set = set()
        self.retrieval = None # BM25Retriever()
    
    def add(self, ccid):
        self.ccid_set.add(ccid)

    def truncate(self, cc_d):
        cc_text_l = []
        for ccid in self.ccid_set:
            cc_text_l.append(cc_d.get(ccid, ""))
        self.retrieval = BM25Retriever(cc_text_l)
        
        hits = self.retrieval.search(self.query)

        recall_gold_citation = set()
        for hit in hits:
            text = hit['text']
            for gold in self.gold_citation:
                if gold in text:
                    recall_gold_citation.add(gold)
        print(f"{len(recall_gold_citation)}/{len(self.gold_citation)}")


query_d = {}
train_df = pd.read_csv("../data/train_rewrite_001.csv")
total_cc = 0
for query_id, query, gold_citations in zip(train_df['query_id'], train_df['query'], train_df['gold_citations']):
    q = Query(query_id, query, set(list(gold_citations.split(";"))))
    ccid_set = set()
    gold_l = gold_citations.split(';')
    for gold_cid in gold_l:
        ccid_set = cid_2_ccid_d.get(gold_cid, set())
        for ccid in ccid_set:
            q.add(ccid)
    print(f"{q.id}: {len(q.ccid_set)}")
    total_cc += len(q.ccid_set)

    query_d[query_id] = q
    q.truncate(cc_d)

print(total_cc)





