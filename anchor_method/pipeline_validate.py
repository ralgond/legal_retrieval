"""
inference_lgbm.py
用训练好的 LightGBM 模型对 valid 集推理并评估 Recall / Precision。
"""

from __future__ import annotations
import math
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import defaultdict

src_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

import reranker_utils
import hits_utils
import citation_utils
import metric_utils

# 构建搜索引擎
from legal_search import SearchResult, build_index, LegalSearchEngine

core_sent_df = pd.read_csv("core_sentence.csv")
core_sentences = []
for citation, text in zip(core_sent_df['citation'], core_sent_df['text']):
    core_sentences.append((citation, text))
build_index(use_core_sentences=core_sentences)

legal_search_engine = LegalSearchEngine()

# ── 加载检索模型 ──────────────────────────────────────────────────────────────
court_consideration_df = pd.read_csv("../data/court_considerations.csv")
court_consideration_d = dict(zip(court_consideration_df['citation'].tolist(), court_consideration_df['text'].tolist()))

print("Loaded cc...")

import common
valid_candidate_d = common.read_candidate("../data/ml3/raw_valid_candidate.pkl", court_consideration_d)


# ── 推理主循环 ────────────────────────────────────────────────────────────────
valid_df  = pd.read_csv("../data/valid_rewrite_001.csv")
result_l  = []
gold_l    = []

def text_2_sentences_which_contain_citation(query_id):
    '''
    分析所有的cc，返回所有带有citation的句子
    '''
    cc_l = valid_candidate_d[query_id]['rerank']

    ret = []
    for cc in cc_l:
        sents = citation_utils.split_sentences(cc['text'])
        for sent in sents:
            cids = citation_utils.extract_citations_from_text(sent)
            if len(cids) > 0:
                ret.append(sent)
    return ret


result_l = []
gold_l = []

threshold_score=0.8

for i, (query_id, query, gold_citations) in enumerate(
        zip(valid_df['query_id'], valid_df['query2'], valid_df['gold_citations'])):

    gold_set = set(gold_citations.split(';'))
    gold_l.append(list(gold_set))
    print(f"[{i+1}/{len(valid_df)}] query_id={query_id}")

    sents = text_2_sentences_which_contain_citation(query_id)

    cid_l = []
    for sent in sents:
        results = legal_search_engine.search(sent, top_k=3)
        max_score = results[0].score
        if max_score> threshold_score:
            cids = citation_utils.extract_citations_from_text(sent)
            cid_l.extend(cids)
    
    cid_l = list(set(cid_l))
    result_l.append(cid_l)

# ── 评估 ──────────────────────────────────────────────────────────────────────
recall    = metric_utils.cal_recall(result_l, gold_l)
precision = metric_utils.cal_precision(result_l, gold_l)
print(f"[{threshold_score}] Recall:{recall:.4f}, Precision:{precision:.4f}, F1:{2*recall*precision/(recall+precision):.4f}")
