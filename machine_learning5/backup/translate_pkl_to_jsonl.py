
import math
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import random
import json

court_consideration_df = pd.read_csv("../data/court_considerations.csv")
court_consideration_d = dict(zip(court_consideration_df['citation'], court_consideration_df['text']))
print("data loaded.")

import common
train_candidate_d = common.read_candidate("../data/ml5/raw_train_candidate.pkl", court_consideration_d)
valid_candidate_d = common.read_candidate("../data/ml5/raw_valid_candidate.pkl", court_consideration_d)
test_candidate_d = common.read_candidate("../data/ml5/raw_test_candidate.pkl", court_consideration_d)

OUTPUT = "../data/ml5"

def gen_train_jsonl():
    train_cc_dense_score_d = {}
    train_cc_sparse_score_d = {}
    train_cc_rerank_score_d = {}

    for train_query_id, d in train_candidate_d.items():
        dense_l = d['dense']
        for cc, dense_score in dense_l:
            train_cc_dense_score_d[cc['citation']] = dense_score
        sparse_l = d['sparse']
        for cc, sparse_score in sparse_l:
            train_cc_sparse_score_d[cc['citation']] = sparse_score
        rerank_l = d['rerank']
        for cc, rerank_score in rerank_l:
            train_cc_rerank_score_d[cc['citation']] = rerank_score

    train_query_id_2_gold_citation_d = {}
    train_df = pd.read_csv("../data/train_rewrite_001.csv")
    for query_id, gold_citations in zip(train_df['query_id'], train_df['gold_citations']):
        train_query_id_2_gold_citation_d[query_id] = gold_citations.split(';')

    l = []
    for train_query_id, d_raw in train_candidate_d.items():
        d = {}
        d['query_id'] = train_query_id
        d['gold_citations'] = train_query_id_2_gold_citation_d[train_query_id]
        cc_list = []
        rerank_list = d_raw['rerank']
        for cc,_ in rerank_list:
            d2 = {}
            d2['cc_id'] = cc['citation']
            d2['text'] = cc['text']
            d2['dense_score'] = float(train_cc_dense_score_d.get(cc['citation'], 0.))
            d2['sparse_score'] = float(train_cc_sparse_score_d.get(cc['citation'], 0.))
            d2['rerank_score'] = float(train_cc_rerank_score_d.get(cc['citation'], 0.))
            cc_list.append(d2)
        d['cc_list'] = cc_list
        l.append(d)
    return l

l = gen_train_jsonl()
with open(f"{OUTPUT}/train.jsonl", "w+", encoding="utf-8") as of:
    for j in l:
        # print(j)
        of.write(json.dumps(j, ensure_ascii=False) + '\n')


def gen_valid_jsonl():
    valid_cc_dense_score_d = {}
    valid_cc_sparse_score_d = {}
    valid_cc_rerank_score_d = {}

    for valid_query_id, d in valid_candidate_d.items():
        dense_l = d['dense']
        for cc, dense_score in dense_l:
            valid_cc_dense_score_d[cc['citation']] = dense_score
        sparse_l = d['sparse']
        for cc, sparse_score in sparse_l:
            valid_cc_sparse_score_d[cc['citation']] = sparse_score
        rerank_l = d['rerank']
        for cc, rerank_score in rerank_l:
            valid_cc_rerank_score_d[cc['citation']] = rerank_score

    valid_query_id_2_gold_citation_d = {}
    valid_df = pd.read_csv("../data/valid_rewrite_001.csv")
    for query_id, gold_citations in zip(valid_df['query_id'], valid_df['gold_citations']):
        valid_query_id_2_gold_citation_d[query_id] = gold_citations.split(';')

    l = []
    for valid_query_id, d_raw in valid_candidate_d.items():
        d = {}
        d['query_id'] = valid_query_id
        d['gold_citations'] = valid_query_id_2_gold_citation_d[valid_query_id]
        cc_list = []
        rerank_list = d_raw['rerank']
        for cc,_ in rerank_list:
            d2 = {}
            d2['cc_id'] = cc['citation']
            d2['text'] = cc['text']
            d2['dense_score'] = float(valid_cc_dense_score_d.get(cc['citation'], 0.))
            d2['sparse_score'] = float(valid_cc_sparse_score_d.get(cc['citation'], 0.))
            d2['rerank_score'] = float(valid_cc_rerank_score_d.get(cc['citation'], 0.))
            cc_list.append(d2)
        d['cc_list'] = cc_list
        l.append(d)
    return l

l = gen_valid_jsonl()
with open(f"{OUTPUT}/valid.jsonl", "w+", encoding="utf-8") as of:
    for j in l:
        # print(j)
        of.write(json.dumps(j, ensure_ascii=False) + '\n')


def gen_test_jsonl():
    test_cc_dense_score_d = {}
    test_cc_sparse_score_d = {}
    test_cc_rerank_score_d = {}

    for test_query_id, d in test_candidate_d.items():
        dense_l = d['dense']
        for cc, dense_score in dense_l:
            test_cc_dense_score_d[cc['citation']] = dense_score
        sparse_l = d['sparse']
        for cc, sparse_score in sparse_l:
            test_cc_sparse_score_d[cc['citation']] = sparse_score
        rerank_l = d['rerank']
        for cc, rerank_score in rerank_l:
            test_cc_rerank_score_d[cc['citation']] = rerank_score

    l = []
    for test_query_id, d_raw in test_candidate_d.items():
        d = {}
        d['query_id'] = test_query_id
        # d['gold_citations'] = test_query_id_2_gold_citation_d[test_query_id]
        cc_list = []
        rerank_list = d_raw['rerank']
        for cc,_ in rerank_list:
            d2 = {}
            d2['cc_id'] = cc['citation']
            d2['text'] = cc['text']
            d2['dense_score'] = float(test_cc_dense_score_d.get(cc['citation'], 0.))
            d2['sparse_score'] = float(test_cc_sparse_score_d.get(cc['citation'], 0.))
            d2['rerank_score'] = float(test_cc_rerank_score_d.get(cc['citation'], 0.))
            cc_list.append(d2)
        d['cc_list'] = cc_list
        l.append(d)
    return l

l = gen_test_jsonl()
with open(f"{OUTPUT}/test.jsonl", "w+", encoding="utf-8") as of:
    for j in l:
        # print(j)
        of.write(json.dumps(j, ensure_ascii=False) + '\n')