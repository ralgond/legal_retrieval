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

stemmer = Stemmer.Stemmer("german")

def tokenize_and_stem(text: str) -> list[str]:
    """
    分词 + 词干提取
    简单按空格/标点 split，然后对每个 token 做 stem
    """
    # 基础清洗：转小写，去标点
    import re
    text = text.lower()
    tokens = re.split(r"[\s\,\.\;\:\!\?\(\)\-\/]+", text)
    tokens = [t for t in tokens if len(t) > 1]

    # 词干提取
    stemmed = stemmer.stemWords(tokens)
    return stemmed
    
def build_synonym_dict(xlsx_path):
    """从同义词表xlsx加载"""
    import pandas as pd
    synonym_dict = {}
    xl = pd.ExcelFile(xlsx_path)
    for sheet in xl.sheet_names:
        if sheet == "使用说明":
            continue
        df = xl.parse(sheet)
        for _, row in df.iterrows():
            main_term = str(row.iloc[0]).strip().lower()
            synonyms_raw = str(row.iloc[1])  # 同义词列
            related_raw  = str(row.iloc[2])  # 相关词列
            syns = [s.strip().lower() for s in synonyms_raw.split(";") if s.strip()]
            rels = [s.strip().lower() for s in related_raw.split(";") if s.strip()]
            all_terms = list(set([main_term] + syns + rels))
            # 双向映射：任何一个词都能找到整组
            for t in all_terms:
                synonym_dict[t] = all_terms
    return synonym_dict

# ── 3. 查询展开：stem → 找同义 → 合并 → 拼回句子 ──────────────
def expand_query(
    query: str,
    synonym_dict: dict,
    repeat_original: int = 1,
) -> tuple[list[str], str]:
    """
    Returns:
        expanded_tokens : 展开后 token 列表（用于 BM25）
        expanded_sentence: 拼成一个字符串（方便查看 / 传给其他系统）
    """
    # Step 1: 分词 + stem
    stemmed_tokens = tokenize_and_stem(query)

    # Step 2: 同义词展开
    expanded = []
    seen = set()

    for token in stemmed_tokens:
        # 原词重复 N 次 → BM25 TF 加权
        for _ in range(repeat_original):
            expanded.append(token)
        seen.add(token)

        # 加入同义词（不重复原词）
        if token in synonym_dict:
            for syn in synonym_dict[token]:
                if syn not in seen:
                    expanded.append(syn)
                    seen.add(syn)

    # print("====>", seen)
    
    # Step 3: 拼成句子
    expanded_sentence = " ".join(expanded)

    return expanded, expanded_sentence
    

def extract_all_citation_from_cc_l(cc_l):
    ret_set = set()
    for cc in cc_l:
        cl = citation_utils.extract_citations_from_text(cc)
        for c in cl:
            ret_set.add(c)
    return list(ret_set)

valid_df = pd.read_csv("../data/valid_rewrite_001.csv")
valid_gold_d = {}
valid_query_d = {}
for query_id, query, gold_citations in zip(valid_df['query_id'], valid_df['query2'], valid_df['gold_citations']):
    valid_gold_d[query_id] = gold_citations.split(";")
    valid_query_d[query_id] = query

# optional: create a stemmer

retriever = bm25s.BM25.load("../data/bm25/cc", load_corpus=True)
synonym_dict = build_synonym_dict("../data/Schweizer_Rechtsterminologie_Synonymwoerterbuch.xlsx")

for query_id, query in valid_query_d.items():
    gold_citations = valid_gold_d[query_id]
    gold_citation_l_l.append(gold_citations)
    
    cid_set = set()
    expanded_tokens, expanded_sentence = expand_query(query, synonym_dict)

    # print(expanded_sentence)

    # os._exit(0)

    query_tokens = bm25s.tokenize(expanded_sentence, stemmer=stemmer)
    # print("type(query_tokens):", type(query_tokens), query_tokens)
    # expanded_tokens = expand_query(query_tokens, synonym_dict)
    # expaened_query = ' '.join(expanded_tokens)
    # query_tokens = bm25s.tokenize(expaened_query, stemmer=stemmer)
    
    results, scores = retriever.retrieve(query_tokens, k=500)
    
    for i in range(results.shape[1]):
        idx, score = results[0, i], scores[0, i]
        # print(f"Rank {i+1} (score: {score:.2f}): {doc}")
        citation = court_doc[idx]['citation']
        text = court_doc[idx]['text']
        cids = citation_utils.extract_citations_from_text(text)
        for cid in cids:
            cid_set.add(cid)
            
    recall_citation_l_l_bm25.append(list(cid_set))
    
recall_bm25 = metric_utils.cal_recall(recall_citation_l_l_bm25, gold_citation_l_l)

print("recall_bm25:", recall_bm25)



