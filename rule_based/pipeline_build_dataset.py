from __future__ import annotations
import math
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import random
from tqdm import tqdm

src_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

import citation_utils

DATA_DIR = "../data/rule_based"
OUTPUT_DIR = "../data/rule_based"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 加载数据 ──────────────────────────────────────────────────────────────────
print("Loading data...")
court_consideration_df = pd.read_csv("../data/court_considerations_maped.csv")
court_consideration_d = dict(zip(court_consideration_df['citation'].tolist(), court_consideration_df['text'].tolist()))

import common
train_candidate_d = common.read_candidate(f"{DATA_DIR}/raw_train_candidate.pkl", court_consideration_d)
valid_candidate_d = common.read_candidate(f"{DATA_DIR}/raw_valid_candidate.pkl", court_consideration_d)
test_candidate_d = common.read_candidate(f"{DATA_DIR}/raw_test_candidate.pkl", court_consideration_d)

train_gold_d = dict()
train_gold_df = pd.read_csv("../data/train_rewrite_001.csv")
for query_id, gold_citations in zip(train_gold_df['query_id'], train_gold_df['gold_citations']):
    train_gold_d[query_id] = set(gold_citations.split(';'))

valid_gold_d = dict()
valid_gold_df = pd.read_csv("../data/valid_rewrite_001.csv")
for query_id, gold_citations in zip(valid_gold_df['query_id'], valid_gold_df['gold_citations']):
    valid_gold_d[query_id] = set(gold_citations.split(';'))


import re
import numpy as np
import pandas as pd
from typing import List, Dict

# =========================
# 1. 关键词表
# =========================

POS_STRONG = [
    "gemäss", "nach", "laut", "gestützt auf",
    "in anwendung von", "im sinne von", "aufgrund von"
]

POS_MED = [
    "vgl.", "vgl. auch", "siehe", "siehe auch"
]

POS_CONCLUSION = [
    "entscheidend ist", "massgebend ist",
    "ist zu prüfen", "ist zu entscheiden",
    "ergibt sich aus", "folgt aus"
]

NEG_STRONG = [
    "kann offen bleiben",
    "nicht entscheidend",
    "braucht nicht entschieden"
]

NEG_FACT = [
    "macht geltend",
    "stellte fest",
    "wird ausgeführt",
    "laut den angaben"
]

NEG_WEAK = [
    "unter anderem", "beispielsweise", "z.b.", "insbesondere"
]

# =========================
# 2. citation regex
# =========================

# CITATION_PATTERN = re.compile(
#     r"(Art\.?\s?\d+[a-zA-Z]*\s?(Abs\.?\s?\d+)?\s?(lit\.?\s?[a-z])?)"
# )

# =========================
# 3. spaCy 加载（带 fallback）
# =========================

import spacy
nlp = spacy.load("de_core_news_lg")
USE_SPACY = True

# =========================
# 4. 工具函数
# =========================

def normalize_text(text: str) -> str:
    return text.lower()


def count_keywords(text: str, keywords: List[str]) -> int:
    text = normalize_text(text)
    return sum(1 for kw in keywords if kw in text)

# =========================
# 5. 句子类型判断
# =========================

def sentence_type_features(sentence: str):
    s = normalize_text(sentence)

    return {
        "is_conclusion_sentence": int(any(k in s for k in POS_CONCLUSION)),
        "is_fact_sentence": int(any(k in s for k in NEG_FACT)),
        "has_strong_positive": int(any(k in s for k in POS_STRONG)),
    }


# =========================
# 6. 句法特征（可选）
# =========================

def dependency_features(sentence: str, citation_text: str):
    if not USE_SPACY:
        return {
            "is_prep_object": 0,
            "is_subject": 0
        }

    doc = nlp(sentence)

    for token in doc:
        if citation_text in token.text:
            return {
                "is_prep_object": int(token.dep_ == "pobj"),
                "is_subject": int(token.dep_ in ["nsubj", "nsubjpass"])
            }

    return {
        "is_prep_object": 0,
        "is_subject": 0
    }


# =========================
# 7. 主函数：extract features
# =========================

def extract_features_from_text(text: str) -> pd.DataFrame:

    sentences = citation_utils.p_split_sentences(text)

    total_sentences = len(sentences)

    all_rows = []

    for sent_idx, sent in enumerate(sentences):

        citations = citation_utils.extract_pcitations_from_text_with_span(sent)

        # print(citations)

        if len(citations) == 0:
            continue

        sent_len = len(sent)

        # sentence-level features
        sent_feats = sentence_type_features(sent)

        pos_score = (
            2 * count_keywords(sent, POS_STRONG)
            + 1 * count_keywords(sent, POS_MED)
            + 3 * count_keywords(sent, POS_CONCLUSION)
        )

        neg_score = (
            3 * count_keywords(sent, NEG_STRONG)
            + 2 * count_keywords(sent, NEG_FACT)
            + 1 * count_keywords(sent, NEG_WEAK)
        )

        for cit_text, start, end in citations:

            char_pos_norm = start / max(sent_len, 1)

            dep_feats = dependency_features(sent, cit_text)

            row = {
                "citation": cit_text,

                # keyword signals
                "pos_keyword_score": pos_score,
                "neg_keyword_score": neg_score,

                # sentence type
                **sent_feats,

                # position
                "sentence_idx": sent_idx,
                "pos_norm": sent_idx / max(total_sentences, 1),
                "char_pos_norm": char_pos_norm,

                # structure
                **dep_feats,

                # density
                "num_citations_in_sentence": len(citations),
                "is_single_citation": int(len(citations) == 1),

                # length
                "sentence_len": sent_len,
            }

            all_rows.append(row)

    return pd.DataFrame(all_rows)



import pandas as pd

def build_query_dataframe(query_id, cc_text_list, gold_citations):

    dfs = []

    for cc_id, cc_text in enumerate(cc_text_list):
        df_cc = extract_features_from_text(cc_text)

        if len(df_cc) == 0:
            continue

        df_cc["query_id"] = query_id
        df_cc["cc_id"] = cc_id

        dfs.append(df_cc)

    if len(dfs) == 0:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # 标注 label（关键）
    df["label"] = df["citation"].isin(gold_citations).astype(int)

    return df


def build_dataset(data):

    all_df = []

    for qid, item in tqdm(data.items(), total=len(data)):

        df = build_query_dataframe(
            query_id=qid,
            cc_text_list=item["cc_text_list"],
            gold_citations=item["gold_citations"]
        )

        if len(df) > 0:
            all_df.append(df)

    return pd.concat(all_df, ignore_index=True)

# =========================
# 8. 示例
# =========================

def main():

    data = {}
    for query_id, d in valid_candidate_d.items():
        rerank_l = d['rerank']
        item = dict()
        item['gold_citations'] = valid_gold_d[query_id]
        cc_text_list = [hit['text'] for hit, score in rerank_l]
        item['cc_text_list'] = cc_text_list
        data[query_id] = item

    valid_df = build_dataset(data)
    valid_df.to_csv("../data/rule_based/valid_df.csv", index=False)


    data = {}
    for query_id, d in train_candidate_d.items():
        rerank_l = d['rerank']
        item = dict()
        item['gold_citations'] = train_gold_d[query_id]
        cc_text_list = [hit['text'] for hit, score in rerank_l]
        item['cc_text_list'] = cc_text_list
        data[query_id] = item

    train_df = build_dataset(data)
    train_df.to_csv("../data/rule_based/train_df.csv", index=False)

if __name__ == "__main__":
    main()

    # text = """
    # Gemäss Art. 123 Abs. 1 OR ist der Vertrag gültig.
    # Der Beschwerdeführer macht geltend, dass Art. 45 OR verletzt wurde.
    # Dies ergibt sich aus Art. 99 OR.
    # """

    # df = extract_features_from_text(text)

    # print(df)
