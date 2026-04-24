import re
import numpy as np
import lightgbm as lgb
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

import math
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import random

src_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)


# =========================
# 1. 模型
# =========================
import embedding_utils

embedder = embedding_utils.BGEEmbedder("BAAI/bge-m3")

# =========================
# 2. 工具函数
# =========================
import citation_utils

def split_sentences(text: str):
    return citation_utils.split_sentences(text)

def get_window(text: str, citation: str, window_size=2):
    sentences = split_sentences(text)

    for i, sent in enumerate(sentences):
        if citation in sent:
            left = sentences[max(0, i - window_size):i]
            right = sentences[i+1:i+1+window_size]
            return " ".join(left + [sent] + right)

    return ""

# =========================
# 3. 特征工程
# =========================

class FeatureExtractor:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=5000)

    def fit_tfidf(self, corpus: List[str]):
        self.tfidf.fit(corpus)

    def tfidf_overlap(self, q: str, doc: str):
        q_vec = self.tfidf.transform([q])
        d_vec = self.tfidf.transform([doc])
        return (q_vec.multiply(d_vec)).sum()

    def dense_sim(self, q_emb, d_emb):
        return np.dot(q_emb, d_emb) / (
            np.linalg.norm(q_emb) * np.linalg.norm(d_emb) + 1e-6
        )

    def extract(self, query, window, q_emb, w_emb, cc_features):
        feats = {}

        # 1. dense
        feats["dense_sim"] = self.dense_sim(q_emb, w_emb)

        # 2. tfidf
        feats["tfidf_overlap"] = self.tfidf_overlap(query, window)

        # 3. 长度
        feats["len"] = len(window)

        # 4. trigger words
        feats["has_gemaess"] = int("gemäss" in window.lower())
        feats["has_vgl"] = int("vgl" in window.lower())

        # 5. cc-level features
        feats.update(cc_features)

        return feats

# =========================
# 4. 构造训练数据
# =========================

def build_training_data(data: List[Dict]):
    X = []
    y = []
    group = []

    all_texts = []

    # 收集 corpus（给 tfidf）
    for item in data:
        for cc in item["cc_list"]:
            all_texts.append(cc["text"])

    extractor = FeatureExtractor()
    extractor.fit_tfidf(all_texts)

    for item in data:
        query = item["query"]
        q_emb = embedder.encode(query)

        group_size = 0

        for cc in item["cc_list"]:
            text = cc["text"]
            is_pos_cc = cc["is_positive"]

            # 你已有的 cc-level 分数
            cc_features = {
                "cc_dense": cc.get("dense_score", 0),
                "cc_sparse": cc.get("sparse_score", 0),
                "cc_rerank": cc.get("rerank_score", 0),
            }

            for citation in cc["citations"]:
                window = get_window(text, citation)

                if not window:
                    continue

                w_emb = embedder.encode(window)

                feats = extractor.extract(
                    query, window, q_emb, w_emb, cc_features
                )

                X.append(list(feats.values()))

                # ====== label 构造 ======
                if is_pos_cc:
                    if citation == cc.get("gold_citation"):
                        label = 1
                    else:
                        label = 0
                else:
                    label = 0

                y.append(label)
                group_size += 1

        if group_size > 0:
            group.append(group_size)

    return np.array(X), np.array(y), group

# =========================
# 5. 训练
# =========================

def train_ltr_with_early_stop(
    X_train, y_train, group_train,
    X_valid, y_valid, group_valid
):
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=2000,          # ⚠️ 给大一点，让 early stop 来控制
        learning_rate=0.05,
        num_leaves=31
    )

    model.fit(
        X_train, y_train,
        group=group_train,

        eval_set=[(X_valid, y_valid)],
        eval_group=[group_valid],

        eval_at=[1, 3, 5],         # NDCG@k

        early_stopping_rounds=50,  # ⭐ 关键
        verbose=10
    )

    return model

# =========================
# 6. 推理（给一个query + cc）
# =========================

def rank_citations(model, extractor, query, cc):
    q_emb = embedder.encode(query)

    scores = []

    for citation in cc["citations"]:
        window = get_window(cc["text"], citation)
        if not window:
            continue

        w_emb = embedder.encode(window)

        feats = extractor.extract(
            query, window, q_emb, w_emb,
            {
                "cc_dense": cc.get("dense_score", 0),
                "cc_sparse": cc.get("sparse_score", 0),
                "cc_rerank": cc.get("rerank_score", 0),
            }
        )

        score = model.predict([list(feats.values())])[0]
        scores.append((citation, score))

    return sorted(scores, key=lambda x: -x[1])