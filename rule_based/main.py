import re
import math
import random
from collections import defaultdict
from typing import List, Dict, Tuple
import pandas as pd
import os
import os.path
import sys
import numpy as np

src_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

import reranker_utils
import hits_utils
import citation_utils
import metric_utils


def _maxmin_normalize_hits(hits):
    max_value = hits[0][1]
    min_value = hits[0][1]
    for i in range(1, len(hits)):
        max_value = max(max_value, hits[i][1])
        min_value = min(min_value, hits[i][1])
    span = max_value - min_value

    ret = [[hit.copy(), score] for hit,score in hits]
    for hit in ret:
        hit[1] = (hit[1] - min_value) * 1. / span

    return [(hit,score) for hit,score in ret]

# ----------------------------
# 规则词表（德语法律）
# ----------------------------
EXPLANATION_WORDS = [
    "gemäss", "nach", "im sinne von", "entsprechend",
    "laut", "stützt sich auf", "beruht auf", "folgt aus"
]

CONCLUSION_WORDS = [
    "somit", "daher", "folglich", "demnach",
    "ergibt sich", "ist festzuhalten", "kommt zum schluss"
]

REASONING_WORDS = [
    "prüfen", "beurteilen", "feststellen",
    "würdigen", "anwenden", "auslegen"
]

NEGATION_WORDS = [
    "nicht einschlägig", "irrelevant",
    "nicht anwendbar", "keine anwendung"
]

# ----------------------------
# citation pattern（瑞士法律）
# ----------------------------
CITATION_PATTERN = re.compile(
    r"(Art\.\s*\d+[a-zA-Z]*\s*(Abs\.\s*\d+)?\s*(OR|ZGB|BV)?|BGE\s*\d+\s*[IVX]+\s*\d+)"
)

# ----------------------------
# 工具函数
# ----------------------------
def tokenize(text):
    return text.lower().split()

def find_citations(text):
    matches = list(CITATION_PATTERN.finditer(text))
    result = defaultdict(list)
    for m in matches:
        cit = m.group()
        result[cit].append(m.start())
    return result

def get_window(text, pos, window_size=100):
    start = max(0, pos - window_size)
    end = min(len(text), pos + window_size)
    return text[start:end].lower()

def contains_any(text, keywords):
    return any(k in text for k in keywords)

def is_sentence_start(text, pos):
    return pos < 50 or text[max(0, pos-2):pos] in [". ", "\n"]

# ----------------------------
# Rule-based 打分
# ----------------------------
def score_citation(text: str, citation_spans: Dict[str, List[int]]):
    scores = {}

    for cit, positions in citation_spans.items():
        score = 0

        # Rule 3: 频率
        freq = len(positions)
        score += math.log(1 + freq) * 2

        for pos in positions:
            window = get_window(text, pos)

            if contains_any(window, EXPLANATION_WORDS):
                score += 3

            if contains_any(window, CONCLUSION_WORDS):
                score += 4

            if contains_any(window, REASONING_WORDS):
                score += 2

            if contains_any(window, NEGATION_WORDS):
                score -= 3

            if is_sentence_start(text, pos):
                score += 1.5

        # Rule 7
        score += 1

        scores[cit] = score

    return scores

import common
print("Loading data...")
court_consideration_df = pd.read_csv("../data/court_considerations.csv")
court_consideration_d = dict(zip(court_consideration_df['citation'].tolist(), court_consideration_df['text'].tolist()))

train_candidate_d = common.read_candidate("../data/rule_based/raw_train_candidate.pkl", court_consideration_d)
valid_candidate_d = common.read_candidate("../data/rule_based/raw_valid_candidate.pkl", court_consideration_d)
test_candidate_d = common.read_candidate("../data/rule_based/raw_test_candidate.pkl", court_consideration_d)

# ===================================================================================
# query -> GT citations
# query_gt = {
#     "query1": ["Art. 17 OR", "BGE 123 III 456"],
# }

train_query_gt = {}
train_df = pd.read_csv("../data/train_rewrite_001.csv")
for query_id, gold_citations in zip(train_df['query_id'], train_df['gold_citations']):
    train_query_gt[query_id] = gold_citations.split(";")

# query -> retrieved CC list
# query_cc = {
#     "query1": [
#         ".... CC text 1 ....",
#         ".... CC text 2 ...."
#     ]
# }

train_query_cc = {}
for query_id in train_query_gt.keys():
    hit_with_score_l = train_candidate_d[query_id]['rerank']
    cc = [hit['text'] for hit, score in hit_with_score_l]
    train_query_cc[query_id] = cc


valid_query_gt = {}
valid_df = pd.read_csv("../data/valid_rewrite_001.csv")
for query_id, gold_citations in zip(valid_df['query_id'], valid_df['gold_citations']):
    valid_query_gt[query_id] = gold_citations.split(";")

# query -> retrieved CC list
# query_cc = {
#     "query1": [
#         ".... CC text 1 ....",
#         ".... CC text 2 ...."
#     ]
# }

valid_query_cc = {}
for query_id in valid_query_gt.keys():
    hit_with_score_l = valid_candidate_d[query_id]['rerank']
    cc = [hit['text'] for hit, score in hit_with_score_l]
    valid_query_cc[query_id] = cc

valid_query_cc_score = {}
for query_id in valid_query_gt.keys():
    hit_with_score_l = valid_candidate_d[query_id]['rerank']
    cc_score = [score for hit, score in hit_with_score_l]
    valid_query_cc_score[query_id] = cc_score
# ===================================================================================



def build_weak_dataset(query_gt, query_cc):
    samples = []

    for q, gt_citations in query_gt.items():
        cc_list = query_cc.get(q, [])

        for cc in cc_list:
            citation_spans = find_citations(cc)

            if not citation_spans:
                continue

            rule_scores = score_citation(cc, citation_spans)

            for cit, score in rule_scores.items():

                # 弱标签（核心）
                label = 1 if cit in gt_citations else 0

                samples.append({
                    "query": q,
                    "cc": cc,
                    "citation": cit,
                    "rule_score": score,
                    "label": label
                })

    return samples


def extract_features(sample):
    cc = sample["cc"].lower()
    cit = sample["citation"].lower()

    features = {}

    # Rule score
    features["rule_score"] = sample["rule_score"]

    # 是否出现解释词
    features["has_explanation"] = int(any(w in cc for w in EXPLANATION_WORDS))

    # 是否结论句
    features["has_conclusion"] = int(any(w in cc for w in CONCLUSION_WORDS))

    # citation 长度
    features["cit_len"] = len(cit)

    # citation 出现次数
    features["cit_freq"] = cc.count(cit)

    # 相对位置
    pos = cc.find(cit)
    features["position"] = pos / (len(cc) + 1)

    return features


def build_dataframe(samples):
    rows = []
    for s in samples:
        feats = extract_features(s)
        feats["label"] = s["label"]
        feats["query"] = s["query"]
        rows.append(feats)

    return pd.DataFrame(rows)


import lightgbm as lgb
from sklearn.model_selection import train_test_split

def split_by_query(df, valid_ratio=0.2):
    queries = df["query"].unique()
    train_q, valid_q = train_test_split(queries, test_size=valid_ratio, random_state=42)

    train_df = df[df["query"].isin(train_q)].reset_index(drop=True)
    valid_df = df[df["query"].isin(valid_q)].reset_index(drop=True)

    return train_df, valid_df

def build_group(df):
    return df.groupby("query").size().to_list()

def train_lgb_ranker_with_earlystop(df):
    features = [c for c in df.columns if c not in ["label", "query"]]

    # 1️⃣ 切分数据（按 query）
    train_df, valid_df = split_by_query(df)

    X_train = train_df[features]
    y_train = train_df["label"]

    X_valid = valid_df[features]
    y_valid = valid_df["label"]

    # 2️⃣ group
    train_group = build_group(train_df)
    valid_group = build_group(valid_df)

    train_data = lgb.Dataset(X_train, label=y_train, group=train_group)
    valid_data = lgb.Dataset(X_valid, label=y_valid, group=valid_group)

    # 3️⃣ 参数
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbosity": -1
    }

    # 4️⃣ 训练（带早停）
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=10)
        ]
    )

    return model

def rank_citations(model, cc_text):
    citation_spans = find_citations(cc_text)
    rule_scores = score_citation(cc_text, citation_spans)

    samples = []
    for cit, score in rule_scores.items():
        samples.append({
            "cc": cc_text,
            "citation": cit,
            "rule_score": score
        })

    feats = [extract_features(s) for s in samples]
    df = pd.DataFrame(feats)

    if len(df) == 0:
        return []

    preds = model.predict(df)

    results = list(zip([s["citation"] for s in samples], preds))
    results.sort(key=lambda x: x[1], reverse=True)

    return results


samples = build_weak_dataset(train_query_gt, train_query_cc)
df = build_dataframe(samples)

model = train_lgb_ranker_with_earlystop(df)

# 推理
# cc_text = train_query_cc["train_0001"][0]
# ranked = rank_citations(model, cc_text)

# print(ranked[:5])

from collections import defaultdict
import numpy as np

def normalize_scores(scores):
    scores = np.array(scores)
    exp = np.exp(scores - scores.max())  # 防止溢出
    return exp / exp.sum()

def normalize_weights(weights):
    w = np.array(weights)
    return w / (w.sum() + 1e-8)

def rank_citations_multi_cc_normalized(model, query, cc_list, cc_weights):
    citation_global = defaultdict(float)

    cc_results = []

    # Step 1: 先算每个 CC 的 citation 分布
    for cc in cc_list:
        citation_spans = find_citations(cc)
        rule_scores = score_citation(cc, citation_spans)

        samples = []
        for cit, score in rule_scores.items():
            samples.append({
                "cc": cc,
                "citation": cit,
                "rule_score": score
            })

        if not samples:
            continue

        feats = [extract_features(s) for s in samples]
        df = pd.DataFrame(feats)

        preds = model.predict(df)

        # 👉 关键：CC 内归一化
        probs = normalize_scores(preds)

        cc_results.append((samples, probs))

        # # 👉 简单 CC 权重（你可以换成 embedding）
        # cc_weights.append(len(cc))

    # Step 2: CC 权重归一化
    cc_weights = normalize_weights(cc_weights)

    # Step 3: 聚合
    for (samples, probs), w in zip(cc_results, cc_weights):
        for s, p in zip(samples, probs):
            cit = s["citation"]
            citation_global[cit] += p * w

    # Step 4: 排序
    results = sorted(citation_global.items(), key=lambda x: x[1], reverse=True)

    return results

result_l = []
gold_l = []
for query_id, cc_list in valid_query_cc.items():
    cc_weights = valid_query_cc_score[query_id]
    ret_l = rank_citations_multi_cc_normalized(model, query_id, cc_list, cc_weights)

    result_l.append([citation for citation,_ in ret_l])
    gold_l.append(valid_query_gt[query_id])

# ── 评估 ──────────────────────────────────────────────────────────────────────
for TOP_K in [5,7,10,12,15,17,20,22,25,27,30,33,35,37,40]:
    result_l2 = [r[:TOP_K] for r in result_l]
    recall    = metric_utils.cal_recall(result_l2, gold_l)
    precision = metric_utils.cal_precision(result_l2, gold_l)
    print(f"[{TOP_K}] Recall@{TOP_K}:{recall:.4f}, Precision:{precision:.4f}, F1:{2*recall*precision/(recall+precision):.4f}")
    
