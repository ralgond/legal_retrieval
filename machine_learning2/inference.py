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

# ── 超参数 ────────────────────────────────────────────────────────────────────
TOP_HIT_THRESHOLD = 10
RETRIEVE_TOP_K    = 100
TOP_K             = 20
MODEL_PATH        = "../data/ml2/lgbm_model/lgbm_ranker.txt"
DATA_DIR          = "../data/ml2/lgbm_data"

# ── 特征名 ────────────────────────────────────────────────────────────────────
with open(f"{DATA_DIR}/feature_names.txt") as f:
    FEATURE_NAMES = [l.strip() for l in f.readlines()]
N_FEATS = len(FEATURE_NAMES)

# ── 加载 LightGBM 模型 ────────────────────────────────────────────────────────
print(f"Loading LightGBM model from {MODEL_PATH} ...")
booster = lgb.Booster(model_file=MODEL_PATH)
print(f"LightGBM model loaded. num_trees={booster.num_trees()}")

# ── 加载检索模型 ──────────────────────────────────────────────────────────────
court_consideration_df = pd.read_csv("../data/court_considerations.csv")
court_consideration_d = dict(zip(court_consideration_df['citation'].tolist(), court_consideration_df['text'].tolist()))

print("Loaded cc...")

import common
train_candidate_d = common.read_candidate("../data/ml2/raw_train_candidate.pkl", court_consideration_d)
valid_candidate_d = common.read_candidate("../data/ml2/raw_valid_candidate.pkl", court_consideration_d)
test_candidate_d = common.read_candidate("../data/ml2/raw_test_candidate.pkl", court_consideration_d)

# ── 文本匹配辅助函数 ──────────────────────────────────────────────────────────
def _char_bigram_overlap(a: str, b: str) -> float:
    def bigrams(s: str) -> set:
        s = s.lower()
        return {s[i:i+2] for i in range(len(s) - 1)} if len(s) >= 2 else set()
    bg_a, bg_b = bigrams(a), bigrams(b)
    if not bg_a or not bg_b:
        return 0.0
    return len(bg_a & bg_b) / len(bg_a | bg_b)


def _query_term_overlap(query_tokens: set, sentence: str) -> float:
    if not query_tokens:
        return 0.0
    sent_tokens = set(sentence.lower().split())
    return len(query_tokens & sent_tokens) / len(query_tokens)


# ── 特征提取 ──────────────────────────────────────────────────────────────────
def extract_features_for_query(query: str) -> dict[str, np.ndarray]:
    if query_id in train_candidate_d:
        hits1 = train_candidate_d[query_id]['dense']
        hits2 = train_candidate_d[query_id]['sparse']
        hits3 = train_candidate_d[query_id]['rerank']
    elif query_id in valid_candidate_d:
        hits1 = valid_candidate_d[query_id]['dense']
        hits2 = valid_candidate_d[query_id]['sparse']
        hits3 = valid_candidate_d[query_id]['rerank']
    elif query_id in test_candidate_d:
        hits1 = test_candidate_d[query_id]['dense']
        hits2 = test_candidate_d[query_id]['sparse']
        hits3 = test_candidate_d[query_id]['rerank']
    else:
        return {}


    dense_score_d  = {hit['citation']: score for hit, score in hits1}
    sparse_score_d = {hit['citation']: score for hit, score in hits2}
    hit_with_score_l = hits3


    query_tokens = set(query.lower().split())

    accum: dict[str, dict] = defaultdict(lambda: {
        "log_pos_decay":          0.0,
        # "hit_rank_decay":         0.0,
        "rr_pos_decay":           0.0,
        "dense_log_pos":          0.0,
        "sparse_log_pos":         0.0,
        # "cite_freq":              0,
        "top_hit_bonus":          0.0,
        "max_score_pos":          0.0,
        "score_sq_log_pos":       0.0,
        "reranker_scores":        [],
        "hit_ranks":              [],
        "sum_dense_score":        0.0,
        "sum_sparse_score":       0.0,
        "hit_ids":                set(),
        "weighted_hit_coverage":  0.0,
        "sent_char_lens":         [],
        "sent_word_lens":         [],
        "sent_pos_ratios":        [],
        "query_term_overlaps":    [],
        "char_bigram_overlaps":   [],
    })

    for hit_rank, (hit, reranker_score) in enumerate(hit_with_score_l):
        parsed_cc  = citation_utils.parse_cc_output_citations_and_sentences(hit["text"])
        hit_doc_id = hit["citation"]
        dense_score  = dense_score_d.get(hit_doc_id, 0.0)
        sparse_score = sparse_score_d.get(hit_doc_id, 0.0)

        sentences   = parsed_cc.get("sentences", [])
        total_sents = max(len(sentences), 1)

        for cid, idx in parsed_cc["citations"]:
            a = accum[cid]
            log_pos   = 1.0 / math.log(2 + idx)
            rr_pos    = 1.0 / (1 + idx)
            hit_decay = 1.0 / math.log(2 + hit_rank)

            a["log_pos_decay"]    += reranker_score * log_pos
            # a["hit_rank_decay"]   += reranker_score * hit_decay
            a["rr_pos_decay"]     += reranker_score * rr_pos
            a["dense_log_pos"]    += dense_score * log_pos
            a["sparse_log_pos"]   += sparse_score * log_pos
            # a["cite_freq"]        += 1
            a["score_sq_log_pos"] += (reranker_score ** 2) * log_pos
            a["sum_dense_score"]  += dense_score
            a["sum_sparse_score"] += sparse_score

            val = reranker_score * log_pos
            if val > a["max_score_pos"]:
                a["max_score_pos"] = val
            if hit_rank < TOP_HIT_THRESHOLD:
                a["top_hit_bonus"] += val

            a["reranker_scores"].append(reranker_score)
            a["hit_ranks"].append(hit_rank)
            a["hit_ids"].add(hit_doc_id)
            a["weighted_hit_coverage"] += reranker_score

            sent_text = sentences[idx] if idx < len(sentences) else ""
            a["sent_char_lens"].append(len(sent_text))
            a["sent_word_lens"].append(len(sent_text.split()))
            a["sent_pos_ratios"].append(idx / total_sents)
            a["query_term_overlaps"].append(_query_term_overlap(query_tokens, sent_text))
            a["char_bigram_overlaps"].append(_char_bigram_overlap(query, sent_text))

    def _doc_feats(cid: str) -> tuple:
        doc_text = court_consideration_d.get(cid, "")
        return float(len(doc_text)), float(len(doc_text.split())), float(doc_text.count("["))

    cid_feat_d: dict[str, np.ndarray] = {}
    for cid, a in accum.items():
        # freq = a["cite_freq"]
        rs   = a["reranker_scores"]
        scl  = a["sent_char_lens"]        or [0]
        swl  = a["sent_word_lens"]        or [0]
        spr  = a["sent_pos_ratios"]       or [0.0]
        qto  = a["query_term_overlaps"]   or [0.0]
        cbo  = a["char_bigram_overlaps"]  or [0.0]
        doc_char, doc_word, doc_ncit = _doc_feats(cid)

        feat_vec = np.array([
            a["log_pos_decay"],
            # a["hit_rank_decay"],
            a["rr_pos_decay"],
            a["dense_log_pos"],
            a["sparse_log_pos"],
            # float(freq),
            # math.log(1 + freq),
            a["top_hit_bonus"],
            a["max_score_pos"],
            a["score_sq_log_pos"],
            float(np.mean(rs)),
            float(np.max(rs)),
            float(min(a["hit_ranks"])),
            a["sum_dense_score"],
            a["sum_sparse_score"],
            a["sum_dense_score"] / (a["sum_sparse_score"] + 1e-9),
            float(len(a["hit_ids"])),
            a["weighted_hit_coverage"],
            float(np.mean(scl)),
            float(np.max(scl)),
            float(np.mean(swl)),
            float(np.mean(spr)),
            float(np.min(spr)),
            float(np.mean(qto)),
            float(np.max(qto)),
            float(np.mean(cbo)),
            doc_char,
            doc_word,
            doc_ncit,
        ], dtype=np.float32)
        assert len(feat_vec) == N_FEATS, \
            f"Feature dim mismatch: {len(feat_vec)} vs {N_FEATS}"
        cid_feat_d[cid] = feat_vec

    return cid_feat_d


# ── 推理主循环 ────────────────────────────────────────────────────────────────
valid_df  = pd.read_csv("../data/valid_rewrite_001.csv")
result_l  = []
gold_l    = []

for i, (query_id, query, gold_citations) in enumerate(
        zip(valid_df['query_id'], valid_df['query2'], valid_df['gold_citations'])):

    gold_set = set(gold_citations.split(';'))
    gold_l.append(list(gold_set))
    print(f"[{i+1}/{len(valid_df)}] query_id={query_id}")

    # 1. 特征提取
    cid_feat_d = extract_features_for_query(query)

    if not cid_feat_d:
        result_l.append([])
        continue

    # 全部候选，不做截断
    all_cids = list(cid_feat_d.keys())

    # 2. 构造特征矩阵
    X = np.stack([cid_feat_d[c] for c in all_cids], axis=0)  # (n_cand, N_FEATS)

    # 3. LightGBM 打分
    scores = booster.predict(X)  # (n_cand,)

    # 4. 按分数降序，取 TOP_K
    order    = np.argsort(scores)[::-1]
    # top_cids = [all_cids[idx] for idx in order[:TOP_K]]
    top_cids = [all_cids[idx] for idx in order]
    result_l.append(top_cids)

# ── 评估 ──────────────────────────────────────────────────────────────────────
for TOP_K in [5,7,10,12,15,17,20,22,25,27,30,33,35,37,40]:
    result_l2 = [r[:TOP_K] for r in result_l]
    recall    = metric_utils.cal_recall(result_l2, gold_l)
    precision = metric_utils.cal_precision(result_l2, gold_l)
    print(f"[{TOP_K}] Recall@{TOP_K}:{recall:.4f}, Precision:{precision:.4f}, F1:{2*recall*precision/(recall+precision):.4f}")
