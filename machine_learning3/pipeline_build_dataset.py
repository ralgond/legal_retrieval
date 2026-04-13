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

OUTPUT_DIR = "../data/ml3/lgbm_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 加载数据 ──────────────────────────────────────────────────────────────────
print("Loading data...")
court_consideration_df = pd.read_csv("../data/court_considerations.csv")
court_consideration_d = dict(zip(court_consideration_df['citation'].tolist(), court_consideration_df['text'].tolist()))

court_doc = [{'citation':citation, 'text':text} for citation,text in zip(court_consideration_df['citation'].tolist(), court_consideration_df['text'].tolist())]

import common
train_candidate_d = common.read_candidate("../data/ml2/raw_train_candidate.pkl", court_consideration_d)
valid_candidate_d = common.read_candidate("../data/ml2/raw_valid_candidate.pkl", court_consideration_d)
test_candidate_d = common.read_candidate("../data/ml2/raw_test_candidate.pkl", court_consideration_d)

import citation_utils


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

class CC:
    def __init__(self, cc_id, cc_text, cc_score, from_hit_type, hit_rank, first_appear_sentence_index):
        self.cc_id = cc_id
        self.cc_text = cc_text
        self.cc_score = cc_score
        self.from_hit_type = from_hit_type
        self.hit_rank = hit_rank
        self.first_appear_sentence_index = first_appear_sentence_index

class Citation:
    FEATURE_NAMES = [
        'dense_max_score',
        'dense_avg_score',
        'dense_top3_score_sum',
        'sparse_max_score',
        'sparse_avg_score',
        'sparse_top3_score_sum',
        'rerank_max_score',
        'rerank_avg_score',
        'rerank_top3_score_sum',
    ]

    N_FEATS = len(FEATURE_NAMES)
    
    def __init__(self, cid):
        self.cid = cid
        self.refer_cc_l = [] 

    def add_refer_cc(self, cc_id, cc_text, cc_score, from_hit_type, hit_rank, first_appear_sentence_index: int | None = None):
        if first_appear_sentence_index is None:
            pass
        else: 
            self.refer_cc_l.append(CC(cc_id, cc_text, cc_score, from_hit_type, hit_rank, first_appear_sentence_index))

    def __extract_feature_method_1(self, from_hit_type):
        if from_hit_type not in ['dense', 'sparse', 'rerank']:
            raise ValueError(f"{from_hit_type} is not valid.")

        # print(len(self.refer_cc_l), from_hit_type, self.cid)
        score_l = [cc.cc_score for cc in self.refer_cc_l if cc.from_hit_type == from_hit_type]
        if len(score_l) > 0:
            max_score = max(score_l)
            avg_score = sum(score_l) / len(score_l)
            top3_score_sum = sum(sorted(score_l, reverse=True)[:3])
        else:
            max_score = 0.
            avg_score = 0.
            top3_score_sum = 0.

        return {
            from_hit_type + "_max_score" : max_score,
            from_hit_type + "_avg_score" : avg_score,
            from_hit_type + "_top3_score_sum" : top3_score_sum,
        }
        
    def extract_feature(self): # return Dict[feature_name->float]

        dense_d = self.__extract_feature_method_1('dense')
        sparse_d = self.__extract_feature_method_1('sparse')
        rerank_d = self.__extract_feature_method_1('rerank')
        
        self.dense_l = []
        self.sparse_l = []
        self.rerank_l = []

        for cc in self.refer_cc_l:
            if cc.from_hit_type == 'dense':
                self.dense_l.append(cc)
            elif cc.from_hit_type == 'sparse':
                self.sparse_l.append(cc)
            elif cc.from_hit_type == 'rerank':
                self.rerank_l.append(cc)

        merged_method_1_dict = {**dense_d, **sparse_d, **rerank_d}

        return merged_method_1_dict

class Query:
    def __init__(self, q_id):
        self.q_id = q_id
        self.cc_id_2_text_d = {}
        self.cc_id_2_norm_dense_score = dict()
        self.cc_id_2_norm_sparse_score = dict()
        self.cc_id_2_norm_rerank_score = dict()
        self.norm_dense_hits = None
        self.norm_sparse_hits = None
        self.norm_rerank_hits = None

    def assign_text_to_cc(self, court_consideration_d):
        cc_id_l = self.get_cc_id_l()
        for cc_id in cc_id_l:
            self.cc_id_2_text_d[cc_id] = court_consideration_d[cc_id]

    def get_text_for_cc(self, cc_id):
        return self.cc_id_2_text_d[cc_id]

    def add_norm_dense_hits(self, norm_dense_hits):
        for hit, score in norm_dense_hits:
            cc_id = hit['citation']
            self.cc_id_2_norm_dense_score[cc_id] = score
        self.norm_dense_hits = norm_dense_hits.copy()

    def add_norm_sparse_hits(self, norm_sparse_hits):
        for hit, score in norm_sparse_hits:
            cc_id = hit['citation']
            self.cc_id_2_norm_sparse_score[cc_id] = score
        self.norm_sparse_hits = norm_sparse_hits.copy()

    def add_norm_rerank_hits(self, norm_rerank_hits):
        for hit, score in norm_rerank_hits:
            cc_id = hit['citation']
            self.cc_id_2_norm_rerank_score[cc_id] = score
        self.norm_rerank_hits = norm_rerank_hits.copy()

    def get_cc_dense_norm(self, cc_id):
        return self.cc_id_2_norm_dense_score.get(cc_id, 0.)

    def get_cc_sparse_norm(self, cc_id):
        return self.cc_id_2_norm_sparse_score.get(cc_id, 0.)

    def get_cc_rerank_norm(self, cc_id):
        return self.cc_id_2_norm_rerank_score.get(cc_id, 0.)

    def get_cc_id_l (self):
        return list(set(self.cc_id_2_norm_dense_score.keys()) | set(self.cc_id_2_norm_sparse_score.keys()) | set(self.cc_id_2_norm_rerank_score.keys()))

    def extract_feature(self): # Dict[citation_id, Citation]
        cc_id_2_parsed_cc_d = {}

        # step 1: parse all cc
        first_appear_sentence_index_d = {}
        for cc_id in self.get_cc_id_l():
            cc_text = self.get_text_for_cc(cc_id)
            parsed_cc = citation_utils.parse_cc_output_citations_and_sentences(cc_text)
            cc_id_2_parsed_cc_d[cc_id] = parsed_cc
            for citation_id, first_appear_sentence_index in parsed_cc['citations']:
                first_appear_sentence_index_d[(citation_id,cc_id)] = first_appear_sentence_index

        # step 2: found all refered citation
        citation_id_2_citation_d = dict()
        for cc_id, parsed_cc in cc_id_2_parsed_cc_d.items():
            for citation_id, _ in parsed_cc['citations']:
                if citation_id not in citation_id_2_citation_d:
                    citation_id_2_citation_d[citation_id] = Citation(citation_id)

        print("self.norm_dense_hits.len:", len(self.norm_dense_hits))
        print("self.norm_sparse_hits.len:", len(self.norm_sparse_hits))
        print("self.norm_rerank_hits.len:", len(self.norm_rerank_hits))
        
        # step 3: assign information to citation
        for rank, (hit, score) in enumerate(self.norm_dense_hits, start=1):
            cc_id = hit['citation']
            cc_text = self.get_text_for_cc(cc_id)
            for citation_id, citation in citation_id_2_citation_d.items():
                citation.add_refer_cc(cc_id, cc_text, score, "dense", rank, first_appear_sentence_index_d.get((citation_id,cc_id), None))
            
        for rank, (hit, score) in enumerate(self.norm_sparse_hits, start=1):
            cc_id = hit['citation']
            cc_text = self.get_text_for_cc(cc_id)
            for citation_id, citation in citation_id_2_citation_d.items():
                citation.add_refer_cc(cc_id, cc_text, score, "sparse", rank, first_appear_sentence_index_d.get((citation_id,cc_id), None))

        for rank, (hit, score) in enumerate(self.norm_rerank_hits, start=1):
            cc_id = hit['citation']
            cc_text = self.get_text_for_cc(cc_id)
            for citation_id, citation in citation_id_2_citation_d.items():
                citation.add_refer_cc(cc_id, cc_text, score, "rerank", rank, first_appear_sentence_index_d.get((citation_id,cc_id), None))

        accum = {}
        for citation_id, citation in citation_id_2_citation_d.items():
            accum[citation_id] = citation.extract_feature()
            
        return accum

def extract_features_for_query(
        query_id: str, query: str
) -> dict[str, np.ndarray]:
    """
    对单个 query 做检索+rerank，返回
      { citation_id: np.ndarray(Citation.N_FEATS,) }
    """

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

    norm_hits1 = _maxmin_normalize_hits(hits1)
    norm_hits2 = _maxmin_normalize_hits(hits2)
    norm_hits3 = _maxmin_normalize_hits(hits3)
    
    q = Query(q_id=query_id)
    q.add_norm_dense_hits(norm_hits1)
    q.add_norm_sparse_hits(norm_hits2)
    q.add_norm_rerank_hits(norm_hits3)
    q.assign_text_to_cc(court_consideration_d)
    accum = q.extract_feature()
    
    # 整理为特征向量
    cid_feat_d: dict[str, np.ndarray] = {}

    for cid, a in accum.items():
        # freq = a["cite_freq"]

        feat_vec = np.array([
            a['dense_max_score'],
            a['dense_avg_score'],
            a['dense_top3_score_sum'],
            a['sparse_max_score'],
            a['sparse_avg_score'],
            a['sparse_top3_score_sum'],
            a['rerank_max_score'],
            a['rerank_avg_score'],
            a['rerank_top3_score_sum'],
        ], dtype=np.float32)
        assert len(feat_vec) == Citation.N_FEATS, \
            f"Feature dim mismatch: {len(feat_vec)} vs {Citation.N_FEATS}  cid={cid}"
        cid_feat_d[cid] = feat_vec

    return cid_feat_d




def build_split(df: pd.DataFrame, split_name: str,
                min_pos_in_candidate: int = 1,
                only_filter_train:    bool = True,
                neg_pos_ratio:        int  = 10):
    """
    对一个 split（train 或 valid）构造特征矩阵和标签。
    每个 query 的所有候选 citation（由检索+rerank 产生）全部参与特征提取；
    训练集对负样本做随机下采样以缓解类别不平衡，valid 集保持完整。

    Parameters
    ----------
    min_pos_in_candidate : int
        候选集中至少需要包含多少个正样本，不足则跳过该 query。
        仅在 only_filter_train=True 且 split_name=="train" 时生效，
        或 only_filter_train=False 时对所有 split 生效。
    only_filter_train : bool
        True  → 只对 train split 做过滤和负采样（valid 保持完整，便于公平评估）
        False → 所有 split 都过滤和负采样
    neg_pos_ratio : int
        训练时每个正样本保留多少个负样本（默认 10，即 1:10）。
        设为 -1 则不做负采样（保留全部负样本）。
        val split 始终不做负采样。
    """
    do_filter     = (split_name == "train") if only_filter_train else True
    do_neg_sample = (split_name == "train") if only_filter_train else True

    rng = np.random.default_rng(seed=42)

    all_feats, all_labels, groups = [], [], []
    skipped, total = 0, 0

    for i, (query_id, query, gold_citations) in enumerate(
            zip(df['query_id'], df['query2'], df['gold_citations'])):
        gold_set = set(gold_citations.split(';'))
        total += 1
        print(f"[{split_name}] {i+1}/{len(df)}  query_id={query_id}")

        cid_feat_d = extract_features_for_query(query_id, query)

        # 全部候选，分为正负两组
        all_cids = list(cid_feat_d.keys())
        pos_cids = [c for c in all_cids if c in gold_set]
        neg_cids = [c for c in all_cids if c not in gold_set]

        # ── 正样本过滤 ────────────────────────────────────────────────────────
        if do_filter:
            if len(pos_cids) < min_pos_in_candidate:
                skipped += 1
                print(f"  ↳ SKIP  pos={len(pos_cids)} "
                      f"< min_pos_in_candidate={min_pos_in_candidate}  "
                      f"(gold_size={len(gold_set)})")
                continue
        # ─────────────────────────────────────────────────────────────────────

        # ── 负采样（仅 train，val 保持完整） ──────────────────────────────────
        if do_neg_sample and neg_pos_ratio >= 0:
            max_neg = len(pos_cids) * neg_pos_ratio
            if len(neg_cids) > max_neg:
                neg_cids = list(rng.choice(neg_cids, size=max_neg, replace=False))
        # ─────────────────────────────────────────────────────────────────────

        sampled_cids = pos_cids + neg_cids
        print(f"  pos={len(pos_cids)}  neg={len(neg_cids)}  "
              f"ratio=1:{len(neg_cids)//max(len(pos_cids),1)}")

        for cid in sampled_cids:
            all_feats.append(cid_feat_d[cid])
            all_labels.append(1 if cid in gold_set else 0)

        groups.append(len(sampled_cids))

    kept = total - skipped
    print(f"\n[{split_name}] kept={kept}/{total}  skipped={skipped}  "
          f"(filter={'on' if do_filter else 'off'}, "
          f"min_pos_in_candidate={min_pos_in_candidate})")

    X = np.stack(all_feats, axis=0)          # (N, N_FEATS)
    y = np.array(all_labels, dtype=np.int32) # (N,)
    g = np.array(groups,     dtype=np.int32) # (n_queries,)
    print(f"[{split_name}] X={X.shape}  pos_rate={y.mean():.4f}")

    np.save(f"{OUTPUT_DIR}/{split_name}_features.npy", X)
    np.save(f"{OUTPUT_DIR}/{split_name}_labels.npy",   y)
    np.save(f"{OUTPUT_DIR}/{split_name}_groups.npy",   g)
    print(f"[{split_name}] Saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min_pos", type=int, default=1,
        help="候选集中至少包含多少个正样本才保留该 query（仅对 train 生效）。"
             "默认 1，即丢弃候选集中 0 个正样本的 query。"
             "设为 2 则更严格，只保留至少命中 2 个 gold 的 query。")
    parser.add_argument(
        "--neg_pos_ratio", type=int, default=10,
        help="训练集每个正样本保留多少负样本（默认 10）。-1 表示不采样（保留全部）。")
    parser.add_argument(
        "--filter_valid", action="store_true",
        help="同时对 valid 集做相同过滤（默认只过滤 train）")
    args = parser.parse_args()

    # 保存特征名供后续使用
    with open(f"{OUTPUT_DIR}/feature_names.txt", "w") as f:
        f.write("\n".join(Citation.FEATURE_NAMES))

    train_df = pd.read_csv("../data/train_rewrite_001.csv")
    valid_df  = pd.read_csv("../data/valid_rewrite_001.csv")

    build_split(train_df, "train",
                min_pos_in_candidate=args.min_pos,
                only_filter_train=not args.filter_valid,
                neg_pos_ratio=args.neg_pos_ratio)
    build_split(valid_df, "valid",
                min_pos_in_candidate=args.min_pos,
                only_filter_train=not args.filter_valid,
                neg_pos_ratio=args.neg_pos_ratio)

    

    