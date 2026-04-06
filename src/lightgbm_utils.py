import numpy as np
import lightgbm as lgb
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
import os
import os.path
import sys
from tqdm.notebook import tqdm

def filter_queries_by_recall(all_queries, ground_truth, min_recall=0.5):
    """
    过滤掉召回覆盖率太低的 query，避免用噪声样本训练
    """
    valid_qids = []
    for qid, data in tqdm(all_queries.items(), total=len(all_queries)):
        gt_citations = ground_truth.get(qid, set())
        if not gt_citations:
            continue

        # print(data['citations'])
        retrieved_citations = set([query_id for query_id,_ in data['citations'].items()])

        # print("[filter_queries_by_recall] retrieved_citations.len:", len(retrieved_citations))

        recall = len(gt_citations & retrieved_citations) / len(gt_citations)
        if recall >= min_recall:
            valid_qids.append(qid)

    print(f"过滤后保留 {len(valid_qids)} / {len(all_queries)} 个 query")

    
    # 返回值：List[str], Dict[qid -> {'query_text': str, 'ccs': List[dict]}]
    return valid_qids 

def convert_to_query_data(data_list, topk_per_citation=None):
    results = []
    
    for item in data_list:
        query_id = item["query_id"]
        gold = set(item.get("gold", []))
        cc_list = item.get("cc", [])
        
        citation_dict = {}
        
        for cc in cc_list:
            # ✅ 只从 cc 读取（唯一来源）
            cc_score = float(cc.get("cc_score", 0.0))
            cc_rank = int(cc.get("rank", 999))
            
            # 你的结构：['citation_id','position']
            citation_infos = cc.get("citations", [])
            
            for cit in citation_infos:
                # 统一解析
                if isinstance(cit, tuple) or isinstance(cit, list):
                    citation_id, position = cit
                elif isinstance(cit, dict):
                    citation_id = cit.get("citation_id")
                    position = cit.get("position", 999)
                else:
                    continue
                
                if citation_id is None:
                    continue
                
                # 初始化
                if citation_id not in citation_dict:
                    citation_dict[citation_id] = []
                
                # ✅ 关键：rank 来自 cc，不从 cit 取
                citation_dict[citation_id].append({
                    "cc_score": cc_score,
                    "rank": cc_rank,          # ✅ 正确位置
                    "position": int(position)
                })
        
        # （可选）截断 top-k
        if topk_per_citation is not None:
            for cid in citation_dict:
                citation_dict[cid] = sorted(
                    citation_dict[cid],
                    key=lambda x: x["cc_score"],
                    reverse=True
                )[:topk_per_citation]
        
        results.append({
            "query_id": query_id,
            "citations": citation_dict,
            "gold": gold
        })
    
    return results


import numpy as np

def extract_features(occurrences, topk=5):
    # 按 cc_score 排序（重要！）
    occs = sorted(occurrences, key=lambda x: x["cc_score"], reverse=True)
    
    scores = np.array([o["cc_score"] for o in occs])
    positions = np.array([o["position"] for o in occs])
    ranks = np.array([o["rank"] for o in occs])
    
    # 归一化（防止不同 query scale 不同）
    max_score = scores.max()
    norm_scores = scores / (max_score + 1e-8)
    
    features = {}
    
    # ===== 基础统计 =====
    features["num_occurrences"] = len(occs)
    features["score_max"] = scores.max()
    features["score_mean"] = scores.mean()
    features["score_sum"] = scores.sum()
    features["score_std"] = scores.std()
    
    # ===== rank 信息 =====
    features["rank_min"] = ranks.min()
    features["rank_mean"] = ranks.mean()
    features["rank_top10_count"] = np.sum(ranks <= 10)
    
    # ===== position 信息 =====
    features["pos_min"] = positions.min()
    features["pos_mean"] = positions.mean()
    features["pos_inv_mean"] = np.mean(1.0 / positions)
    features["pos_inv_max"] = np.max(1.0 / positions)
    
    # ===== top-k raw features（关键！）=====
    for i in range(topk):
        if i < len(occs):
            features[f"top{i+1}_score"] = occs[i]["cc_score"]
            features[f"top{i+1}_pos"] = occs[i]["position"]
            features[f"top{i+1}_rank"] = occs[i]["rank"]
        else:
            features[f"top{i+1}_score"] = 0.0
            features[f"top{i+1}_pos"] = 999
            features[f"top{i+1}_rank"] = 999
    
    # ===== log-sum-exp（比 sum 更稳）=====
    features["logsumexp_score"] = np.log(np.sum(np.exp(scores)))
    
    # ===== 归一化统计 =====
    features["norm_score_mean"] = norm_scores.mean()
    features["norm_score_max"] = norm_scores.max()
    
    return features


def build_lgb_dataset(all_queries):
    X = []
    y = []
    group = []
    
    for q in tqdm(all_queries, total=len(all_queries), desc="build_lgb_dataset"):
        q_features = []
        q_labels = []
        
        for citation, occs in q["citations"].items():
            feats = extract_features(occs)
            q_features.append(list(feats.values()))
            
            label = 1 if citation in q["gold"] else 0
            q_labels.append(label)

        pos_label_cnt = 0
        for label in q_labels:
            if label == 1:
                pos_label_cnt += 1
        print(f"[{q['query_id']}]: {pos_label_cnt}/{len(q_labels)}")
        
        X.extend(q_features)
        y.extend(q_labels)
        group.append(len(q_features))  # 每个 query 一个 group
    
    return np.array(X), np.array(y), group


import numpy as np
import random

def build_lgb_dataset2(all_queries, valid_ratio=0.2, seed=42):
    random.seed(seed)
    
    # ===== 先按“是否有正样本”分组 =====
    pos_queries = []
    neg_queries = []
    
    for q in all_queries:
        if len(q["citations"]) == 0:
            continue  # 跳过空 query
        
        has_pos = any(c in q["gold"] for c in q["citations"])
        
        if has_pos:
            pos_queries.append(q)
        else:
            neg_queries.append(q)
    
    # ===== 打乱 =====
    random.shuffle(pos_queries)
    random.shuffle(neg_queries)
    
    # ===== valid 优先从有正样本的 query 里取 =====
    n_valid = int(len(all_queries) * valid_ratio)
    
    valid_queries = []
    
    # 1️⃣ 先拿有正样本的
    for q in pos_queries:
        if len(valid_queries) < n_valid:
            valid_queries.append(q)
        else:
            break
    
    # 2️⃣ 如果不够，再补负样本 query
    i = 0
    while len(valid_queries) < n_valid and i < len(neg_queries):
        valid_queries.append(neg_queries[i])
        i += 1
    
    # ===== train = 剩下的 =====
    valid_set = set(id(q) for q in valid_queries)
    train_queries = [q for q in all_queries if id(q) not in valid_set]
    
    # ===== 构建 dataset =====
    def build_split(queries):
        X = []
        y = []
        group = []
        
        for q in queries:
            q_features = []
            q_labels = []
            
            for citation, occs in q["citations"].items():
                feats = extract_features(occs)
                q_features.append(list(feats.values()))
                
                label = 1 if citation in q["gold"] else 0
                q_labels.append(label)
            
            # ⚠️ skip 空 query
            if len(q_features) == 0:
                continue
            
            X.extend(q_features)
            y.extend(q_labels)
            group.append(len(q_features))
        
        return np.array(X), np.array(y), group
    
    train_X, train_y, train_group = build_split(train_queries)
    valid_X, valid_y, valid_group = build_split(valid_queries)
    
    return train_X, train_y, train_group, valid_X, valid_y, valid_group

import citation_utils

def reranked_hits_to_json(query_id, hit_with_score_l):
    '''
    query_id = item["query_id"]
    gold = set(item.get("gold", []))
    cc_list = item.get("cc", [])
    '''
    data_list = []
    for rank, (hit, cc_score) in enumerate(hit_with_score_l):
        _tmp = citation_utils.parse_cc_output_citations_and_sentences_2(hit['text'])
        cc = []
        for citation_id, position in _tmp['citations']:
            cc.append({'citation_id':citation_id, 'position':position})
        
        data_list.append({'query_id':query_id, 'cc':cc, 'rank':rank, 'cc_score':cc_score})
        
    return data_list