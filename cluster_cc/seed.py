import faiss
import numpy as np
import pandas as pd
import bm25s
from collections import Counter, defaultdict
import os
import os.path
import sys

valid_df = pd.read_csv("../data/valid_rewrite_001.csv")
v_qid_2_gold = {}
for query_id, golds in zip(valid_df['query_id'], valid_df['gold_citations']):
    v_qid_2_gold[query_id] = set(golds.split(";"))
    
src_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

import citation_utils
from embedding_utils import BGEEmbedder

cc_df = pd.read_csv("../data/court_considerations.csv")
corpus = []
for text in cc_df['text']:
    corpus.append(text)
    
# ── 假设已有 kmeans 和 cluster_ids ──
from common import load_embedding

embeddings, parent_idx_l = load_embedding("/root/autodl-fs/bge-processed/_dense_sparse_court/")

# embedding.len > corpus.len
assert len(corpus) == len(set(parent_idx_l))

N = embeddings.shape[0]
D = embeddings.shape[1]  # 1024
K = 200
print(f"共 {N} 条文档，embedding 维度 {D}，聚类数 {K}")

embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
# faiss.normalize_L2(embeddings)
print("embedding loaded.")

kmeans = faiss.Kmeans(d=D, k=K, gpu=False, verbose=True, niter=20)
kmeans.train(embeddings)
# kmeans 训练完之后
_, cluster_ids = kmeans.index.search(embeddings, 1)
cluster_ids = cluster_ids.flatten()  # shape: (N,)
print("train doned.")

# 建簇 → 文档下标的映射
cluster_to_docs = defaultdict(list)
for doc_idx, cluster_id in enumerate(cluster_ids):
    cluster_to_docs[int(cluster_id)].append(doc_idx)

print(f"聚类完成，簇大小：最大={max(len(v) for v in cluster_to_docs.values())}，"
      f"最小={min(len(v) for v in cluster_to_docs.values())}，"
      f"平均={N//K}")

# ─────────────────────────────────────────
# 4. 建全库向量检索索引（用于 query 检索种子）
# ─────────────────────────────────────────

print("建向量索引...")
vec_index = faiss.IndexFlatIP(D)  # 内积 = 归一化后的余弦相似度
vec_index.add(embeddings)
print("向量索引建完")

# ─────────────────────────────────────────
# 5. 建 BM25 全库索引（bm25s 内存版）
# ─────────────────────────────────────────
import Stemmer
stemmer = Stemmer.Stemmer("german")
# print("建 BM25 索引...")
# tokenized_corpus = bm25s.tokenize(corpus, stopwords="de", stemmer=stemmer)  # 德语停用词
# bm25_index = bm25s.BM25()
# bm25_index.index(tokenized_corpus)
# print("BM25 索引建完")

# ─────────────────────────────────────────
# 6. 检索函数
# ─────────────────────────────────────────

def retrieve(
    query_id: str,
    query_text: str,
    query_embedding: np.ndarray,
    top_k: int = 100,
    seed_k: int = 100,       # 向量检索种子数量
    min_cluster_hits: int = 5,  # 簇扩展的最低门槛
    top_clusters: int = 3,   # 最多扩展几个簇,
) -> list[dict]:
    """
    query_text:      查询文本（用于BM25）
    query_embedding: 查询向量，shape (768,)，需已归一化
    top_k:           最终返回文档数
    """

    # ── Step 1: 向量检索 top seed_k 篇种子文档 ──
    q = query_embedding.astype(np.float32).reshape(1, -1)
    # faiss.normalize_L2(q)
    _, seed_ids = vec_index.search(q, seed_k)
    seed_ids = seed_ids.flatten()

    # ── Step 2: 统计种子落在哪些簇，投票 ──
    seed_cluster_ids = cluster_ids[seed_ids]
    cluster_counts = Counter(seed_cluster_ids.tolist())

    # 只取票数 >= min_cluster_hits 的簇，最多 top_clusters 个
    high_conf_clusters = [
        cluster_id
        for cluster_id, count in cluster_counts.most_common(top_clusters)
        if count >= min_cluster_hits
    ]

    # ── Step 3: 扩展候选文档 ──
    candidate_ids = set(seed_ids.tolist())  # 先放入种子文档
    for cid in high_conf_clusters:
        candidate_ids.update(cluster_to_docs[cid])

    candidate_ids = list(candidate_ids)
    candidate_corpus = [corpus[parent_idx_l[i]] for i in candidate_ids]
    print(f"候选文档数：{len(candidate_corpus)}（种子{seed_k} + 簇扩展）")

    # 判断下召回率
    citation_id_l = []
    for text in candidate_corpus:
        _cid_l = citation_utils.extract_citations_from_text(text)
        for cid in _cid_l:
            citation_id_l.append(cid)
    result_l = list(set(citation_id_l))

    # ── Step 4: BM25 在候选集内精排 ──
    tokenized_query = bm25s.tokenize([query_text], stopwords="de")
    
    # bm25s 支持指定子集检索
    tokenized_candidates = bm25s.tokenize(candidate_corpus, stopwords="de")
    local_bm25 = bm25s.BM25()
    local_bm25.index(tokenized_candidates)

    results, scores = local_bm25.retrieve(
        tokenized_query,
        corpus=candidate_corpus,
        k=min(top_k, len(candidate_corpus)),
    )

    # ── Step 5: 整理结果 ──
    final_results = []
    for doc_text, score in zip(results[0], scores[0]):
        # 找回原始下标
        original_idx = candidate_ids[candidate_corpus.index(doc_text)]
        final_results.append({
            "doc_id": original_idx,
            "text": doc_text,
            "bm25_score": float(score),
        })

    return final_results, result_l

import metric_utils

if __name__ == "__main__":
    recall_l = []
    gold_l = []

    embedder = BGEEmbedder('/root/.cache/modelscope/hub/models/BAAI/bge-m3')
    
    valid_df = pd.read_csv("../data/valid_rewrite_001.csv")
    for query_id, query_text, gold_citations in zip(valid_df['query_id'], valid_df['query2'], valid_df['gold_citations']):
        query_emb = embedder.encode(query_text).astype(np.float32)

        results, result_l = retrieve(
            query_id=query_id,
            query_text=query_text,
            query_embedding=query_emb,
            top_k=100,
            seed_k=100,
            min_cluster_hits=5,
            top_clusters=3,
        )

        # for i, r in enumerate(results[:5]):
        #     print(f"#{i+1} [doc_id={r['doc_id']}] score={r['bm25_score']:.4f}")
        #     print(f"     {r['text'][:100]}...")
    
        recall_l.append(result_l)
        gold_l.append(gold_citations.split(";"))

    recall = metric_utils.cal_recall(recall_l, gold_l)
    print(recall)

    



    