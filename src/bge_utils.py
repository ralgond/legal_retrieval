import numpy as np

def colbert_topk_hits(model, query, hits, top_k=100):
    q_result = corpus_embeddings = model.encode(
        query,
        batch_size=10,
        max_length=512,
        return_dense=False,      # ColBERT 模式不需要稠密向量
        return_sparse=False,     # 不需要稀疏向量
        return_colbert_vecs=True # 关键：返回 ColBERT token 级向量
    )['colbert_vecs']

    p_result_l = []
    corpus = [hit['text'] for hit in hits]
    # 启用 ColBERT 模式进行编码
    p_result_l = model.encode(
        corpus,
        batch_size=10,
        max_length=512,
        return_dense=False,      # ColBERT 模式不需要稠密向量
        return_sparse=False,     # 不需要稀疏向量
        return_colbert_vecs=True # 关键：返回 ColBERT token 级向量
    )['colbert_vecs']

    scores = []
    for p_result in p_result_l:
        score = model.colbert_score(q_result, p_result)
        scores.append(score)

    top_k_indices = np.argsort(scores)[::-1][:top_k]

    ret_hits = []
    
    for index in top_k_indices:
        ret_hits.append((hits[index], scores[index]))

    return sorted(ret_hits, key=lambda x: x[1], reverse=True)

import numpy as np

def calc_dense_scores(model, query, hits):
    q_result = model.encode(
        query,
        batch_size=10,
        max_length=512,
        return_dense=True,      # ColBERT 模式不需要稠密向量
        return_sparse=False,     # 不需要稀疏向量
        return_colbert_vecs=False # 关键：返回 ColBERT token 级向量
    )['dense_vecs']

    p_result_l = []
    corpus = [hit['text'] for hit in hits]

    p_result_l = model.encode(
        corpus,
        batch_size=10,
        max_length=512,
        return_dense=True,      # ColBERT 模式不需要稠密向量
        return_sparse=False,     # 不需要稀疏向量
        return_colbert_vecs=False # 关键：返回 ColBERT token 级向量
    )['dense_vecs']

    scores = []
    for p_result in p_result_l:

        # dot product
        score = np.dot(q_result, p_result)

        scores.append(score)

    return scores

import numpy as np

def calc_sparse_scores(model, query, hits):
    q_result = corpus_embeddings = model.encode(
        query,
        batch_size=10,
        max_length=512,
        return_dense=False,      # ColBERT 模式不需要稠密向量
        return_sparse=True,     # 不需要稀疏向量
        return_colbert_vecs=False # 关键：返回 ColBERT token 级向量
    )['lexical_weights']

    p_result_l = []
    corpus = [hit['text'] for hit in hits]
    # 启用 ColBERT 模式进行编码
    p_result_l = model.encode(
        corpus,
        batch_size=10,
        max_length=512,
        return_dense=False,      # ColBERT 模式不需要稠密向量
        return_sparse=True,     # 不需要稀疏向量
        return_colbert_vecs=False # 关键：返回 ColBERT token 级向量
    )['lexical_weights']

    scores = []
    for p_result in p_result_l:
        score = model.compute_lexical_matching_score(q_result, p_result)
        scores.append(score)

    return scores

def hybrid_scores(model, query, hits, weight1=0.5, weight2=0.2, weight3=0.3):
    q_result = model.encode(
        query,
        batch_size=10,
        max_length=512,
        return_dense=True,      # ColBERT 模式不需要稠密向量
        return_sparse=True,     # 不需要稀疏向量
        return_colbert_vecs=True # 关键：返回 ColBERT token 级向量
    )

    corpus = [hit['text'] for hit in hits]

    p_result_l = model.encode(
        corpus,
        batch_size=10,
        max_length=512,
        return_dense=True,      # ColBERT 模式不需要稠密向量
        return_sparse=True,     # 不需要稀疏向量
        return_colbert_vecs=True # 关键：返回 ColBERT token 级向量
    )

    dense_q = q_result['dense_vecs']
    dense_p_l = p_result_l['dense_vecs']

    dense_scores = []
    for dense_p in dense_p_l:
        # dot product
        score = np.dot(dense_q, dense_p)
        dense_scores.append(score)

    
    sparse_q = q_result['lexical_weights']
    sparse_p_l = p_result_l['lexical_weights']
    sparse_scores = []
    for sparse_p in sparse_p_l:
        score = model.compute_lexical_matching_score(sparse_q, sparse_p)
        sparse_scores.append(score)


    colbert_q = q_result['colbert_vecs']
    colbert_p_l = p_result_l['colbert_vecs']
    colbert_scores = []
    for colbert_p in colbert_p_l:
        score = model.colbert_score(colbert_q, colbert_p)
        colbert_scores.append(score)

    ret_scores = []
    for dense,sparse,colbert in zip(dense_scores, sparse_scores, colbert_scores):
        ret_scores.append(weight1*dense + weight2*sparse + weight3*colbert)

    return ret_scores
