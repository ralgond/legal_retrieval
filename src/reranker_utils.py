import sys
import os
from tqdm import tqdm

import text_chunk



def rerank_batch_with_anything(reranker, query: str, sentence_with_anything_l: list, batch_size=10):

    null_fp = open(os.devnull, 'w')
    sys.stderr = sys.__stderr__

    any_l = []
    sentences = []
    for anything, sentence in sentence_with_anything_l:
        any_l.append(anything)
        sentences.append(sentence)
    
    l = []
    pairs = []
    docs2 = []
    for sentence in sentences:
        pairs.append([query, sentence])
        docs2.append(sentence)
        if len(pairs) == batch_size:
            sys.stderr = null_fp
            scores = reranker.compute_score(pairs, max_length=512, batch_size=batch_size, normalize=True, verbose=False)
            sys.stderr = sys.__stderr__
            for idx, _score in enumerate(scores):
                l.append((docs2[idx], _score))
            pairs = []
            docs2 = []
    if len(pairs) > 0:
        sys.stderr = null_fp
        scores = reranker.compute_score(pairs, max_length=512, batch_size=batch_size, normalize=True, verbose=False)
        sys.stderr = sys.__stderr__
        for idx, _score in enumerate(scores):
            l.append((docs2[idx], _score))
        pairs = []
        docs2 = []

    l2 = []
    for item, (doc, _score) in zip(any_l, l):
        l2.append((item, _score))
        
    sorted_l = sorted(l2, key=lambda x: x[1], reverse=True)

    null_fp.close()
    
    return sorted_l
    
def rerank_by_dense_batch(reranker, query: str, sentences: list, top_k=10, batch_size=10):

    null_fp = open(os.devnull, 'w')
    sys.stderr = sys.__stderr__
    
    l = []
    pairs = []
    docs2 = []
    for sentence in sentences:
        pairs.append([query, sentence])
        docs2.append(sentence)
        if len(pairs) == batch_size:
            sys.stderr = null_fp
            scores = reranker.compute_score(pairs, max_length=1024, batch_size=batch_size, normalize=True, verbose=False)
            sys.stderr = sys.__stderr__
            for idx, _score in enumerate(scores):
                l.append((docs2[idx], _score))
            pairs = []
            docs2 = []
    if len(pairs) > 0:
        sys.stderr = null_fp
        scores = reranker.compute_score(pairs, max_length=1024, batch_size=batch_size, normalize=True, verbose=False)
        sys.stderr = sys.__stderr__
        for idx, _score in enumerate(scores):
            l.append((docs2[idx], _score))
        pairs = []
        docs2 = []
        
    sorted_l = sorted(l, key=lambda x: x[1], reverse=True)

    null_fp.close()
    
    return sorted_l[:top_k]
    
def rerank_by_dense_batch_chunked(reranker, query: str, documents: list, top_k=10, batch_size=10, chunk_size=256, overlap_size=64):
    
    null_fp = open(os.devnull, 'w')
    sys.stderr = sys.__stderr__

    l = []
    pairs = []
    docs2 = []
    for doc in documents:
        for chunk in text_chunk.chunk_with_sliding_window(doc['text'], chunk_size, overlap_size):
            pairs.append([query, doc['text']])
            docs2.append(doc)
        if len(pairs) >= batch_size:
            sys.stderr = null_fp
            scores = reranker.compute_score(pairs, max_length=1024, batch_size=batch_size, normalize=True, verbose=False)
            sys.stderr = sys.__stderr__
            for idx, _score in enumerate(scores):
                l.append((docs2[idx], _score))
            pairs = []
            docs2 = []
    if len(pairs) > 0:
        sys.stderr = null_fp
        scores = reranker.compute_score(pairs, max_length=1024, batch_size=batch_size, normalize=True, verbose=False)
        sys.stderr = sys.__stderr__
        for idx, _score in enumerate(scores):
            l.append((docs2[idx], _score))
        pairs = []
        docs2 = []

    sorted_l1 = sorted(l, key=lambda x: x[1], reverse=True)
    
    l2 = []
    seen_citation = set()
    for (doc, score) in sorted_l1:
        if doc['citation'] in seen_citation:
            continue
        else:
            l2.append((doc,score))
            seen_citation.add(doc['citation'])

    null_fp.close()
    
    return l2[:top_k]


from typing import List, Tuple, Any

def dedup_with_max_tuple(
    ids: List[int], 
    scores: List[Tuple[Any, float]]
) -> List[Tuple[Any, float]]:
    
    if not ids:
        return [], []
    
    new_ids = []
    new_scores = []
    
    cur_id = ids[0]
    cur_best = scores[0]  # (meta, score)
    
    for i in range(1, len(ids)):
        meta, score = scores[i]
        
        if ids[i] == cur_id:
            # 比较 score（tuple 第二个元素）
            if score > cur_best[1]:
                cur_best = scores[i]
        else:
            new_ids.append(cur_id)
            new_scores.append(cur_best)
            
            cur_id = ids[i]
            cur_best = scores[i]
    
    # 最后一组
    new_ids.append(cur_id)
    new_scores.append(cur_best)
    
    return new_scores
    
def rerank_by_batch_chunked2(reranker, query: str, documents: list, batch_size=10, chunk_size=382, overlap_size=124):
    '''
    len(ret) == len(documents)
    '''
    null_fp = open(os.devnull, 'w')
    sys.stderr = sys.__stderr__

    
    parent_idx_l = []
    l = []
    pairs = []
    docs2 = []
    for parent_idx, doc in enumerate(documents):
        for chunk in text_chunk.chunk_with_sliding_window(doc['text'], chunk_size, overlap_size):
            pairs.append([query, doc['text']])
            docs2.append(doc)
            parent_idx_l.append(parent_idx)
        if len(pairs) >= batch_size:
            sys.stderr = null_fp
            scores = reranker.compute_score(pairs, max_length=1024, batch_size=batch_size, normalize=True, verbose=False)
            sys.stderr = sys.__stderr__
            for idx, _score in enumerate(scores):
                l.append((docs2[idx], _score))
            pairs = []
            docs2 = []
    if len(pairs) > 0:
        sys.stderr = null_fp
        scores = reranker.compute_score(pairs, max_length=1024, batch_size=batch_size, normalize=True, verbose=False)
        sys.stderr = sys.__stderr__
        for idx, _score in enumerate(scores):
            l.append((docs2[idx], _score))
        pairs = []
        docs2 = []

    null_fp.close()

    doc_with_score_l = dedup_with_max_tuple(parent_idx_l, l)

    assert(len(doc_with_score_l) == len(documents))
    
    return doc_with_score_l

if __name__ == "__main__":
    ids = [1, 1, 2, 2, 2, 3]
    scores = [
        ("a", 0.5),
        ("b", 0.8),
        ("c", 0.3),
        ("d", 0.6),
        ("e", 0.4),
        ("f", 0.9)
    ]
    
    ret = dedup_with_max_tuple(ids, scores)
    
    print(ret)
        