import sys
import os
from tqdm import tqdm

import text_chunk

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