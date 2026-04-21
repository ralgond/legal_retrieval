import os
import os.path
import numpy as np

def load_embedding(embeddings_path):
    embedding_l = []
    i = 0
    while True:
        fn = os.path.join(embeddings_path, f'{i}.npy')
        if not os.path.exists(fn):
            break
        embedding = np.load(fn)
        embedding_l.append(embedding)
        i += 1

    parent_idx_l = []
    parent_fn = fn = os.path.join(embeddings_path, f'parent.txt')
    with open(parent_fn) as inf:
        for line in inf:
            parent_idx_l.append(int(line.strip()))

    return np.vstack(embedding_l), parent_idx_l