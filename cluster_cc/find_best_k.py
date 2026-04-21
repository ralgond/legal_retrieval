import faiss
import os
import os.path
import numpy as np
from tqdm import tqdm

from common import load_embedding

embeddings, parent_idx_l = load_embedding("/root/autodl-fs/bge-processed/_dense_sparse_court/")
embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
faiss.normalize_L2(embeddings)
print("embedding loaded.")

inertias = []
k_values = [50, 100, 200, 300, 500, 1000]

for k in k_values:
    kmeans = faiss.Kmeans(d=1024, k=k, gpu=True, niter=20)
    kmeans.train(embeddings)
    distances, _ = kmeans.index.search(embeddings, 1)
    inertia = distances.sum()
    inertias.append(inertia)
    print(f"k={k}, inertia={inertia:.2f}")