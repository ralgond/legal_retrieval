import os
import os.path
import numpy as np
import faiss
from typing import List, Tuple

class DenseIndex:
    def __init__(self, model, embeddings_path, documents):
        self.model = model # sentence_transformers

        embeddings = self._load_embedding(embeddings_path)

        print("DenseIndex.embeddings: ", embeddings.shape)
        
        dim = embeddings.shape[1]

        # =========================
        # 3. 构建 FAISS 索引
        # =========================
        # 因为做了 normalize，所以用 Inner Product 等价于 cosine
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        self.documents = documents

    def info(self):
        print("[dense_index] documents.len:",len(self.documents), "parent_idx.len:", len(self.parent_indices))

    def _load_embedding(self, embeddings_path):
        embedding_l = []
        i = 0
        while True:
            fn = os.path.join(embeddings_path, f'{i}.npy')
            if not os.path.exists(fn):
                break
            embedding = np.load(fn)
            embedding_l.append(embedding)
            i += 1

        return np.vstack(embedding_l)

    def search(self, q, top_k):
        '''
        return: list of index of embeddings
        '''
        # =========================
        # 4. 查询
        # =========================
        query_encoded_result = self.model.encode(
            [q],
        )

        query_embedding = query_encoded_result['dense_vecs']

        scores, indices = self.index.search(query_embedding, top_k)
        
        ret = []
        for idx in indices[0]:
            ret.append(self.documents[idx].copy())
            
        return ret