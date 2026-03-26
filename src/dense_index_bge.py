import os
import os.path
import numpy as np
import faiss
from typing import List, Tuple

class DenseIndex:
    def __init__(self, model, embeddings_path, documents):
        self.model = model # sentence_transformers

        embeddings, parent_indices = self._load_embedding(embeddings_path)

        print("DenseIndex.embeddings: ", embeddings.shape)
        
        dim = embeddings.shape[1]

        # =========================
        # 3. 构建 FAISS 索引
        # =========================
        # 因为做了 normalize，所以用 Inner Product 等价于 cosine
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        self.documents = documents
        self.parent_indices = parent_indices

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

        parent_idx_l = []
        parent_fn = fn = os.path.join(embeddings_path, f'parent.txt')
        with open(parent_fn) as inf:
            for line in inf:
                parent_idx_l.append(int(line.strip()))

        return np.vstack(embedding_l), parent_idx_l

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

        # query_embedding = np.array(query_embedding)
        query_embedding = query_encoded_result['dense_vecs']
        # print("query_embedding.shape:", query_embedding.shape)

        scores, indices = self.index.search(query_embedding, top_k)

        parent_indics = [self.parent_indices[idx] for idx in indices[0]]

        seen_parent_indics = set()
        parent_indics2 = []
        for parent_idx in parent_indics:
            if parent_idx in seen_parent_indics:
                pass
            else:
                seen_parent_indics.add(parent_idx)
                parent_indics2.append(parent_idx)
        
        ret = []
        for idx in parent_indics2:
            ret.append(self.documents[idx])
            
        return ret

    def __deduplicate_by_max_score(self, data: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        d = {}
        for i, score in data:
            if i not in d:
                d[i] = score
            elif score > d[i]:
                d[i] = score

        result = sorted([(i,score) for i,score in d.items()], key=lambda x: x[1], reverse=True)
        return result

    def search_with_score(self, q, top_k):
        query_encoded_result = self.model.encode(
            [q]
        )

        query_embedding = query_encoded_result['dense_vecs']

        scores, indices = self.index.search(query_embedding, top_k)

        parent_index_score_l = [(self.parent_indices[idx], scores[0][i]) for i, idx in enumerate(indices[0])]

        sorted_l = self.__deduplicate_by_max_score(parent_index_score_l)

        ret = [(self.documents[idx], score) for idx, score in sorted_l]

        return ret
    