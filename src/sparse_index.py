import os
import os.path
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import re
import Stemmer
import text_chunk
from sparse_engine import SparseSearchEngine
from typing import List, Tuple

class SparseIndex:
    def __init__(self, model, work_path: str, parent_documents: list[dict]):
        self.model = model
        self.parent_documents = parent_documents
        self.parent_indices = []
        self.work_path = work_path
        self.matrix_path = os.path.join(self.work_path, 'matrix.npz')
        self.vocab_path = os.path.join(self.work_path, 'vocab.txt')
        self.doc_ids_path = os.path.join(self.work_path, "doc_ids.txt")
        self.engine = SparseSearchEngine(self.work_path)

    def __load_sparse_dict(self, sparse_dict_path: str):
        ret = []
        i = 0
        while True:
            fn = os.path.join(sparse_dict_path, f"{i}.pkl")
            if not os.path.exists(fn):
                break
            with open(fn, 'rb') as inf:
                ret.extend(pickle.load(inf))
            i += 1
        return ret
            

    def load(self):
        with open(os.path.join(self.work_path, 'parent.txt'), 'r') as inf:
            for line in inf:
                self.parent_indices.append(int(line.strip()))
        
        if not os.path.exists(self.matrix_path) or not os.path.exists(self.vocab_path) or not os.path.exists(self.doc_ids_path):
            sparse_dict_l = self.__load_sparse_dict(self.work_path)
            print("loaded, sparse_dict_l.len:", len(sparse_dict_l))
            self.engine.build_index_by_dict_list(sparse_dict_l)
            self.engine.save()

        self.engine.load()

    def __gen_lexical_weights(self, texts):
        l = self.model.encode(texts, 
                         batch_size=10, 
                         return_dense=False, 
                         return_sparse=True, 
                         return_colbert_vecs=False)['lexical_weights']
        tokenizer = self.model.tokenizer
        ret = []
        for lexical_weights in l:
            token_weights = {}
            for k, v in lexical_weights.items():
                token_weights[tokenizer.convert_ids_to_tokens([k])[0].replace("▁", "")] = v
            ret.append(token_weights)
        return ret

    def search(self, query: str, top_k=10):
        q = self.__gen_lexical_weights([query])

        res_l = self.engine.search(q[0], top_k) # [(ids, score)]

        ret_doc_l = []
        seen_parent_idx_set = set([self.parent_indices[doc_id] for doc_id,score in res_l])

        for parent_idx in seen_parent_idx_set:
            ret_doc_l.append(self.parent_documents[parent_idx])

        return ret_doc_l

    def __deduplicate_by_float(self, data: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        # 用 float 作为 key 去重
        d = {}
        for i, f in data:
            if f not in d:
                d[f] = (i, f)
    
        # 按 float 逆序排序
        result = sorted(d.values(), key=lambda x: x[1], reverse=True)
        return result

    def __deduplicate_by_max_score(self, data: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        d = {}
        for i, score in data:
            if i not in d:
                d[i] = score
            elif score > d[i]:
                d[i] = score

        result = sorted([(i,score) for i,score in d.items()], key=lambda x: x[1], reverse=True)
        return result
        
    def search_with_score(self, query, top_k=10):
        q = self.__gen_lexical_weights([query])

        res_l = self.engine.search(q[0], top_k) # [(ids, score)]

        parent_index_score_l = [(self.parent_indices[idx], score) for idx, score in res_l]

        sorted_l = self.__deduplicate_by_max_score(parent_index_score_l)

        ret = [(self.parent_documents[idx], score) for idx, score in sorted_l]

        return ret

        
        

    
    
    
        
        
        