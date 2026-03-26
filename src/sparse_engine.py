import numpy as np
from scipy import sparse
from scipy.sparse import save_npz, load_npz
import time
import os
import os.path

class SparseSearchEngine:
    def __init__(self, work_dir="./sparse-index", dtype=np.float32):
        self.doc_matrix = None  # CSR Matrix: (n_docs, n_terms)
        self.doc_ids = None     # 映射矩阵行号到实际文档 ID
        self.dtype = dtype
        self.work_dir = work_dir
        self.vocab = None
        self.doc_matrix = None
        os.makedirs(self.work_dir, exist_ok=True)

    def save(self):
        with open(os.path.join(self.work_dir, "vocab.txt"), "w+") as of:
            for k, v in self.vocab.items():
                of.write(f"{k}\t{v}\n")

        save_npz(os.path.join(self.work_dir, "matrix.npz"), self.doc_matrix)

        with open(os.path.join(self.work_dir, "doc_ids.txt"), "w+") as of:
            for id in self.doc_ids:
                of.write(f"{id}\n")

    def load(self):
        vocab_path = os.path.join(self.work_dir, "vocab.txt")
        matrix_path = os.path.join(self.work_dir, "matrix.npz")
        doc_ids_path = os.path.join(self.work_dir, "doc_ids.txt")

        self.vocab = {}
        self.doc_ids = []
        if os.path.exists(vocab_path) and os.path.exists(matrix_path) and os.path.exists(doc_ids_path):
            with open(vocab_path, "r") as inf:
                for line in inf:
                    k,v = line.strip().split('\t')
                    self.vocab[k] = int(v)

            self.doc_matrix = load_npz(matrix_path)

            with open(doc_ids_path, "r") as inf:
                for line in inf:
                    id = int(line.strip())
                    self.doc_ids.append(id)
        
    def build_index_by_dict_list(self, dict_list, doc_ids=None):
        """
        将 List[Dict] 高效转换为 scipy.sparse.csr_matrix
        :param dict_list: List[Dict[str, float]], 例如 [{'A': 0.1, 'B': 0.2}, ...]
        :param dtype: 矩阵数据类型，推荐 np.float32 以节省内存
        :return: csr_matrix, vocab_dict (Term -> ColIndex)
        """
        for d in dict_list:
            if '' in d:
                del d['']
        n_docs = len(dict_list)
        
        # --- 第一步：构建词汇表 (Term -> ID) ---
        # 为了性能，我们先遍历一次收集所有唯一词，避免在填充数据时动态扩展 dict
        vocab = {}
        next_id = 0
        # 使用 set 去重加速，然后再转 dict 分配 ID
        all_terms = set()
        for d in dict_list:
            all_terms.update([k for k in d.keys() if len(k) > 0])
        
        # 分配 ID (这里可以按字母排序，也可以随机，不影响检索效果)
        for term in all_terms:
            vocab[term] = next_id
            next_id += 1
        
        n_terms = len(vocab)
        print(f"Vocabulary size: {n_terms}")
    
        # --- 第二步：预估算非零元素总数 (nnz) ---
        # 预分配 numpy 数组比 append 到 list 更快且内存更连续
        # 如果数据量极大，这一步遍历开销可接受，能避免 list 动态扩容
        nnz = sum(len(d) for d in dict_list)
        print(f"Total non-zeros: {nnz}")
        
        rows = np.empty(nnz, dtype=np.int32)
        cols = np.empty(nnz, dtype=np.int32)
        data = np.empty(nnz, dtype=self.dtype)
        
        # --- 第三步：填充数据 ---
        current_idx = 0
        for i, d in enumerate(dict_list):
            # 获取该行非零元素个数
            count = len(d)
            if count == 0:
                continue
                
            # 切片范围
            start = current_idx
            end = current_idx + count
            
            # 填充行索引 (该行所有元素的行号都是 i)
            rows[start:end] = i
            
            # 填充列索引和数据
            # 列表推导式比循环 append 快
            cols[start:end] = [vocab[k] for k in d.keys()]
            data[start:end] = list(d.values())
            
            current_idx = end
    
        # --- 第四步：构建 COO 矩阵并转换为 CSR ---
        # COO 格式构建非常快，tocsr() 是高度优化的 C 代码
        coo = sparse.coo_matrix((data, (rows, cols)), shape=(n_docs, n_terms), dtype=self.dtype)
        
        # 转换为 CSR 以支持快速行切片和矩阵乘法
        csr = coo.tocsr()
        
        # 清理中间变量释放内存
        del rows, cols, data, coo

        # return csr, vocab
        self.vocab = vocab
        self.__build_index(csr)

        # self.save()

    def __build_index(self, doc_vectors, doc_ids=None):
        """
        构建索引
        :param doc_vectors: scipy.sparse.csr_matrix, shape (n_docs, n_terms)
        :param doc_ids: list, 文档唯一标识
        """
        # 1. 确保格式为 CSR 且类型为 self.dtype
        if not sparse.isspmatrix_csr(doc_vectors):
            doc_vectors = doc_vectors.tocsr()
        
        self.doc_matrix = doc_vectors.astype(self.dtype)
        
        self.doc_ids = doc_ids if doc_ids is not None else [id for id in range(self.doc_matrix.shape[0])]
        print(f"Index built: {self.doc_matrix.shape[0]} docs, {self.doc_matrix.nnz} non-zeros")

    def _dict_to_vector(self, query_dict):
        """
        将 dict 查询转换为 dense vector (内部使用)
        """
        vector = np.zeros(len(self.vocab), dtype=self.dtype)
        for term, score in query_dict.items():
            if term in self.vocab:
                col_idx = self.vocab[term]
                vector[col_idx] = score
        return vector

    def search(self, query_dict: dict, top_k=10):
        """
        单次查询
        :param query_dict: dict, 如{'A':0.1}
        :param top_k: 返回结果数量
        :return: list of (doc_id, score)
        """
        if '' in query_dict:
            del query_dict['']
            
        query_vector = self._dict_to_vector(query_dict)

        # 2. 核心计算：矩阵向量乘法 (Scores = Docs @ Query)
        # SciPy CSR @ dense vector 是非常优化的操作
        scores = self.doc_matrix.dot(query_vector)
        
        # 3. 获取 Top-K (使用 argpartition 避免全排序 O(N) -> O(N))
        n_docs = scores.shape[0]
        if top_k >= n_docs:
            best_indices = np.argsort(scores)[::-1]
        else:
            # argpartition 返回的是前 k 大的索引，但顺序未定，需再排序
            partition_indices = np.argpartition(scores, -top_k)[-top_k:]
            # 对这 k 个索引按分数排序
            best_indices = partition_indices[np.argsort(scores[partition_indices])[::-1]]
            
        # 4. 过滤零分 (可选，视业务而定)
        # 5. 映射回 Doc ID
        results = []
        for idx in best_indices:
            score = scores[idx]
            if score > 1e-6: # 忽略极小分数
                results.append((self.doc_ids[idx], float(score)))
                
        return results

if __name__ == "__main__":
    l = [
            {'A':0.1, 'B':0.2, 'C':0.3},
            {'C':0.1, 'D':0.2, 'E':0.3},
            {'E':0.1, 'F':0.2, 'G':0.3},
            {'G':0.1, 'H':0.2, 'I':0.3},
            {'I':0.1, 'J':0.2, 'K':0.3},
        ]
    engine = SparseSearchEngine()
    engine.build_index_by_dict_list(l)
    for k in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']:
        query = {k:0.2}
        print(k, engine.search(query))

    print()
    
    engine.save()
    engine = SparseSearchEngine()
    engine.load()
    for k in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']:
        query = {k:0.2}
        print(k, engine.search(query))
    