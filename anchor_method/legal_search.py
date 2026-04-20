"""
瑞士法律判决句语义检索系统
模型：BAAI/bge-m3
功能：
  - build_index(): 从JSONL读取判决句，建立FAISS索引
  - LegalSearchEngine.search(): 输入自然语言问题，返回最相关的citation列表
依赖：
  pip install FlagEmbedding faiss-cpu
"""

import json
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from FlagEmbedding import BGEM3FlagModel

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────

MODEL_NAME  = "BAAI/bge-m3"
INDEX_FILE  = "../data/anchor_method/decisions.faiss"
META_FILE   = "../data/anchor_method/decisions_meta.jsonl"
INPUT_JSONL = "../data/anchor_method/erwaegungen.jsonl"


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class SearchResult:
    rank: int
    citation: str
    sentence: str
    score: float

    def __str__(self):
        return (
            f"{self.rank}. {self.citation}  (相似度 {self.score:.3f})\n"
            f"   {self.sentence}\n"
        )


# ─────────────────────────────────────────────
# 句子切分
# ─────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    import re
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 30]


# ─────────────────────────────────────────────
# 离线：建索引
# ─────────────────────────────────────────────

def build_index(
    input_jsonl: str = INPUT_JSONL,
    index_file: str  = INDEX_FILE,
    meta_file: str   = META_FILE,
    batch_size: int  = 32,
    use_core_sentences: Optional[list[tuple[str, str]]] = None,
):
    """
    两种输入模式：
    1. use_core_sentences: [(citation, sentence), ...] 直接传入核心判决句
    2. 否则从 input_jsonl 读取，自动切句
    """
    print(f"Loading model: {MODEL_NAME}")
    model = BGEM3FlagModel(MODEL_NAME, use_fp16=True)  # fp16省内存

    sentences = []
    citations = []

    if use_core_sentences is not None:
        for citation, sentence in use_core_sentences:
            sentences.append(sentence)
            citations.append(citation)
    else:
        print(f"Reading from {input_jsonl}")
        with open(input_jsonl, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if not row.get("success") or not row.get("erwaegungen"):
                    continue
                citation = row["citation"]
                for sent in split_sentences(row["erwaegungen"]):
                    sentences.append(sent)
                    citations.append(citation)

    print(f"Total sentences to index: {len(sentences)}")
    print("Encoding sentences...")

    # BGE-M3 返回字典，取 dense_vecs
    output = model.encode(
        sentences,
        batch_size=batch_size,
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    embeddings = np.array(output["dense_vecs"], dtype=np.float32)

    # 归一化，使内积等价于余弦相似度
    faiss.normalize_L2(embeddings)

    # 建FAISS索引
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    Path(index_file).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_file)
    print(f"FAISS index saved: {index_file}  ({index.ntotal} vectors)")

    with open(meta_file, "w", encoding="utf-8") as f:
        for citation, sentence in zip(citations, sentences):
            f.write(json.dumps({"citation": citation, "sentence": sentence}, ensure_ascii=False) + "\n")
    print(f"Metadata saved: {meta_file}")


# ─────────────────────────────────────────────
# 在线：查询
# ─────────────────────────────────────────────

class LegalSearchEngine:
    def __init__(
        self,
        index_file: str = INDEX_FILE,
        meta_file: str  = META_FILE,
    ):
        print(f"Loading model: {MODEL_NAME}")
        self.model = BGEM3FlagModel(MODEL_NAME, use_fp16=True)

        print(f"Loading FAISS index: {index_file}")
        self.index = faiss.read_index(index_file)

        print(f"Loading metadata: {meta_file}")
        self.meta = []
        with open(meta_file, encoding="utf-8") as f:
            for line in f:
                self.meta.append(json.loads(line))

        print(f"Ready. {self.index.ntotal} sentences indexed.")

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        output = self.model.encode(
            [query],
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        q_vec = np.array(output["dense_vecs"], dtype=np.float32)
        faiss.normalize_L2(q_vec)

        scores, indices = self.index.search(q_vec, top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            if idx == -1:
                continue
            meta = self.meta[idx]
            results.append(SearchResult(
                rank=rank,
                citation=meta["citation"],
                sentence=meta["sentence"],
                score=float(score),
            ))
        return results

    def print_results(self, query: str, top_k: int = 10):
        print(f"\n{'='*60}")
        print(f"Frage: {query}")
        print(f"{'='*60}")
        for r in self.search(query, top_k):
            print(r)


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Step 1: 建索引（只需运行一次）──────────────
    #
    # 选项A：直接用144个核心判决句
    # core_sentences = [
    #     ("BGE 138 III 123", "Der Arbeitgeber ist verpflichtet..."),
    #     ("BGer 4A_456/2021", "Eine vertragliche Abrede kann..."),
    # ]
    # build_index(use_core_sentences=core_sentences)
    #
    # 选项B：从完整Erwägungen JSONL自动切句
    # build_index()

    core_sent_df = pd.read_csv("core_sentence.csv")
    core_sentences = []
    for citation, text in zip(core_sent_df['citation'], core_sent_df['text']):
        core_sentences.append((citation, text))
    build_index(use_core_sentences=core_sentences)

    # ── Step 2: 查询 ────────────────────────────────
    engine = LegalSearchEngine()

    # 单次查询
    engine.print_results("Muss mein Chef mir Überstunden bezahlen?", top_k=5)

    # 交互模式
    print("\n--- Interaktiver Modus (Eingabe 'quit' zum Beenden) ---")
    while True:
        query = input("\nFrage: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if query:
            engine.print_results(query, top_k=5)