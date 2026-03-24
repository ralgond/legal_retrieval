import pandas as pd
import os
import os.path
import sys
from pathlib import Path
import numpy as np
from more_itertools import chunked
from tqdm import tqdm
import text_chunk
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("/root/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0___6B", model_kwargs={"torch_dtype": "float16"})

path_str = '../data/processed/_dense_court'
Path(path_str).mkdir(parents=True, exist_ok=True)

def batch_calc_dense(doc_l):
    embeddings_l = []
    chunks = list(chunked(doc_l, 10))
    for chunk in tqdm(chunks, total=len(chunks), desc="batch_calc_dense"):
        embeddings = model.encode(chunk, 
                            batch_size=10, 
                            normalize_embeddings=True,
                            show_progress_bar=False
                            )
        embeddings_l.append(embeddings)

    return np.vstack(embeddings_l)

rows = []
file_no = 0
row_count = 0
csv_path = '../data/court_considerations.csv'
csv = pd.read_csv(csv_path)
print("data loaded.")

# 拆分
text_l = []
parent_idx_l = []
for parent_idx, court_text in enumerate(csv['text'].tolist()):
    texts = text_chunk.chunk_with_sliding_window(court_text, 384, 128)
    text_l.extend(texts)
    for text in texts:
        parent_idx_l.append(parent_idx)

print("slice done.")

with open(os.path.join(path_str, f"parent.txt"), "w+") as of:
    for parent_idx in parent_idx_l:
        of.write(f'{parent_idx}\n')

chunked_list = list(chunked(text_l, 10000))
print("chunked_list.len:", len(chunked_list))
       
for rows in tqdm(chunked_list, total=len(chunked_list)):
    if os.path.exists(os.path.join(path_str, f"{file_no}.npy")):
        file_no += 1
        continue
    embeddings = batch_calc_dense(rows)
    print(embeddings.shape)
    np.save(os.path.join(path_str, f"{file_no}.npy"), embeddings)
    rows = []
    file_no += 1
