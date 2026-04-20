from __future__ import annotations
import math
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm
from pathlib import Path
from more_itertools import chunked

model = BGEM3FlagModel('/root/.cache/modelscope/hub/models/BAAI/bge-m3', use_fp16=True)
path_str = '../data/processed/new_index_fact'
Path(path_str).mkdir(parents=True, exist_ok=True)

src_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

import citation_utils

cc_df = pd.read_csv("../data/court_considerations.csv")
print("cc_df loaded.")

def has_citation(sentence):
    return len(citation_utils.extract_citations_from_text(sentence)) > 0

def get_fact(text):
    sentences = citation_utils.split_sentences(text)
    s_l = []
    for s in sentences:
        if has_citation(s):
            break
        else:
            s_l.append(s)

    return ' '.join(s_l)

def batch_calc(model, doc_l):
    null_fp = open(os.devnull, 'w')

    embeddings_l = []
    chunks = list(chunked(doc_l, 10))
    
    for chunk in tqdm(chunks, total=len(chunks), desc="batch_calc_dense_sparse"):
        sys.stderr = null_fp
        encode_d = model.encode(chunk, 
                            batch_size=10, 
                            max_length=256,
                            return_dense=True, 
                            return_sparse=False, 
                            return_colbert_vecs=False,
                            verbose=False
                            )
        sys.stderr = sys.__stderr__

        embeddings_l.append(encode_d['dense_vecs'])
        
    null_fp.close()

    dense_return = np.vstack(embeddings_l)

    return dense_return

text_l = []
for citation, text in tqdm(zip(cc_df['citation'], cc_df['text']), total=len(cc_df)):
    fact = get_fact(text)
    if len(fact) == 0:
        text_l.append('none')
    else:
        text_l.append(fact)

chunked_list = list(chunked(text_l, 10000))
print("chunked_list.len:", len(chunked_list))

file_no=0
for rows in tqdm(chunked_list, total=len(chunked_list)):
    if os.path.exists(os.path.join(path_str, f"{file_no}.pkl")):
        file_no += 1
        continue

    embeddings = batch_calc(model, rows)
    
    print(embeddings.shape)
    np.save(os.path.join(path_str, f"{file_no}.npy"), embeddings)
    
    file_no += 1