import sys
import os
import pandas as pd

src_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)
    
import pandas as pd
import os
import os.path
import sys
from pathlib import Path
import numpy as np
from more_itertools import chunked
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm
import text_chunk
import pickle

import citation_utils

model = BGEM3FlagModel('/root/.cache/modelscope/hub/models/BAAI/bge-m3', use_fp16=True)

path_str = '../data/processed/_dense_court_removed_citation'
Path(path_str).mkdir(parents=True, exist_ok=True)

def __gen_lexical_weights(tokenizer, l):
    tokenizer = model.tokenizer
    
    ret = []
    for lexical_weights in l:
        token_weights = {}
        for k, v in lexical_weights.items():
            token_weights[tokenizer.convert_ids_to_tokens([k])[0].replace("▁", "")] = v
        ret.append(token_weights)

    return ret

def batch_calc(model,doc_l):
    null_fp = open(os.devnull, 'w')

    embeddings_l = []
    sparse_l = []
    chunks = list(chunked(doc_l, 10))
    
    for chunk in tqdm(chunks, total=len(chunks), desc="batch_calc_dense_sparse"):
        sys.stderr = null_fp
        encode_d = model.encode(chunk, 
                            batch_size=10, 
                            max_length=512,
                            return_dense=True, 
                            return_sparse=False, 
                            return_colbert_vecs=False,
                            verbose=False
                            )
        sys.stderr = sys.__stderr__

        embeddings_l.append(encode_d['dense_vecs'])
        # print(encode_d['lexical_weights'][0])
        # raise ValueError("debug")
        # for term in encode_d['lexical_weights']:
        # sparse_l.extend(__gen_lexical_weights(model.tokenizer, encode_d['lexical_weights']))
        
    null_fp.close()

    dense_return = np.vstack(embeddings_l)
    # sparse_return = sparse_l

    return dense_return, []

rows = []
row_count = 0
csv_path = '../data/court_considerations.csv'
csv = pd.read_csv(csv_path)
print("data loaded. court_considerations.len:", len(csv))

# 去除citation并拆分
text_l = []
parent_idx_l = []
for parent_idx, court_text in enumerate(csv['text'].tolist()):
    court_text = citation_utils.remove_citation_from_text(court_text)
    texts = text_chunk.chunk_with_sliding_window(court_text, 384, 128)
    text_l.extend(texts)
    for text in texts:
        parent_idx_l.append(parent_idx)

print("slice done. parent_idx_l.len:", len(parent_idx_l))

with open(os.path.join(path_str, f"parent.txt"), "w+") as of:
    for parent_idx in parent_idx_l:
        of.write(f'{parent_idx}\n')

chunked_list = list(chunked(text_l, 10000))
print("chunked_list.len:", len(chunked_list))

file_no=0
for rows in tqdm(chunked_list, total=len(chunked_list)):
    if os.path.exists(os.path.join(path_str, f"{file_no}.pkl")):
        file_no += 1
        continue

    embeddings, sparse_l = batch_calc(model, rows)
    
    print(embeddings.shape)
    np.save(os.path.join(path_str, f"{file_no}.npy"), embeddings)
        
    # with open(os.path.join(path_str, f"{file_no}.pkl"), "wb+") as of:
    #     pickle.dump(sparse_l, of)
    
    file_no += 1

    


    
# if __name__ == "__main__":
#     model = BGEM3FlagModel('/root/.cache/modelscope/hub/models/BAAI/bge-m3', use_fp16=True)
#     ret = __gen_lexical_weights(model, ['今天天气不错'])
#     print(ret)
        
        