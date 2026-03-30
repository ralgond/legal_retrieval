import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import pickle
import os
import os.path
import sys

src_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

import citation_utils

citation_2_cc_d = defaultdict(set)

cc_df = pd.read_csv("../data/court_considerations.csv", chunksize=100)
for idx,chunk in enumerate(cc_df):
    for _, row in chunk.iterrows():
        cc_citation = row['citation']
        cc_text = row['text']
        citations = citation_utils.extract_citations_from_text(cc_text)
        for c in citations:
            citation_2_cc_d[c].add(cc_citation)
    if idx%100 == 0:
        print("====>", idx*100)

with open("./data/citation_2_cc.pkl", "wb+") as of:
    pickle.dump(citation_2_cc_d, of)


        