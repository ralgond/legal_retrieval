import os
import os.path
import sys
import numpy as np
import math
from collections import defaultdict
import pandas as pd
from tqdm import tqdm


src_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

import citation_utils

court_consideration_df = pd.read_csv("../data/court_considerations.csv")

citation_freq = defaultdict(int)

def gen_citation_idf():
    for text in court_consideration_df['text']:
        parsed_cc = citation_utils.parse_cc_output_citations_and_sentences(text)
        for citation, pos in parsed_cc['citations']:
            citation_freq[citation] += 1

    N = len(court_consideration_df)

    
    citation_idf_list = []
    for citation, freq in citation_freq.items():
        citation_idf_list.append((citation, math.log(N/(1+freq))))

    citation_idf_list = sorted(citation_idf_list, key=lambda x: x[1])

    df = pd.DataFrame({"citation":[item[0] for item in citation_idf_list], "idf":[item[1] for item in citation_idf_list]})

    df.to_csv("../data/idf.csv", index=False)

gen_citation_idf()

