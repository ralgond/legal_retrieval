from __future__ import annotations
import math
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import re
from typing import List, Optional

path_str = '../data/processed/new_index2'
Path(path_str).mkdir(parents=True, exist_ok=True)

src_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

import citation_utils

cc_df = pd.read_csv("../data/court_considerations.csv")
print("cc_df loaded.")

import re
from typing import List, Optional

def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|\S", text)


def get_window_tokens(
    text: str,
    citation: str,
    window_size: int = 128
) -> Optional[str]:
    """
    基于 token 的窗口截取（更适合法律检索）
    """

    tokens = simple_tokenize(text)
    citation_tokens = simple_tokenize(citation)

    # 找 citation token 的起点
    for i in range(len(tokens)):
        if tokens[i:i+len(citation_tokens)] == citation_tokens:
            cite_start = i
            cite_end = i + len(citation_tokens)
            break
    else:
        return None

    half = window_size // 2

    start = max(0, cite_start - half)
    end = min(len(tokens), cite_end + half)

    window_tokens = tokens[start:end]

    return " ".join(window_tokens)

corpus_of = open("../data/new_index2/corpus.txt", "w+")

for text in tqdm(cc_df['text'], total=len(cc_df)):
    citations = citation_utils.extract_citations_from_text(text)
    
    for citation in citations:
        window_set = set()
        for window_size in [128, 256]:
            s = get_window_tokens(text, citation, window_size)
            if s is not None:
                window_set.add(s)
        for s in window_set:
            corpus_of.write(s+"\n")

corpus_of.close()
        

    
