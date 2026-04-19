from __future__ import annotations
import math
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import random
from tqdm import tqdm

src_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

import citation_utils

# ── 加载数据 ──────────────────────────────────────────────────────────────────
print("Loading data...")
court_consideration_df = pd.read_csv("../data/court_considerations.csv")

citation_l = []
text_l = []
for citation, text in tqdm(zip(court_consideration_df['citation'], court_consideration_df['text']), total=len(court_consideration_df)):
    cl = citation_utils.extract_citations_from_text(text)
    for c in cl:
        text = text.replace(c, citation_utils.map_citation_2_pcitation(c))
    citation_l.append(citation)
    text_l.append(text)

res_df = pd.DataFrame({'citation':citation_l, 'text':text_l})
res_df.to_csv("../data/court_considerations_maped.csv")