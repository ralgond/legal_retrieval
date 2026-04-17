import pandas as pd

import re
import math
import random
from collections import defaultdict
from typing import List, Dict, Tuple
import pandas as pd
import os
import os.path
import sys
import numpy as np

src_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

import reranker_utils
import hits_utils
import citation_utils
import metric_utils

train_df = pd.read_csv("../data/court_considerations.csv")
print("data loaded.")

top10_text = []
for idx,row in train_df.iterrows():
    top10_text.append(row['text'])
    if idx == 9:
        break

import spacy
from swiss_legal_citation_analyzer import analyze_text, print_report

nlp = spacy.load("de_core_news_lg")
# text = '''Nach BGE 140 III 115 ist die Klage begründet; vgl. auch BGer 4A_312/2019. Die ältere Rechtsprechung gemäss BGE 120 II 20 ist überholt und kann nicht mehr herangezogen werden.'''
# text = '''Gemäss Art. 41 Abs. 1 OR haftet der Schuldner für jeden Schaden, wobei Art. 44 OR sinngemäss gilt, insbesondere aber nicht Art. 43 Abs. 2 OR, da diese Bestimmung hier nicht anwendbar ist.'''
for text in top10_text:
    doc, results = analyze_text(text, nlp, verbose=True)

    citation_l = citation_utils.extract_citations_from_text(text)

    print("==>result.len:", len(results), "citation_count.len:", len(citation_l), citation_l)
    # if len(results) > 0:
    #     result = results[0]
    #     #print(result.citation.start,result.citation.end, len(doc))
    #     print(text[result.citation.start:result.citation.end], result.score)
    #     # print_report(results)