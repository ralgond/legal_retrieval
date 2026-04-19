import spacy
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

nlp = spacy.load("de_core_news_lg")

text = "Gemäss CITE_START_Art._228_Abs._1_StPO_CITE_END ist der Vertrag gültig."

doc = nlp(text)

for token in doc:
    print(
        token.text,
        token.dep_,
        token.head.text,
        token.pos_
    )


text = "Gemäss CITE_START_Art._228_Abs._1_StPO_CITE_END. ist der Vertrag gültig."

print("="*60)
for s in citation_utils.p_split_sentences(text):
    print(s)