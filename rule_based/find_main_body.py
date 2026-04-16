import re
from typing import Optional
from collections import defaultdict

class ErwExtractor:

    def __init__(self):
        # Erwägungen 开始标志
        self.start_patterns = [
            r"Erwägungen?:?",                 # 标准
            r"Consid[eé]rants?:?",            # 法语
            r"Considerazioni?:?",             # 意大利语
        ]

        # 结束标志（判决结果）
        self.end_patterns = [
            r"Demnach erkennt",               # 最常见
            r"Par ces motifs",                # 法语
            r"Per questi motivi",             # 意大利语
            r"Das Bundesgericht erkennt",     # 变体
        ]

        self.start_regex = re.compile(
            "|".join(self.start_patterns),
            re.IGNORECASE
        )
        self.end_regex = re.compile(
            "|".join(self.end_patterns),
            re.IGNORECASE
        )

    def extract(self, text: str) -> Optional[str]:
        """
        抽取 Erwägungen 主体
        """
        if not text:
            return None, None, None

        # 标准化
        text = text.replace("\r", "\n")

        # 找 start
        start_match = self.start_regex.search(text)
        if not start_match:
            return None, None, None

        start_idx = start_match.end()

        # 找 end（从 start 之后找）
        end_match = self.end_regex.search(text[start_idx:])
        if end_match:
            end_idx = start_idx + end_match.start()
        else:
            end_idx = len(text)

        erw = text[start_idx:end_idx].strip()

        left_not_erw = text[:start_idx]
        right_not_erw =  text[end_idx:]

        if erw:
            return erw, left_not_erw, right_not_erw 
        else:
            return None, None, None

import pandas as pd
from tqdm import tqdm
court_consideration_df = pd.read_csv("../data/court_considerations.csv")

erw_words = defaultdict(int)
not_erw_words = defaultdict(int)
count = 0
extractor = ErwExtractor()
for text in tqdm(court_consideration_df['text'], total=len(court_consideration_df)):
    erw_segment, left_not_erw_segment, right_not_erw_segment = extractor.extract(text)
    if erw_segment is not None:
        for word in erw_segment.split():
            word = word.lower()
            word = word.strip(';,.():-')
            if len(word) > 0 and word[0].isdigit():
                continue
            erw_words[word] += 1
        count += 1

        for word in left_not_erw_segment.split():
            word = word.lower()
            word = word.strip(';,.():-')
            if len(word) > 0 and word[0].isdigit():
                continue
            not_erw_words[word] += 1
            
        for word in right_not_erw_segment.split():
            word = word.lower()
            word = word.strip(';,.():-')
            if len(word) > 0 and word[0].isdigit():
                continue
            not_erw_words[word] += 1
print(f"{count}/{len(court_consideration_df)}")


import math
def compute_log_odds(erw_words, not_erw_words, alpha=1.0):
    V = set(erw_words) | set(not_erw_words)

    N_erw = sum(erw_words.values())
    N_not = sum(not_erw_words.values())

    scores = {}

    for w in V:
        p_erw = (erw_words.get(w, 0) + alpha) / (N_erw + alpha * len(V))
        p_not = (not_erw_words.get(w, 0) + alpha) / (N_not + alpha * len(V))
        scores[w] = math.log(p_erw / p_not)

    return scores

scores = compute_log_odds(erw_words, not_erw_words)
top_terms = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:100]
print(top_terms)
top_terms = set([w for w, _ in top_terms])

