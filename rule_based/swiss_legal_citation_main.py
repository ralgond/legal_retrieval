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

print("Loading data...")
court_consideration_df = pd.read_csv("../data/court_considerations.csv")
court_consideration_d = dict(zip(court_consideration_df['citation'].tolist(), court_consideration_df['text'].tolist()))

import common
train_candidate_d = common.read_candidate("../data/rule_based/raw_train_candidate.pkl", court_consideration_d)
valid_candidate_d = common.read_candidate("../data/rule_based/raw_valid_candidate.pkl", court_consideration_d)
test_candidate_d = common.read_candidate("../data/rule_based/raw_test_candidate.pkl", court_consideration_d)



import spacy
from swiss_legal_citation_analyzer import analyze_text, print_report
from swiss_legal_citation_aggregator import from_analyzer_results, aggregate, print_aggregation_report

nlp = spacy.load("de_core_news_lg")
# text = '''Nach BGE 140 III 115 ist die Klage begründet; vgl. auch BGer 4A_312/2019. Die ältere Rechtsprechung gemäss BGE 120 II 20 ist überholt und kann nicht mehr herangezogen werden.'''
# text = '''Gemäss Art. 41 Abs. 1 OR haftet der Schuldner für jeden Schaden, wobei Art. 44 OR sinngemäss gilt, insbesondere aber nicht Art. 43 Abs. 2 OR, da diese Bestimmung hier nicht anwendbar ist.'''

valid_df = pd.read_csv("../data/valid_rewrite_001.csv")
gold_d = {query_id:gold_citations.split(';') for query_id, gold_citations in zip(valid_df['query_id'], valid_df['gold_citations'])}

idf_df = pd.read_csv("../data/idf.csv")
idf_d = {citation:idf for citation,idf in zip(idf_df['citation'],idf_df['idf'])}

result_l = []
gold_l = []
# for query_id, d in valid_candidate_d.items():
#     # print(query_id, len(d['rerank']))
#     cc_list = sorted(d['rerank'], key=lambda x: x[1], reverse=True)
#     entry_list = []
#     for cc,score in cc_list:
#         cc_id = cc['citation']
#         cc_text = cc['text']
#         doc, results = analyze_text(cc_text, nlp)
#         entry = from_analyzer_results(cc_id, score, results)  # 适配器
#         entry_list.append(entry)

#     final = aggregate(entry_list)

#     # print_aggregation_report(final)
#     citations = [r.citation for r in final]
#     result_l.append(citations)
#     gold_l.append(gold_d[query_id])


from collections import defaultdict

for query_id, d in valid_candidate_d.items():
    citation_d = dict()
    cc_list = sorted(d['rerank'], key=lambda x: x[1], reverse=True)
    total_sum = sum([score for _,score in cc_list])
    weights = [score/total_sum for _,score in cc_list]
    for idx, (cc, score) in enumerate(cc_list):
        weight = weights[idx]
        cc_id = cc['citation']
        cc_text = cc['text']
        doc, results = analyze_text(cc_text, nlp)
        for r in results:
            if r.citation.original not in citation_d:
                citation_d[r.citation.original] = r.score * weight * idf_d.get(r.citation.original, 1.)
            # elif citation_d[r.citation.original] < r.score * weight:
            #     citation_d[r.citation.original] = r.score * weight
            else:
                citation_d[r.citation.original] += r.score * weight * idf_d.get(r.citation.original, 1.)

    # print("====>citation_d.size:", len(citation_d))

    l = [(citation,score) for citation,score in citation_d.items()]
    l.sort(key=lambda x: x[1], reverse=True)

    result_l.append([citation for citation,_ in l])
    gold_l.append(gold_d[query_id])


    
# ── 评估 ──────────────────────────────────────────────────────────────────────
for TOP_K in [5,7,10,12,15,17,20,22,25,27,30,33,35,37,40]:
    result_l2 = [r[:TOP_K] for r in result_l]
    recall    = metric_utils.cal_recall(result_l2, gold_l)
    precision = metric_utils.cal_precision(result_l2, gold_l)
    print(f"[{TOP_K}] Recall@{TOP_K}:{recall:.4f}, Precision:{precision:.4f}, F1:{2*recall*precision/(recall+precision):.4f}")
    

os._exit(0)

for text in top10_text:
    doc, results = analyze_text(text, nlp, verbose=True)

    citation_l = citation_utils.extract_citations_from_text(text)

    print("==>result.len:", len(results), "citation_count.len:", len(citation_l), citation_l)
    # if len(results) > 0:
    #     result = results[0]
    #     #print(result.citation.start,result.citation.end, len(doc))
    #     print(text[result.citation.start:result.citation.end], result.score)
    #     # print_report(results)

def aggregate_scores(scores):
    '''
    一段文本中出现多个相同的citation
    '''
    max_score = max(scores)
    min_score = min(scores)

    has_positive = any(s >= 2 for s in scores)
    has_negative = any(s <= -2 for s in scores)

    conflict = has_positive and has_negative

    if not has_positive:
        return min_score  # 全负

    if conflict:
        return max_score + 0.7 * min_score

    return max_score