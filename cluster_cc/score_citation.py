import os
import os.path
import sys
from collections import defaultdict

import citation_utils

def score_citation_in_one_text(text):
    ret = citation_utils.parse_cc_output_citations_and_sentences_2(text)

    citation_d = defaultdict(float)

    for citation, index in ret['citations']:
        citation_d[citation] += (1./(1+index))

    return citation_d

def score_citation(cc_list, citation_idf_d):
    citation_d = defaultdict(float)
    for cc in cc_list:
        ret = score_citation_in_one_text(cc['text'])
        for citation, score in ret.items():
            citation_d[citation] += score

    l = [(citation, score * citation_idf_d.get(citation, 1.0)) for citation, score in citation_d.items()]
    l.sort(key=lambda x: x[1], reverse=True)

    return l
    