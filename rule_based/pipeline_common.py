
from __future__ import annotations
import math
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import random

src_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

import citation_utils


pos_signal = [
    # 直接引用强度
    'vgl.',
    'vgl. auch',
    'siehe',
    's.',
    'gemäss',
    'gemäß',
    'nach',
    'i.S.v.',
    'im Sinne von',
    'i.V.m.',
    'in Verbindung mit',
    'gestützt auf',
    'basierend auf',
    # 法律判定核心动词
    'gilt',
    'gelten',
    'ist anzuwenden',
    'findet Anwendung',
    'ist massgebend',
    'ist massgeblich',
    'bestimmt',
    'regelt',
    'sieht vor',
    'vorgesehen',
    'verstösst gegen',
    'verletzt',
    # 解释性强调
    'insbesondere',
    'namentlich',
    'ausdrücklich',
    'explizit',
    'zwingend',
    'unmittelbar',
    'analog',
    'entsprechend',
    # 论证结构词
    'daher',
    'deshalb',
    'folglich',
    'somit',
    'demnach',
    'mithin',
    'ergibt sich',
    'daraus folgt',
]

neg_signal = [
    # 反事实/否定语境
    'nicht',
    'kein',
    'entgegen',
    'abweichend von',
    'ungeachtet',
    'trotz',
    'soweit nicht',
    # 边缘引用标志
    'a.M.',
    'anderer Meinung',
    'a.A.',
    'anderer Ansicht',
    'str.',
    'strittig',
    'umstritten',
    'fraglich',
    'zweifelhaft',
    'offen gelassen',
    'dahingestellt',

    # 纯列举/背景性引用
    'u.a.',
    'unter anderem',
    'etwa',
    'z.B.',
    'zum Beispiel',
    'allgemein',
    'allgemeine Meinung',
    'herrschende Lehre',
    'h.L.',
    'herrschende Meinung',
    'h.M.'
]

def calc_contain_pos_score(sentence, signals):
    ret = 0
    for signal in signals:
        if signal in sentence:
            ret += 1
    return ret

def calc_contain_neg_score(sentence, signals):
    ret = 0
    for signal in signals:
        if signal in sentence:
            ret -= 1
    return ret


def calculate_score_for_cc(parsed_cc):
    citation_score_d = defaultdict(int)
    sentences = parsed_cc['sentences']
    for citation, index in parsed_cc['citations']:
        pos_score = calc_contain_pos_score(sentences[index], pos_signal)
        neg_score = calc_contain_neg_score(sentences[index], neg_signal)
        total_score = pos_score + neg_score
        citation_score_d[citation] += total_score
    return citation_score_d

def extract_feature_cid_2_cid_feature_list(query_id, train_candidate_d, gold_citations):
    reranked_l = train_candidate_d[query_id]['rerank']
    for hit, score in reranked_l:
        parsed_cc = citation_utils.parse_cc_output_citations_and_sentences_2(hit['text'])

        cid_2_score_d = calculate_score_for_cc(parsed_cc)