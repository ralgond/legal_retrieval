import os
import os.path
import sys
import numpy as np
import math
from collections import defaultdict

src_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

import citation_utils

def _maxmin_normalize_hits(hits):
    max_value = hits[0][1]
    min_value = hits[0][1]
    for i in range(1, len(hits)):
        max_value = max(max_value, hits[i][1])
        min_value = min(min_value, hits[i][1])
    span = max_value - min_value

    ret = [[hit.copy(), score] for hit,score in hits]
    for hit in ret:
        hit[1] = (hit[1] - min_value) * 1. / span

    return [(hit,score) for hit,score in ret]

class CC:
    '''
    这是被Query持有的
    '''
    def __init__(self, cc_id, cc_text, score):
        self.cc_id = cc_id
        self.cc_text = cc_text
        self.cc_score = score
        self.citation_d = defaultdict(list)
        self.contains_gold = False
        self.parsed_cc = None

    def add_citation(self, citation_id, index, is_gold):
        self.citation_d[citation_id].append(index)
        if is_gold:
            self.contains_gold = True

class Hit:
    '''
    这是被citation持有的
    '''
    def __init__(self, cc_id, cc_text, cc_score, hit_rank, first_appear_sentence_index):
        self.cc_id = cc_id
        self.cc_text = cc_text
        self.cc_score = cc_score
        self.hit_rank = hit_rank
        self.first_appear_sentence_index = first_appear_sentence_index

class Citation:
    def __getattr__(self, name):
        # 如果你想访问实例字典中的真实属性，请使用以下方式：
        # 注意：通常 __getattr__ 只在属性不存在时触发，
        # 这里是为了演示如何安全访问属性而不触发递归
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            # 如果确实找不到，返回默认值或抛异常
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    FEATURE_NAMES = [
        # 'coverage',
        # 'in_cc_which_contain_gold_count',
        'max_pooling_first_sentence_length',
        'bool_pooling_endswith_30p',
        'has_key_words_1_in_sentence',
        'has_key_words_2_in_sentence',
        'has_key_words_3_in_sentence',
        'has_key_words_4_in_sentence',
        'avg_sentence_char_count'
    ]

    N_FEATS = len(FEATURE_NAMES)

    def __init__(self, cid, query):
        self.cid = cid
        self.query = query
        self.refer_hit_l = []
        self.appear_in_rerank_count = 0

    def add_refer_hit(self, cc_id, cc_text, cc_score, hit_rank, first_appear_sentence_index):
        self.refer_hit_l.append(Hit(cc_id, cc_text, cc_score, hit_rank, first_appear_sentence_index))
        self.appear_in_rerank_count += 1

    def extract_feature(self):
        coverage = self.appear_in_rerank_count / self.query._get_cc_count()
        
        in_cc_which_contain_gold_count = 0
        for hit in self.refer_hit_l:
            if self.query._is_cc_contains_gold(hit.cc_id):
                in_cc_which_contain_gold_count += 1

        '''
        寻找所有所在的cc中citation在的第一个句子的长度，然后进行max_pooling
        '''
        max_length = 0
        for hit in self.refer_hit_l:
            length = self.query._found_the_first_sentence_length_for_citation(hit.cc_id, self.cid)
            if length > max_length:
                max_length = length

        '''
        '''
        bool_l = []
        for hit in self.refer_hit_l:
            bool_l.append(self.query._found_if_endswith_30p_citation(hit.cc_id, self.cid))

        has_key_words_1_in_sentence = 0
        for hit in self.refer_hit_l:
            has_key_words_1_in_sentence += self.query._has_key_words_1_in_sentence(hit.cc_id, self.cid)

        has_key_words_2_in_sentence = 0
        for hit in self.refer_hit_l:
            has_key_words_2_in_sentence += self.query._has_key_words_2_in_sentence(hit.cc_id, self.cid)

        has_key_words_3_in_sentence = 0
        for hit in self.refer_hit_l:
            has_key_words_3_in_sentence += self.query._has_key_words_3_in_sentence(hit.cc_id, self.cid)

        has_key_words_4_in_sentence = 0
        for hit in self.refer_hit_l:
            has_key_words_4_in_sentence += self.query._has_key_words_4_in_sentence(hit.cc_id, self.cid)


        all_sentence = []
        for hit in self.refer_hit_l:
            all_sentence.extend(self.query._get_sentences(hit.cc_id, self.cid))
        N = len(all_sentence)
        char_cout = 0
        for s in all_sentence:
            char_cout += len(s)
        avg_sentence_char_count = char_cout / N


        return {
            # 'coverage': coverage,
            # 'in_cc_which_contain_gold_count': in_cc_which_contain_gold_count,
            'max_pooling_first_sentence_length': max_length,
            'bool_pooling_endswith_30p': any(bool_l),
            'has_key_words_1_in_sentence': has_key_words_1_in_sentence,
            'has_key_words_2_in_sentence': has_key_words_2_in_sentence,
            'has_key_words_3_in_sentence': has_key_words_3_in_sentence,
            'has_key_words_4_in_sentence': has_key_words_4_in_sentence,
            'avg_sentence_char_count': avg_sentence_char_count
        }
        
class Query:
    def __init__(self, q_id, gold_citation_set):
        self.q_id = q_id
        self.cc_d = {}
        self.cc_l = []
        self.gold_citation_set = gold_citation_set

    def parse(self, cc_list):
        sorted_cc_list = sorted(cc_list, key=lambda x: x[1], reverse=True)
        for cc, score in sorted_cc_list:
            cc_id = cc['citation']
            cc_text = cc['text']
            new_cc = CC(cc_id, cc_text, score)

            new_cc.parsed_cc = citation_utils.parse_cc_output_citations_and_sentences_2(cc_text)
            
            for citation_id, sentence_index in new_cc.parsed_cc['citations']:
                if citation_id in self.gold_citation_set:
                    new_cc.add_citation(citation_id, sentence_index, True)
                else:
                    new_cc.add_citation(citation_id, sentence_index, False)

            self.cc_l.append(new_cc)
            self.cc_d[new_cc.cc_id] = new_cc

    def _get_cc_count(self):
        return len(self.cc_l)

    def _is_cc_contains_gold(self, cc_id):
        cc = self.cc_d[cc_id]
        return cc.contains_gold

    def _found_the_first_sentence_length_for_citation(self, cc_id, cid):
        cc = self.cc_d[cc_id]
        index_l = [index for _cid, index in cc.parsed_cc['citations'] if _cid == cid]
        return len(cc.parsed_cc['sentences'][index_l[0]])

    def _found_if_endswith_30p_citation(self, cc_id, cid):
        cc = self.cc_d[cc_id]
        index_l = [index for _cid, index in cc.parsed_cc['citations'] if _cid == cid]
        for index in index_l:
            _i = cc.parsed_cc['sentences'][index].find(cid)
            if _i / len(cc.parsed_cc['sentences'][index]) > 0.7:
                return True
        return False

    def _has_key_words_1_in_sentence(self, cc_id, cid):
        ret = 0
        key_words = ['Anspruch', 'Recht', 'Pflicht' ,'Vertrag', 'Haftung', 'Schaden']
        cc = self.cc_d[cc_id]
        index_l = [index for _cid, index in cc.parsed_cc['citations'] if _cid == cid]
        for index in index_l:
            s = cc.parsed_cc['sentences'][index]
            for kw in key_words:
                if kw.lower() in s.lower():
                    ret += 1
        return ret

    def _has_key_words_2_in_sentence(self, cc_id, cid):
        ret = 0
        key_words = ['mithin', 'folglich', 'demnach', 'ergibt sich' 'im Sinne von', 'nach Massgabe', 'gemäss', 'ist zu prüfen', 'zu beurteilen',
                     'verletzt', 'verstösst gegen', 'widerspricht', 'anwendbar', 'anwendbarkeit']
        cc = self.cc_d[cc_id]
        index_l = [index for _cid, index in cc.parsed_cc['citations'] if _cid == cid]
        for index in index_l:
            s = cc.parsed_cc['sentences'][index]
            for kw in key_words:
                if kw.lower() in s.lower():
                    ret += 1
        return ret

    def _has_key_words_3_in_sentence(self, cc_id, cid):
        ret = 0
        key_words = ['Erwägung', 'E.', 'Erw.', 'in casu', 'im vorliegenden Fall', 'soweit ersichtlich', 'nach ständiger Rechtsprechung']
        cc = self.cc_d[cc_id]
        index_l = [index for _cid, index in cc.parsed_cc['citations'] if _cid == cid]
        for index in index_l:
            s = cc.parsed_cc['sentences'][index]
            for kw in key_words:
                if kw.lower() in s.lower():
                    ret += 1
        return ret

    def _has_key_words_4_in_sentence(self, cc_id, cid):
        ret = 0
        key_words = ['Bundesgericht', 'BGer', 'BGE', 'ständige Rechtsprechung', 'gefestigte Praxis', 'Leitentscheid', 'publiziert in']
        cc = self.cc_d[cc_id]
        index_l = [index for _cid, index in cc.parsed_cc['citations'] if _cid == cid]
        for index in index_l:
            s = cc.parsed_cc['sentences'][index]
            for kw in key_words:
                if kw.lower() in s.lower():
                    ret += 1
        return ret

    def _get_sentences(self, cc_id, cid):
        cc = self.cc_d[cc_id]
        index_l = [index for _cid, index in cc.parsed_cc['citations'] if _cid == cid]
        return [cc.parsed_cc['sentences'][index] for index in index_l]


    def extract_feature(self):
        all_citation_d = {}
        for rank, cc in enumerate(self.cc_l, 1):
            for cid, pos_l in cc.citation_d.items():
                if cid not in all_citation_d:
                    all_citation_d[cid] = Citation(cid, self)
                all_citation_d[cid].add_refer_hit(cc.cc_id, cc.cc_text, cc.cc_score, rank, pos_l[0])

        ret = {}
        for cid, citation in all_citation_d.items():
            ret[cid] = citation.extract_feature()
        return ret


def extract_features_for_query(
        query_id: str, query: str, candidate_d, gold_citation_set
) -> dict[str, np.ndarray]:

    hits3 = candidate_d[query_id]['rerank']

    norm_hits3 = _maxmin_normalize_hits(hits3)

    q = Query(query_id, gold_citation_set)

    q.parse(norm_hits3)
        
    accum = q.extract_feature()
    
    # 整理为特征向量
    cid_feat_d: dict[str, np.ndarray] = {}

    for cid, a in accum.items():
        # freq = a["cite_freq"]

        feat_vec = np.array([
            # a['coverage'],
            # a['in_cc_which_contain_gold_count'],
            a['max_pooling_first_sentence_length'],
            a['bool_pooling_endswith_30p'],
            a['has_key_words_1_in_sentence'],
            a['has_key_words_2_in_sentence'],
            a['has_key_words_3_in_sentence'],
            a['has_key_words_4_in_sentence'],
            a['avg_sentence_char_count']
        ], dtype=np.float32)
        assert len(feat_vec) == Citation.N_FEATS, \
            f"Feature dim mismatch: {len(feat_vec)} vs {Citation.N_FEATS}  cid={cid}"
        cid_feat_d[cid] = feat_vec

    return cid_feat_d

