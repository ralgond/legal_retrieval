import citation_utils

import os
import os.path
import sys
import numpy as np
import math
from collections import defaultdict

src_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

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
    def __init__(self, cc_id, cc_text, cc_score, from_hit_type, hit_rank, first_appear_sentence_index):
        self.cc_id = cc_id
        self.cc_text = cc_text
        self.cc_score = cc_score
        self.from_hit_type = from_hit_type
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
        'dense_min_rank',
        # 'dense_max_score',
        'dense_avg_score',
        'dense_top3_score_sum',
        # 'dense_score_multiply_decay_and_get_max_contribution',
        'dense_avg_decay_reciprocal',
        'dense_avg_decay_log',
        'dense_best_position_score',
        'dense_cc_score_decay_avg_reciproca',
        'dense_cc_score_decay_avg_log',

        'sparse_min_rank',
        # 'sparse_max_score',
        'sparse_avg_score',
        'sparse_top3_score_sum',
        # 'sparse_score_multiply_decay_and_get_max_contribution',
        'sparse_avg_decay_reciprocal',
        'sparse_avg_decay_log',
        'sparse_best_position_score',
        'sparse_cc_score_decay_avg_reciprocal',
        'sparse_cc_score_decay_avg_log',

        'rerank_min_rank',
        # 'rerank_max_score',
        'rerank_avg_score',
        'rerank_top3_score_sum',
        # 'rerank_score_multiply_decay_and_get_max_contribution',
        'rerank_avg_decay_reciprocal',
        'rerank_avg_decay_log',
        'rerank_best_position_score',
        'rerank_cc_score_decay_avg_reciprocal',
        'rerank_cc_score_decay_avg_log',
        # 'boolean_in_dense_and_sparse',
        # 'freq_reciprocal'
    ]

    N_FEATS = len(FEATURE_NAMES)
    
    def __init__(self, cid):
        self.cid = cid
        self.refer_cc_l = []
        self.appear_in_dense_count = 0
        self.appear_in_sparse_count = 0
        self.appear_in_rerank_count = 0

        self.dense_cc_count = 0
        self.sparse_cc_count = 0
        self.rerank_cc_count = 0

    def add_refer_cc(self, cc_id, cc_text, cc_score, from_hit_type, hit_rank, first_appear_sentence_index: int | None = None):
        if from_hit_type == 'dense':
            self.dense_cc_count += 1
        elif from_hit_type == 'sparse':
            self.sparse_cc_count += 1
        elif from_hit_type == 'rerank':
            self.rerank_cc_count += 1
            
        if first_appear_sentence_index is None:
            pass
        else: 
            self.refer_cc_l.append(CC(cc_id, cc_text, cc_score, from_hit_type, hit_rank, first_appear_sentence_index))
            if from_hit_type == 'dense':
                self.appear_in_dense_count += 1
            elif from_hit_type == 'sparse':
                self.appear_in_sparse_count += 1
            elif from_hit_type == 'rerank':
                self.appear_in_rerank_count += 1

    def __extract_feature_method_1(self, from_hit_type):
        if from_hit_type not in ['dense', 'sparse', 'rerank']:
            raise ValueError(f"{from_hit_type} is not valid.")

        # print(len(self.refer_cc_l), from_hit_type, self.cid)
        score_l = [cc.cc_score for cc in self.refer_cc_l if cc.from_hit_type == from_hit_type]
        if len(score_l) > 0:
            min_rank = min([cc.hit_rank for cc in self.refer_cc_l if cc.from_hit_type == from_hit_type])
            max_score = max(score_l)
            avg_score = sum(score_l) / len(score_l)
            top3_score_sum = sum(sorted(score_l, reverse=True)[:3])
        else:
            min_rank = 99999
            max_score = 0.
            avg_score = 0.
            top3_score_sum = 0.

        return {
            from_hit_type + "_min_rank" : min_rank,
            from_hit_type + "_max_score" : max_score,
            from_hit_type + "_avg_score" : avg_score,
            from_hit_type + "_top3_score_sum" : top3_score_sum,
        }

    def __extract_feature_method_2_score_multiply_decay_and_get_max_contribution(self, from_hit_type):
        if from_hit_type not in ['dense', 'sparse', 'rerank']:
            raise ValueError(f"{from_hit_type} is not valid.")

        l = [(cc.cc_score, cc.first_appear_sentence_index) for cc in self.refer_cc_l if cc.from_hit_type == from_hit_type]
        max_contribution = 0.
        if len(l) > 0:
            l2 = [score * 1./math.log(2+first_appear_sentence_index) for score, first_appear_sentence_index in l]
            max_contribution = max(l2)
            
        return {from_hit_type + "_score_multiply_decay_and_get_max_contribution": max_contribution }

    def __extract_feature_method_3_avg_decay(self, from_hit_type):
        reciprocal_l = [1/(1+cc.first_appear_sentence_index) for cc in self.refer_cc_l if cc.from_hit_type == from_hit_type]
        reciprocal_score = 0.
        if len(reciprocal_l) > 0:
            reciprocal_score = sum(reciprocal_l) / len(reciprocal_l)

        log_l = [1/math.log(2+cc.first_appear_sentence_index) for cc in self.refer_cc_l if cc.from_hit_type == from_hit_type]
        log_score = 0.
        if len(log_l) > 0:
            log_score = sum(log_l) / len(log_l)

        return {from_hit_type + "_avg_decay_reciprocal": reciprocal_score, from_hit_type + "_avg_decay_log": log_score,}

    def __extract_feature_method_4_boolean_in_dense_and_sparse(self):
        in_dense_and_sparse = 0.
        if self.appear_in_dense_count > 0 and self.appear_in_sparse_count > 0:
            in_dense_and_sparse = 1.
        return { "boolean_in_dense_and_sparse": in_dense_and_sparse }

    def __extract_feature_method_5_best_position_score(self, from_hit_type):
        l = [(cc.cc_score, cc.first_appear_sentence_index) for cc in self.refer_cc_l if cc.from_hit_type == from_hit_type]
        score = 0.
        if len(l) > 0:
            l2 = sorted(l, key=lambda x:x[1], reverse=True)
            best_pos = min([pos for _, pos in l2])
            d = defaultdict(list)
            for score, pos in l2:
                d[pos].append(score)
            score = sum(d[best_pos]) / len(d[best_pos])
        return {f"{from_hit_type}_best_position_score" : score}

    def __extract_feature_method_6_cc_score_decay_avg(self, from_hit_type):
        l = [(cc.cc_score, cc.first_appear_sentence_index) for cc in self.refer_cc_l if cc.from_hit_type == from_hit_type]

        reciprocal_score = 0.
        if len(l) > 0:
            l2 = [score*1/(1+idx) for score, idx in l]
            reciprocal_score = sum(l2) / len(l2)

        log_score = 0.
        if len(l) > 0:
            l2 = [score*1/math.log(2+idx) for score, idx in l]
            log_score = sum(l2) / len(l2)

        return {f"{from_hit_type}_cc_score_decay_avg_reciprocal": reciprocal_score, f"{from_hit_type}_cc_score_decay_avg_log":log_score}
        
    def extract_feature(self): # return Dict[feature_name->float]

        dense_d = self.__extract_feature_method_1('dense')
        sparse_d = self.__extract_feature_method_1('sparse')
        rerank_d = self.__extract_feature_method_1('rerank')

        dense_d2 = self.__extract_feature_method_2_score_multiply_decay_and_get_max_contribution('dense')
        sparse_d2 = self.__extract_feature_method_2_score_multiply_decay_and_get_max_contribution('sparse')
        rerank_d2 = self.__extract_feature_method_2_score_multiply_decay_and_get_max_contribution('rerank')

        dense_d3 = self.__extract_feature_method_3_avg_decay('dense')
        sparse_d3 = self.__extract_feature_method_3_avg_decay('sparse')
        rerank_d3 = self.__extract_feature_method_3_avg_decay('rerank')

        # d4 = self.__extract_feature_method_4_boolean_in_dense_and_sparse()

        dense_d5 = self.__extract_feature_method_5_best_position_score('dense')
        sparse_d5 = self.__extract_feature_method_5_best_position_score('sparse')
        rerank_d5 = self.__extract_feature_method_5_best_position_score('rerank')

        dense_d6 = self.__extract_feature_method_6_cc_score_decay_avg('dense')
        sparse_d6 = self.__extract_feature_method_6_cc_score_decay_avg('sparse')
        rerank_d6 =self.__extract_feature_method_6_cc_score_decay_avg('rerank')
        
        self.dense_l = []
        self.sparse_l = []
        self.rerank_l = []

        for cc in self.refer_cc_l:
            if cc.from_hit_type == 'dense':
                self.dense_l.append(cc)
            elif cc.from_hit_type == 'sparse':
                self.sparse_l.append(cc)
            elif cc.from_hit_type == 'rerank':
                self.rerank_l.append(cc)

        merged_method_1_dict = {
            **dense_d, **sparse_d, **rerank_d, 
            **dense_d2, **sparse_d2, **rerank_d2,
            **dense_d3, **sparse_d3, **rerank_d3,
            # **d4
            **dense_d5, **sparse_d5, **rerank_d5,
            **dense_d6, **sparse_d6, **rerank_d6,
        }

        return merged_method_1_dict

class Query:
    def __init__(self, q_id):
        self.q_id = q_id
        self.cc_id_2_text_d = {}
        self.cc_id_2_norm_dense_score = dict()
        self.cc_id_2_norm_sparse_score = dict()
        self.cc_id_2_norm_rerank_score = dict()
        self.norm_dense_hits = None
        self.norm_sparse_hits = None
        self.norm_rerank_hits = None

    def assign_text_to_cc(self, court_consideration_d):
        cc_id_l = self.get_cc_id_l()
        for cc_id in cc_id_l:
            self.cc_id_2_text_d[cc_id] = court_consideration_d[cc_id]

    def get_text_for_cc(self, cc_id):
        return self.cc_id_2_text_d[cc_id]

    def add_norm_dense_hits(self, norm_dense_hits):
        for hit, score in norm_dense_hits:
            cc_id = hit['citation']
            self.cc_id_2_norm_dense_score[cc_id] = score
        self.norm_dense_hits = sorted(norm_dense_hits.copy(), key=lambda x:x[1], reverse=True)

    def add_norm_sparse_hits(self, norm_sparse_hits):
        for hit, score in norm_sparse_hits:
            cc_id = hit['citation']
            self.cc_id_2_norm_sparse_score[cc_id] = score
        self.norm_sparse_hits = sorted(norm_sparse_hits.copy(), key=lambda x:x[1], reverse=True)

    def add_norm_rerank_hits(self, norm_rerank_hits):
        for hit, score in norm_rerank_hits:
            cc_id = hit['citation']
            self.cc_id_2_norm_rerank_score[cc_id] = score
        self.norm_rerank_hits = sorted(norm_rerank_hits.copy(), key=lambda x:x[1], reverse=True)

    def get_cc_dense_norm(self, cc_id):
        return self.cc_id_2_norm_dense_score.get(cc_id, 0.)

    def get_cc_sparse_norm(self, cc_id):
        return self.cc_id_2_norm_sparse_score.get(cc_id, 0.)

    def get_cc_rerank_norm(self, cc_id):
        return self.cc_id_2_norm_rerank_score.get(cc_id, 0.)

    def get_cc_id_l (self):
        return list(set(self.cc_id_2_norm_dense_score.keys()) | set(self.cc_id_2_norm_sparse_score.keys()) | set(self.cc_id_2_norm_rerank_score.keys()))

    def extract_feature(self): # Dict[citation_id, Citation]
        cc_id_2_parsed_cc_d = {}

        # step 1: parse all cc
        first_appear_sentence_index_d = {}
        for cc_id in self.get_cc_id_l():
            cc_text = self.get_text_for_cc(cc_id)
            parsed_cc = citation_utils.parse_cc_output_citations_and_sentences(cc_text)
            cc_id_2_parsed_cc_d[cc_id] = parsed_cc
            for citation_id, first_appear_sentence_index in parsed_cc['citations']:
                first_appear_sentence_index_d[(citation_id,cc_id)] = first_appear_sentence_index


        citation_freq = defaultdict(int)
        for (citation_id,cc_id),_ in first_appear_sentence_index_d.items():
            citation_freq[citation_id] += 1

        # step 2: found all refered citation
        citation_id_2_citation_d = dict()
        for cc_id, parsed_cc in cc_id_2_parsed_cc_d.items():
            for citation_id, _ in parsed_cc['citations']:
                if citation_id not in citation_id_2_citation_d:
                    citation_id_2_citation_d[citation_id] = Citation(citation_id)

        print("self.norm_dense_hits.len:", len(self.norm_dense_hits))
        print("self.norm_sparse_hits.len:", len(self.norm_sparse_hits))
        print("self.norm_rerank_hits.len:", len(self.norm_rerank_hits))
        
        # step 3: assign information to citation
        for rank, (hit, score) in enumerate(self.norm_dense_hits, start=1):
            cc_id = hit['citation']
            cc_text = self.get_text_for_cc(cc_id)
            for citation_id, citation in citation_id_2_citation_d.items():
                citation.add_refer_cc(cc_id, cc_text, score, "dense", rank, first_appear_sentence_index_d.get((citation_id,cc_id), None))
            
        for rank, (hit, score) in enumerate(self.norm_sparse_hits, start=1):
            cc_id = hit['citation']
            cc_text = self.get_text_for_cc(cc_id)
            for citation_id, citation in citation_id_2_citation_d.items():
                citation.add_refer_cc(cc_id, cc_text, score, "sparse", rank, first_appear_sentence_index_d.get((citation_id,cc_id), None))

        for rank, (hit, score) in enumerate(self.norm_rerank_hits, start=1):
            cc_id = hit['citation']
            cc_text = self.get_text_for_cc(cc_id)
            for citation_id, citation in citation_id_2_citation_d.items():
                citation.add_refer_cc(cc_id, cc_text, score, "rerank", rank, first_appear_sentence_index_d.get((citation_id,cc_id), None))

        accum = {}
        for citation_id, citation in citation_id_2_citation_d.items():
            accum[citation_id] = citation.extract_feature()

        return accum, citation_freq

def extract_features_for_query(
        query_id: str, query: str, court_consideration_d, train_candidate_d, valid_candidate_d, test_candidate_d
) -> dict[str, np.ndarray]:
    """
    对单个 query 做检索+rerank，返回
      { citation_id: np.ndarray(Citation.N_FEATS,) }
    """

    if query_id in train_candidate_d:
        hits1 = train_candidate_d[query_id]['dense']
        hits2 = train_candidate_d[query_id]['sparse']
        hits3 = train_candidate_d[query_id]['rerank']
    elif query_id in valid_candidate_d:
        hits1 = valid_candidate_d[query_id]['dense']
        hits2 = valid_candidate_d[query_id]['sparse']
        hits3 = valid_candidate_d[query_id]['rerank']
    elif query_id in test_candidate_d:
        hits1 = test_candidate_d[query_id]['dense']
        hits2 = test_candidate_d[query_id]['sparse']
        hits3 = test_candidate_d[query_id]['rerank']
    else:
        return {}

    norm_hits1 = _maxmin_normalize_hits(hits1)
    norm_hits2 = _maxmin_normalize_hits(hits2)
    norm_hits3 = _maxmin_normalize_hits(hits3)
    
    q = Query(q_id=query_id)
    q.add_norm_dense_hits(norm_hits1)
    q.add_norm_sparse_hits(norm_hits2)
    q.add_norm_rerank_hits(norm_hits3)
    q.assign_text_to_cc(court_consideration_d)
    accum, citation_freq = q.extract_feature()
    
    # 整理为特征向量
    cid_feat_d: dict[str, np.ndarray] = {}

    for cid, a in accum.items():
        # freq = a["cite_freq"]

        feat_vec = np.array([
            a['dense_min_rank'],
            # a['dense_max_score'] * 1./math.log(2+citation_freq.get(cid,0.)),
            a['dense_avg_score'],
            a['dense_top3_score_sum']  * 1./(1+citation_freq.get(cid,0.)),
            # a['dense_score_multiply_decay_and_get_max_contribution'] 
            a['dense_avg_decay_reciprocal'],
            a['dense_avg_decay_log'],
            a['dense_best_position_score'],
            a['dense_cc_score_decay_avg_reciprocal'],
            a['dense_cc_score_decay_avg_log'],

            a['sparse_min_rank'],
            # a['sparse_max_score'] * 1./math.log(2+citation_freq.get(cid,0.)),
            a['sparse_avg_score'],
            a['sparse_top3_score_sum']  * 1./(1+citation_freq.get(cid,0.)),
            # a['sparse_score_multiply_decay_and_get_max_contribution'] 
            a['sparse_avg_decay_reciprocal'],
            a['sparse_avg_decay_log'],
            a['sparse_best_position_score'],
            a['sparse_cc_score_decay_avg_reciprocal'],
            a['sparse_cc_score_decay_avg_log'],

            a['rerank_min_rank'],
            # a['rerank_max_score'] * 1./math.log(2+citation_freq.get(cid,0.)),
            a['rerank_avg_score'],
            a['rerank_top3_score_sum']  * 1./(1+citation_freq.get(cid,0.)),
            # a['rerank_score_multiply_decay_and_get_max_contribution'] 
            a['rerank_avg_decay_reciprocal'],
            a['rerank_avg_decay_log'],
            a['rerank_best_position_score'],
            a['rerank_cc_score_decay_avg_reciprocal'],
            a['rerank_cc_score_decay_avg_log'],
            # a['boolean_in_dense_and_sparse']
            # 1/ (1 + citation_freq.get(cid, 0.)) # 'freq_reciprocal'
        ], dtype=np.float32)
        assert len(feat_vec) == Citation.N_FEATS, \
            f"Feature dim mismatch: {len(feat_vec)} vs {Citation.N_FEATS}  cid={cid}"
        cid_feat_d[cid] = feat_vec

    return cid_feat_d