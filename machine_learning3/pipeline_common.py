import citation_utils

import os
import os.path
import sys
import numpy as np
import math

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
        'dense_max_score',
        'dense_avg_score',
        'dense_top3_score_sum',
        'dense_score_multiply_decay_and_get_max_contribution',
        'dense_appear_percentage',
        'sparse_max_score',
        'sparse_avg_score',
        'sparse_top3_score_sum',
        'sparse_score_multiply_decay_and_get_max_contribution',
        'sparse_appear_percentage',
        'rerank_max_score',
        'rerank_avg_score',
        'rerank_top3_score_sum',
        'rerank_score_multiply_decay_and_get_max_contribution',
        'rerank_appear_percentage',
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
            max_score = max(score_l)
            avg_score = sum(score_l) / len(score_l)
            top3_score_sum = sum(sorted(score_l, reverse=True)[:3])
        else:
            max_score = 0.
            avg_score = 0.
            top3_score_sum = 0.

        return {
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

    def __extract_feature_method_3_appear_percentage(self, from_hit_type):
        total_count = self.__getattr__(f"{from_hit_type}_cc_count")
        appear_percentage = 0.
        if total_count > 0:
            appear_percentage = len([1 for cc in self.refer_cc_l if cc.from_hit_type == from_hit_type]) / total_count

        return {from_hit_type + "_appear_percentage": appear_percentage }
        
        
    def extract_feature(self): # return Dict[feature_name->float]

        dense_d = self.__extract_feature_method_1('dense')
        sparse_d = self.__extract_feature_method_1('sparse')
        rerank_d = self.__extract_feature_method_1('rerank')

        dense_d2 = self.__extract_feature_method_2_score_multiply_decay_and_get_max_contribution('dense')
        sparse_d2 = self.__extract_feature_method_2_score_multiply_decay_and_get_max_contribution('sparse')
        rerank_d2 = self.__extract_feature_method_2_score_multiply_decay_and_get_max_contribution('rerank')

        dense_d3 = self.__extract_feature_method_3_appear_percentage('dense')
        sparse_d3 = self.__extract_feature_method_3_appear_percentage('sparse')
        rerank_d3 = self.__extract_feature_method_3_appear_percentage('rerank')
        
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
        self.norm_dense_hits = norm_dense_hits.copy()

    def add_norm_sparse_hits(self, norm_sparse_hits):
        for hit, score in norm_sparse_hits:
            cc_id = hit['citation']
            self.cc_id_2_norm_sparse_score[cc_id] = score
        self.norm_sparse_hits = norm_sparse_hits.copy()

    def add_norm_rerank_hits(self, norm_rerank_hits):
        for hit, score in norm_rerank_hits:
            cc_id = hit['citation']
            self.cc_id_2_norm_rerank_score[cc_id] = score
        self.norm_rerank_hits = norm_rerank_hits.copy()

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
            
        return accum

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
    accum = q.extract_feature()
    
    # 整理为特征向量
    cid_feat_d: dict[str, np.ndarray] = {}

    for cid, a in accum.items():
        # freq = a["cite_freq"]

        feat_vec = np.array([
            a['dense_max_score'],
            a['dense_avg_score'],
            a['dense_top3_score_sum'],
            a['dense_score_multiply_decay_and_get_max_contribution'],
            a['dense_appear_percentage'],
            a['sparse_max_score'],
            a['sparse_avg_score'],
            a['sparse_top3_score_sum'],
            a['sparse_score_multiply_decay_and_get_max_contribution'],
            a['sparse_appear_percentage'],
            a['rerank_max_score'],
            a['rerank_avg_score'],
            a['rerank_top3_score_sum'],
            a['rerank_score_multiply_decay_and_get_max_contribution'],
            a['rerank_appear_percentage'],
        ], dtype=np.float32)
        assert len(feat_vec) == Citation.N_FEATS, \
            f"Feature dim mismatch: {len(feat_vec)} vs {Citation.N_FEATS}  cid={cid}"
        cid_feat_d[cid] = feat_vec

    return cid_feat_d