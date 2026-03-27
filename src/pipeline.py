import pandas as pd
import re
import math
from collections import defaultdict

from tqdm.notebook import tqdm

import hits_utils
import citation_utils
import reranker_utils
import metric_utils
import text_chunk

# 瑞士德语法律文档典型段落标志
SECTION_PATTERNS = {
    "sachverhalt":   (r"(Sachverhalt|Tatbestand|A\.\s)", 1.4),   # 事实陈述
    "erwaegungen":   (r"(Erwägung|Erwägungen|E\.\s)",    1.2),   # 法律考量（核心）
    "dispositiv":    (r"(Dispositiv|Demnach|erkennt)",   1.8),   # 判决主文 ← 最高权重
    "rechtsmittel":  (r"(Rechtsmittel|Beschwerde)",      0.7),   # 上诉告知
    "unterschrift":  (r"(\d{1,2}\.\s\w+\s\d{4}|Im Namen)", 0.3), # 日期/签名
}

class Pipeline:
    def __init__(self,
                 court_consideration_df: pd.DataFrame,
                 court_consideration_d: dict,
                 law_df: pd.DataFrame,
                 law_d: dict,
                 dense_index,
                 sparse_index,
                 reranker,
                 test_df: pd.DataFrame,
                 valid_df: pd.DataFrame,
                 **kwargs
                ):
        self.court_consideration_df = court_consideration_df
        self.court_consideration_d = court_consideration_d,
        self.law_df = law_df
        self.law_d = law_d
        self.dense_index = dense_index
        self.sparse_index = sparse_index
        self.reranker = reranker
        self.test_df = test_df
        self.valid_df = valid_df

        self.dense_recall_count = kwargs.get('dense_recall_count', 1000)
        self.sparse_recall_count = kwargs.get('sparse_recall_count', 1000)
        self.citation_agg_w1 = kwargs.get('citation_agg_w1', 0.5)
        self.citation_agg_w2 = kwargs.get('citation_agg_w2', 0.5)
        self.citation_agg_w3 = kwargs.get('citation_agg_w3', 0.5)
        self.global_citaion_ranking_pool_method = kwargs.get('global_citaion_ranking_pool_method', 'sum')
        self.false_positive_threshold_score = kwargs.get('false_positive_threshold_score', 1.0)
        self.window_size = kwargs.get('window_size', 4)
        self.step = kwargs.get('step', 1)
        
    def recall(self, query):
        hit_with_score_l1 = self.dense_index.search_with_score(query, self.dense_recall_count)
        hit_with_score_l2 = self.sparse_index.search_with_score(query, self.sparse_recall_count)
        hit_with_score_l = hits_utils.merge_hits_with_score_l_by_weighted_add(hit_with_score_l1, hit_with_score_l2, 0.5, 0.5)
        print("[recall] hit_with_score_l.len:", len(hit_with_score_l))
        return hit_with_score_l

    def normalize_sr(self, hit_with_recall_score_l):
        ret = []
        count = 0
        for hit, score in hit_with_recall_score_l:
            old_text = hit['text']
            hit['text'] = citation_utils.normalized_sr(hit['text'])
            if old_text != hit['text']:
                count += 1
        print("[normalize_sr] count:", count)


    def __structural_weight(self, window_text: str, fallback: float = 1.0) -> float:
        for section, (pattern, weight) in SECTION_PATTERNS.items():
            if re.search(pattern, window_text, re.IGNORECASE):
                return weight
        return fallback
    
    def __score_citation_for_doc(self, citation_parent_child_score_l):
        # 1. 该 citation 被覆盖的 window 数量（coverage）
        citation_coverage_window_count_d = defaultdict(int)
        for citation, parent, child, score in citation_parent_child_score_l:
            citation_coverage_window_count_d[citation] += 1
            
        # 2. 最高 window 分数（peak relevance）
        citation_peak_rel_window_d = defaultdict(float)
        for citation, parent, child, score in citation_parent_child_score_l:
            if citation not in citation_coverage_window_count_d:
                citation_coverage_window_count_d[citation] = score
            elif citation_coverage_window_count_d[citation] < score:
                citation_coverage_window_count_d[citation] = score
        
        # 3. window 在文档中的位置权重（开头和结尾的段落通常更重要）
        citation_window_pos_d = defaultdict(float)
        for citation, parent, child, score in citation_parent_child_score_l:
            pos_score = self.__structural_weight(child)
            if citation not in citation_coverage_window_count_d:
                citation_coverage_window_count_d[citation] = pos_score
            elif citation_coverage_window_count_d[citation] < pos_score:
                citation_coverage_window_count_d[citation] = pos_score

        s1 = set(citation_coverage_window_count_d.keys()) | set(citation_peak_rel_window_d.keys()) | set(citation_window_pos_d.keys())
        ret = []
        for citation in s1:
            score = 0.
            score += self.citation_agg_w1  * math.log(citation_coverage_window_count_d[citation])
            score += self.citation_agg_w2  * citation_peak_rel_window_d[citation]
            score += self.citation_agg_w3  * citation_window_pos_d[citation]

            if citation_coverage_window_count_d[citation] == 1 and score < self.false_positive_threshold_score:
                # False positive 抑制
                continue
                
            ret.append((citation, score))

        return sorted(ret, key=lambda x: x[1], reverse=True)

    def citation_aggregation(self, sorted_parent_child_with_score_l):
        citation_2_citation_parent_child_score_l_d = defaultdict(list)
        for parent_child, reranker_score in sorted_parent_child_with_score_l:
            parent, child = parent_child[0], parent_child[1]
            citations = citation_utils.extract_citations_from_text(child)
            for c in citations:
                citation_2_citation_parent_child_score_l_d[c].append([c, parent, child, reranker_score])

        doc_d = defaultdict(list)
        for citation, citation_parent_child_score_l in citation_2_citation_parent_child_score_l_d.items():
            for _, parent, child, reranker_score in citation_parent_child_score_l:
                doc_d[parent].append([citation, parent, child, reranker_score])

        ret = []
        for parent, term in doc_d.items():
            citation_score_l = self.__score_citation_for_doc(term)
            ret.append(citation_score_l)

        return ret

    def global_citation_ranking(self, citation_score_l_l):
        d = defaultdict(float)
        for citation_score_l in citation_score_l_l:
            if self.global_citaion_ranking_pool_method == 'sum':
                # sum pooling
                for citation, score in citation_score_l:
                    d[citation] += score
            elif self.global_citaion_ranking_pool_method == 'max':
                # max pooling
                for citation, score in citation_score_l:
                    if citation not in d:
                        d[citation] = score
                    elif d[citation] < score:
                        d[citation] = score
            else:
                raise ValueError("unknown method:", self.global_citaion_ranking_pool_method)

        return sorted([(citation,score) for citation,score in d.items()], key=lambda x: x[1], reverse=True)

    
    def rerank(self, query, hit_with_recall_score_l):
        sentences = []
        count = 0
        citation_2_parent_child_score_l_d = defaultdict(list)

        sentence_with_parent_child_l = []
        for hit, score in hit_with_recall_score_l:
            parent = hit['text']
            s_l = citation_utils.split_sentences(parent)
            s_l_2 = text_chunk.sliding_window_merge_last_unique(s_l, self.window_size, self.step)
            if len(s_l_2) > 1:
                count += 1

            for child in s_l_2:
                sentence_with_parent_child_l.append(((parent, child), child))

        sorted_parent_child_with_score_l = reranker_utils.rerank_batch_with_anything(self.reranker, query, sentence_with_parent_child_l)

        print(f"[rerank] sliced: {count}/{len(sentence_with_parent_child_l)}")

        return sorted_parent_child_with_score_l

    def generate_submission(self, limit=40) -> pd.DataFrame:
        query_id_l = []
        predicted_citations_l = []
        for query_id, query in tqdm(zip(self.test_df['query_id'].tolist(), self.test_df['query'].tolist()), total=len(self.test_df), desc="generate_submission"):
            ret = self.recall(query)
            self.normalize_sr(ret)
            ret = self.rerank(query, ret)
            ret = self.citation_aggregation(ret)
            ret = self.global_citation_ranking(ret)

            ret2 = []
            for citation,_ in ret:
                if citation in self.court_consideration_d or citation in self.law_d:
                    ret2.append(citation)

            pred = ';'.join(ret2[:limit])

            query_id_l.append(query_id)
            predicted_citations_l.append(pred)
            
        return pd.DataFrame({'query_id':query_id_l, 'predicted_citations':predicted_citations_l})
            
        
    def evaluate(self, stop=None):
        count = 0
        ret_l = []
        gold_citation_l = []
        for query, gold_citations in tqdm(zip(self.valid_df['query2'].tolist(), 
                                              self.valid_df['gold_citations'].tolist()), total=len(self.valid_df), desc="evaluation"):
            ret = self.recall(query)
            self.normalize_sr(ret)
            ret = self.rerank(query, ret)
            ret = self.citation_aggregation(ret)
            ret = self.global_citation_ranking(ret)

            ret2 = []
            for citation,_ in ret:
                if citation in self.court_consideration_d or citation in self.law_d:
                    ret2.append(citation)
                    
            ret_l.append(ret2)
            gold_citation_l.append(gold_citations.split(';'))
            
            count += 1
            if stop is not None and count >= stop:
                break

        max_limit = None
        max_f1 = 0
        for limit in [5,10,15,20,25,30,35,40,45]:
            r = metric_utils.cal_recall(ret_l, gold_citation_l, lambda x: limit)
            p = metric_utils.cal_precision(ret_l, gold_citation_l, lambda x: limit)
            f1 = metric_utils.cal_f1(r, p)
            if max_f1 < f1:
                max_limit = limit
                max_f1 = f1
        print(f"[{max_limit}] f1:",max_f1)
        