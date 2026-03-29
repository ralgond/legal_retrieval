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
import rrf

# 瑞士德语法律文档典型段落标志
SECTION_PATTERNS = {
    "sachverhalt":   (r"(Sachverhalt|Tatbestand|A\.\s)", 0.7),   # 事实陈述
    "erwaegungen":   (r"(Erwägung|Erwägungen|E\.\s)",    0.6),   # 法律考量（核心）
    "dispositiv":    (r"(Dispositiv|Demnach|erkennt)",   0.9),   # 判决主文 ← 最高权重
    "rechtsmittel":  (r"(Rechtsmittel|Beschwerde)",      0.35),   # 上诉告知
    "unterschrift":  (r"(\d{1,2}\.\s\w+\s\d{4}|Im Namen)", 0.15), # 日期/签名
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
                 valid_multi_question_df: pd.DataFrame,
                 **kwargs
                ):
        self.court_consideration_df = court_consideration_df
        self.court_consideration_d = court_consideration_d
        self.law_df = law_df
        self.law_d = law_d
        self.dense_index = dense_index
        self.sparse_index = sparse_index
        self.reranker = reranker
        self.test_df = test_df
        self.valid_df = valid_df
        self.valid_multi_question_df = valid_multi_question_df

        self.valid_id_2_gold_citations_d = {}
        self.valid_id_2_query_d = {}
        for query_id, gold_citations, query in zip(valid_df['query_id'], valid_df['gold_citations'], valid_df['query2']):
            self.valid_id_2_gold_citations_d[query_id] = gold_citations.split(';')
            self.valid_id_2_query_d[query_id] = query

        self.gold_citations_l = []
        
        self.query_id_2_query_list = defaultdict(list)
        
        for query_id, query in zip(self.valid_multi_question_df['query_id'], self.valid_multi_question_df['query']):
            self.query_id_2_query_list[query_id].append(query)
        
        for query_id, query in valid_id_2_query_d.items():
            self.query_id_2_query_list[query_id].append(query) # 完整的问题在最后一个
    

        self.dense_recall_count = kwargs.get('dense_recall_count', 1000)
        self.sparse_recall_count = kwargs.get('sparse_recall_count', 1000)
        self.citation_agg_w1 = kwargs.get('citation_agg_w1', 0.5)
        self.citation_agg_w2 = kwargs.get('citation_agg_w2', 0.5)
        self.citation_agg_w3 = kwargs.get('citation_agg_w3', 0.5)
        self.global_citaion_ranking_pool_method = kwargs.get('global_citaion_ranking_pool_method', 'sum')
        self.global_citation_ranking_agg_weight = kwargs.get('global_citation_ranking_agg_weight', 0.7)
        self.global_citation_ranking_recall_weight = kwargs.get('global_citation_ranking_recall_weight', 0.3)
        self.global_citation_ranking_recall_decay_fn = kwargs.get('global_citation_ranking_recall_decay_fn', lambda x: x)
        self.false_positive_threshold_score = kwargs.get('false_positive_threshold_score', 1.0)
        self.window_size = kwargs.get('window_size', 4)
        self.step = kwargs.get('step', 1)
        
        
    def recall(self, query):
        recall_citations_l = []
        for query_id, query_list in query_id_2_query_list.items():
            all_hits = []
            for query in query_list:
                hits1 = dense_index.search_with_score(query, top_k=1000)
                hits2 = sparse_index.search_with_score(query, top_k=1000)
                hits_merge = hits_utils.merge_hits_with_score_l_by_max(hits1, hits2)
                all_hits = hits_utils.merge_hits_with_score_l_by_max(all_hits, hits_merge)
                
            print(f"[{query_id}] hits_merge.len:", len(all_hits))
            self.gold_citations_l.append(valid_id_2_gold_citations_d[query_id])
        
            citations = [hit['citation'] for hit, score in all_hits]
            
            second_layer = citation_utils.compute_citation_score_with_sentence_pos(hits_merge, decay="reciprocal")
        
            for citation, score in second_layer:
                if citation in court_consideration_d:
                    citations.append(citation)
                if citation in law_d:
                    citations.append(citation)
        
            recall_citations_l.append(citations)

        r = metric_utils.cal_recall(recall_citations_l, self.gold_citations_l)
        print('[recall] r:', r)
        
        return recall_citations_l

    
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
            hit_with_score_l = self.recall(query)
            self.normalize_sr(hit_with_score_l)
            ret = self.rerank(query, hit_with_score_l)
            ret = self.citation_aggregation(ret)
            ret = self.global_citation_ranking(ret, hit_with_score_l)

            ret2 = []
            for citation,_ in ret:
                if citation in self.court_consideration_d or citation in self.law_d:
                    ret2.append(citation)

            pred = ';'.join(ret2[:limit])

            query_id_l.append(query_id)
            predicted_citations_l.append(pred)
            
        return pd.DataFrame({'query_id':query_id_l, 'predicted_citations':predicted_citations_l})
            
        
    def evaluate(self, start=0, stop=None):
        count = 0
        ret_l = []
        gold_citation_l = []
        for query, gold_citations in tqdm(zip(self.valid_df['query2'].tolist(), 
                                              self.valid_df['gold_citations'].tolist()), total=len(self.valid_df), desc="evaluation"):
            if count >= start:
                hit_with_score_l = self.recall(query)
                self.normalize_sr(hit_with_score_l)
                ret = self.rerank(query, hit_with_score_l)
                ret = self.citation_aggregation(ret)
                ret = self.global_citation_ranking(ret, hit_with_score_l)
    
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
        max_r = None
        max_p = None
        max_f1 = 0
        for limit in [5,10,15,20,25,30,35,40,45]:
            ret_l2 = [l[:limit] for l in ret_l]
            result = metric_utils.macro_f1(ret_l2, gold_citation_l)
            if max_f1 < result['macro_f1']:
                max_r = result['macro_recall']
                max_p = result['macro_precision']
                max_limit = limit
                max_f1 = result['macro_f1']
        print(f"[{max_limit}] r:", max_r, ", p:", max_p, "f1:",max_f1)
        