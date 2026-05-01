"""
Citation LambdaRank — Single Layer
===================================
排序粒度：group = query_id
  把同一 query 下所有 CC 里的所有 citations 放入同一排序组，
  让模型直接学「哪个 citation 对该 query 是 gold」。

CC 的 dense/sparse/rerank 分数作为 citation 的特征传入，
不需要单独的 CC Ranker。

评估指标：Recall@K, MRR, NDCG@K（无 threshold）

数据格式（train/valid JSONL，每行一个对象）：
  {"query_id": "...", "gold_citations": ["BGE ..."], "cc_list": [
    {"cc_id": "...", "dense_score": 0.1, "sparse_score": 0.1,
     "rerank_score": 0.2, "text": "..."}
  ]}

test JSONL 无 gold_citations 字段。
"""


import json
import re
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict
from sklearn.decomposition import PCA
import os
import os.path
import sys
import pickle

# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────
CITATION_RE = re.compile(
    r"""(?:
        SR\s*\d{3}(?:\.\d+)?(?:\s+Art\.?\s*\d+[a-z]?)?
      | BGE\s+\d{1,3}\s+[IVX]+[a-z]?\s+\d+(?:\s+E\.\s*\d+[a-z]?)?
      | Art\.?\s+\d+[a-z]?\s+(?:Abs\.?\s*\d+\s+)*(?:[A-Z][a-zA-ZäöüÄÖÜß0-9]*)
    )""",
    re.VERBOSE,
)

@dataclass
class CitationInstance:
    citation_id: str
    cc_id: str
    query_id: str
    query_text: str
    raw_query_text: str
    preceding_text: str
    following_text: str
    sentence_index: int
    total_sentences: int
    frequency_in_doc: int      # 在本 CC 内出现次数

    # 继承自所属 CC 的检索分数
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: float = 0.0

    is_bge: int = 0
    is_sr: int = 0
    is_art: int = 0

    total_citations_in_cc: int = 0
    citation_density: float = 0.0

    is_isolated:          int = 0
    # in_parenthesis:       int = 0
    # n_citations_in_ctx:   int = 0
    bm25_score: float = 0.0
    keyword_hit_rate: float = 0.0

    # ── query–doc 交互特征（由 FeatureBuilder 回填）──
    emb_cosine:   float = 0.0   # query 与 ctx embedding 的余弦相似度
    emb_dot:      float = 0.0   # 点积
    emb_l2:       float = 0.0   # L2 距离
    tfidf_cosine: float = 0.0   # query 与 ctx TF-IDF 向量的余弦
    tfidf_dot:    float = 0.0   # TF-IDF 点积（≈ 词汇重叠加权）

    # query 结构特征（按 query 级别，同一 query 的所有 inst 值相同）
    q_has_bge:        int   = 0
    q_has_sr:         int   = 0
    q_has_art:        int   = 0
    q_n_citations:    int   = 0
    q_n_tokens:       int   = 0
    q_is_short:       int   = 0
    q_is_long:        int   = 0
    q_is_question:    int   = 0
    q_n_commas:       int   = 0
    q_has_quotes:     int   = 0
    
    q_is_procedural:  int   = 0
    q_is_substantive: int   = 0
    q_is_interpretive: int  = 0
    q_domain_criminal:int   = 0
    q_domain_civil:   int   = 0
    q_domain_admin:   int   = 0
    q_lang_de:        int   = 0
    q_lang_fr:        int   = 0
    
    # query–citation 交互特征（每个 inst 不同）
    q_mentions_this_cit:   int   = 0
    q_citation_type_match: int   = 0
    q_keyword_hit_pre:     float = 0.0
    q_keyword_hit_post:    float = 0.0
    q_hit_asymmetry:       float = 0.0

    is_gold: int = 0


# pip install rank_bm25
from rank_bm25 import BM25Okapi

def compute_bm25_scores(instances: list[CitationInstance]) -> None:
    """
    按 query_id 分组，对该 query 下所有 citation 的 ctx 构建 BM25，
    计算每个 citation 的 ctx 与 query 的相关性分数，回填到 inst.bm25_score。
    原地修改，无返回值。
    """
    from collections import defaultdict

    # 按 query 分组
    query_buckets: dict[str, list[CitationInstance]] = defaultdict(list)
    for inst in instances:
        query_buckets[inst.query_id].append(inst)

    for query_id, insts in query_buckets.items():
        # tokenize
        query_tokens = _bm25_tok(insts[0].query_text)
        corpus = [_bm25_tok(inst.preceding_text + " " + inst.following_text)
                  for inst in insts]

        if not query_tokens or not any(corpus):
            continue

        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query_tokens)

        # 归一化到 [0, 1]
        max_s = scores.max()
        if max_s > 0:
            scores = scores / max_s

        for inst, score in zip(insts, scores):
            inst.bm25_score = float(score)


def _bm25_tok(text: str) -> list[str]:
    return re.findall(r"\b[a-züäöA-ZÜÄÖ]{2,}\b", text.lower())

def compute_keyword_hit_rate(instances: list[CitationInstance]) -> None:
    for inst in instances:
        query_tokens = set(_bm25_tok(inst.query_text))
        ctx_tokens   = set(_bm25_tok(inst.preceding_text + " " + inst.following_text))
        if not query_tokens:
            inst.keyword_hit_rate = 0.0
        else:
            inst.keyword_hit_rate = len(query_tokens & ctx_tokens) / len(query_tokens)

def query_structural_features(query_text: str, raw_query_text) -> dict:
    q = query_text
    raw_q = raw_query_text

    return {
        # ── citation 类型 ──────────────────────────
        "q_has_bge":      int(bool(re.search(r"BGE\s+\d", raw_q))),
        "q_has_sr":       int(bool(re.search(r"SR\s*\d", raw_q))),
        "q_has_art":      int(bool(re.search(r"Art\.?\s*\d", raw_q))),
        "q_n_citations":  len(CITATION_RE.findall(raw_q)),

        # ── 长度 ───────────────────────────────────
        "q_n_tokens":     len(q.split()),
        "q_n_chars":      len(q),
        "q_is_short":     int(len(q.split()) < 10),   # 关键词式
        "q_is_long":      int(len(q.split()) > 40),   # 长句式

        # ── 标点/结构 ──────────────────────────────
        "q_is_question":  int("?" in q),
        "q_n_commas":     q.count(","),
        "q_has_quotes":   int('"' in q or "«" in q),
    }

# 查询意图分类
PROCEDURAL_KW  = ["verfahren", "zuständigkeit", "frist", "rechtsmittel",
                   "beschwerde", "klage", "antrag"]
SUBSTANTIVE_KW = ["haftung", "schadenersatz", "vertrag", "eigentum",
                   "strafe", "verjährung", "anspruch"]
INTERPRETIVE_KW= ["auslegung", "analog", "sinn", "zweck", "ratio",
                   "bedeutung", "begrif"]

def query_intent_features(query_text: str) -> dict:
    q = query_text.lower()
    return {
        # 查询意图
        "q_is_procedural":   int(any(kw in q for kw in PROCEDURAL_KW)),
        "q_is_substantive":  int(any(kw in q for kw in SUBSTANTIVE_KW)),
        "q_is_interpretive": int(any(kw in q for kw in INTERPRETIVE_KW)),

        # 法律领域
        "q_domain_criminal": int(any(kw in q for kw in
            ["straf", "delikt", "täter", "opfer", "schuld"])),
        "q_domain_civil":    int(any(kw in q for kw in
            ["vertrag", "schuld", "eigentum", "miete", "kauf"])),
        "q_domain_admin":    int(any(kw in q for kw in
            ["verwaltung", "behörde", "bewilligung", "verfügung"])),
    }

def query_citation_interaction(
    inst: CitationInstance,
) -> dict:
    q   = inst.query_text.lower()
    raw_q = inst.raw_query_text.lower()
    cit = inst.citation_id.lower()
    ctx = (inst.preceding_text + " " + inst.following_text).lower()

    # query 里是否直接提到了这个 citation
    q_mentions_this_cit = int(
        inst.citation_id[:6].lower() in raw_q
    )

    # query 里的 citation 类型和 inst 的类型是否匹配
    q_wants_bge = int(bool(re.search(r"bge\s+\d", raw_q)))
    q_wants_sr  = int(bool(re.search(r"sr\s*\d",  raw_q)))
    type_match  = int(
        (q_wants_bge and inst.is_bge) or
        (q_wants_sr  and inst.is_sr)
    )

    # query 关键词在 citation 上下文的命中（已有 keyword_hit_rate，补充方向性）
    q_tokens   = set(_bm25_tok(inst.query_text))
    pre_tokens = set(_bm25_tok(inst.preceding_text))
    post_tokens= set(_bm25_tok(inst.following_text))

    pre_hit  = len(q_tokens & pre_tokens)  / (len(q_tokens) + 1e-8)
    post_hit = len(q_tokens & post_tokens) / (len(q_tokens) + 1e-8)

    return {
        "q_mentions_this_cit":  q_mentions_this_cit,
        "q_citation_type_match": type_match,
        "q_keyword_hit_pre":    pre_hit,
        "q_keyword_hit_post":   post_hit,
        "q_hit_asymmetry":      pre_hit - post_hit,  # 前置命中 vs 后置命中
    }

def compute_query_features(instances: list[CitationInstance]) -> None:
    """原地回填所有 query 相关特征"""
    from collections import defaultdict
    buckets = defaultdict(list)
    for inst in instances:
        buckets[inst.query_id].append(inst)

    for qid, insts in buckets.items():
        q = insts[0].query_text
        raw_q = insts[0].raw_query_text

        # query 级别特征（同 query 共享）
        struct  = query_structural_features(q, raw_q)
        intent  = query_intent_features(q)

        for inst in insts:
            # 回填 query 级别特征
            for k, v in {**struct, **intent}.items():
                if hasattr(inst, k):
                    setattr(inst, k, v)

            # 回填 query–citation 交互特征
            inter = query_citation_interaction(inst)
            for k, v in inter.items():
                if hasattr(inst, k):
                    setattr(inst, k, v)
# ─────────────────────────────────────────────
# Citation 抽取
# ─────────────────────────────────────────────

class CitationExtractor:
    def __init__(self, context_sentences: int = 2):
        self.context_sentences = context_sentences

    def _parse_citation_type(self, cit_id: str) -> dict:
        """
        解析 citation 类型及 BGE 内部结构。
        返回一个字典，供构造 CitationInstance 时解包。
        """
        is_bge = int(bool(re.match(r"BGE\s+\d", cit_id, re.IGNORECASE)))
        is_sr  = int(bool(re.match(r"SR\s*\d", cit_id, re.IGNORECASE)))
        is_art = int(bool(re.match(r"Art\.?\s*\d", cit_id, re.IGNORECASE)))

        return dict(
            is_bge=is_bge,
            is_sr=is_sr,
            is_art=is_art
        )
        
    def extract(
        self,
        cc_text: str,
        cc_id: str,
        query_id: str,
        gold_citations: list[str],
        dense_score: float = 0.0,
        sparse_score: float = 0.0,
        rerank_score: float = 0.0,
    ) -> list[CitationInstance]:
        sentences = self._split_sentences(cc_text)
        total_sents = len(sentences)

        freq_map: dict[str, int] = {}
        for c in CITATION_RE.findall(cc_text):
            key = self._normalize(c)
            freq_map[key] = freq_map.get(key, 0) + 1

        gold_normalized = {self._normalize(g) for g in gold_citations}
        seen: set[str] = set()
        instances: list[CitationInstance] = []

        total_citations_in_cc = len(freq_map)   # CC内不同citation的种数
        citation_density = total_citations_in_cc / max(total_sents, 1)  # 每句平均citation数


        for sent_idx, sent in enumerate(sentences):
            for m in CITATION_RE.finditer(sent):
                cit_id = self._normalize(m.group(0))
                if cit_id in seen:
                    continue
                seen.add(cit_id)
                pre, post = self._context(sentences, sent_idx)

                ctx = pre + " " + sent + " " + post

                # ── citation role features ──────────────────────
                n_cit_in_ctx = len(CITATION_RE.findall(ctx))
                is_isolated  = int(n_cit_in_ctx == 1)

                # 检查 citation 是否出现在括号内
                in_paren = False
                for par_m in re.finditer(r'\(([^)]{0,200})\)', ctx):
                    if cit_id[:6].lower() in par_m.group(1).lower():
                        in_paren = True
                        break
                # ────────────────────────────────────────────────

                type_feats = self._parse_citation_type(cit_id)

                instances.append(CitationInstance(
                    citation_id=cit_id,
                    cc_id=cc_id,
                    query_id=query_id,
                    query_text="",
                    raw_query_text="",
                    preceding_text=pre,
                    following_text=post,
                    sentence_index=sent_idx,
                    total_sentences=total_sents,
                    frequency_in_doc=freq_map.get(cit_id, 1),
                    dense_score=dense_score,
                    sparse_score=sparse_score,
                    rerank_score=rerank_score,
                    total_citations_in_cc=total_citations_in_cc,
                    citation_density=citation_density,
                    is_isolated=is_isolated,
                    is_bge = type_feats['is_bge'],
                    is_sr = type_feats['is_sr'],
                    is_art = type_feats['is_art'],
                    # in_parenthesis=int(in_paren),
                    # n_citations_in_ctx=n_cit_in_ctx,
                    is_gold=int(cit_id in gold_normalized),
                ))
        return instances

    def _context(self, sentences, idx):
        n = self.context_sentences
        pre = " ".join(sentences[max(0, idx - n): idx])
        post = " ".join(sentences[idx + 1: idx + 1 + n])
        return pre, post

    def _split_sentences(self, text):
        text = re.sub(r"\b(E|Nr|Abs|Art|Ziff|bzw|ca|vgl|usw|etc)\.", r"\1<DOT>", text)
        sents = re.split(r"(?<=[.!?])\s+(?=[A-ZÜÄÖ\d])", text)
        return [s.replace("<DOT>", ".").strip() for s in sents if s.strip()]

    def _normalize(self, raw):
        # return re.sub(r"\s+", " ", raw.strip().upper())
        return re.sub(r"\s+", " ", raw.strip())



# ─────────────────────────────────────────────
# 特征工程
# ─────────────────────────────────────────────
import random

def _safe_rerank(r):
    if random.random() < 0.1:
        return 0.0
    return r

def _safe_dense(d):
    if random.random() < 0.1:
        return 0.0
    return d

def _safe_sparse(s):
    if random.random() < 0.1:
        return 0.0
    return s

class CitationFeatureBuilder:
    POSITIVE_KW = [
        "grundlegend", "massgebend", "wegweisend", "ständige rechtsprechung",
        "bestätigt", "gemäss", "analog", "entsprechend", "gilt", "ist anzuwenden",
        "ergibt sich", "findet anwendung", "in rechtlicher hinsicht",
        "ratio decidendi", "leitentscheid",
    ]
    NEGATIVE_KW = [
        "vgl.", "anders als", "entgegen", "zweifelhaft", "offen gelassen",
        "nicht anwendbar", "abweichend", "kritisch",
    ]
    # ── 语义子组关键词 ────────────────────────────────────────────
    # 权威性：说明这个 citation 是主流判例／权威来源
    AUTHORITY_KW = [
        "grundlegend", "massgebend", "wegweisend",
        "ständige rechtsprechung", "leitentscheid",
        "konstante rechtsprechung", "gefestigte rechtsprechung",
    ]

    # 适用性：说明这个 citation 被直接援引适用
    APPLICATION_KW = [
        "gemäss", "analog", "entsprechend", "gilt",
        "ist anzuwenden", "findet anwendung",
        "ratio decidendi", "in anwendung",
        "gestützt auf", "im sinne von",
    ]

    # 确认性：说明这个 citation 验证／支持了某个观点
    CONFIRMATION_KW = [
        "bestätigt", "ergibt sich", "in rechtlicher hinsicht",
        "wie das bundesgericht", "bereits entschieden",
        "wie dargelegt", "in diesem sinne",
    ]

    # 否定性：说明这个 citation 被质疑／区分／不适用
    NEGATION_KW = [
        "vgl.", "anders als", "entgegen", "zweifelhaft",
        "offen gelassen", "nicht anwendbar", "abweichend",
        "kritisch", "nicht einschlägig", "zu unterscheiden",
        "kann nicht", "ist nicht", "widerspricht",
    ]

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def fit(self, instances: list[CitationInstance]):
        corpus = [inst.preceding_text + " " + inst.following_text for inst in instances]
        self._build_vocab(corpus, max_features=500)
        return self

    def transform(self, instances: list[CitationInstance]) -> np.ndarray:
        return np.array([self._vec(inst) for inst in instances], dtype=np.float32)

    def _semantic_group_features(self, inst: CitationInstance) -> list[float]:
        pre  = inst.preceding_text.lower()
        post = inst.following_text.lower()
        ctx  = pre + " " + post

        # 各子组在整体上下文的计数
        authority_score    = sum(1 for kw in self.AUTHORITY_KW    if kw in ctx)
        application_score  = sum(1 for kw in self.APPLICATION_KW  if kw in ctx)
        confirmation_score = sum(1 for kw in self.CONFIRMATION_KW if kw in ctx)
        negation_score     = sum(1 for kw in self.NEGATION_KW     if kw in ctx)

        # 左右方向区分
        authority_pre      = sum(1 for kw in self.AUTHORITY_KW    if kw in pre)
        authority_post     = sum(1 for kw in self.AUTHORITY_KW    if kw in post)
        application_pre    = sum(1 for kw in self.APPLICATION_KW  if kw in pre)
        application_post   = sum(1 for kw in self.APPLICATION_KW  if kw in post)
        negation_pre       = sum(1 for kw in self.NEGATION_KW     if kw in pre)
        negation_post      = sum(1 for kw in self.NEGATION_KW     if kw in post)

        # 正向子组总分（权威 + 适用 + 确认）
        total_pos = authority_score + application_score + confirmation_score

        return [
            # 子组计数
            float(authority_score),
            float(application_score),
            float(confirmation_score),
            float(negation_score),
            # 正负对比
            float(total_pos),
            float(total_pos - negation_score),      # 净正向分
            float(total_pos / max(negation_score + 1, 1)),  # 正负比
            # 方向特征
            float(authority_pre),
            float(authority_post),
            float(application_pre),
            float(application_post),
            float(negation_pre),
            float(negation_post),
            float(application_pre - application_post),   # 前置适用 vs 后置适用
        ]

    def feature_names(self) -> list[str]:
        base = [
            "dense_score", "sparse_score", "rerank_score",
            "dense_x_rerank", "sparse_x_rerank",
            "dense_plus_sparse", "rerank_minus_dense",
            "pos_relative", "pos_in_first_quarter", "pos_in_last_quarter",
            "freq_raw", "freq_log", "freq_normalized",
            "ctx_pos_kw", "ctx_neg_kw", "ctx_kw_ratio",
            "ctx_pre_len", "ctx_post_len",
            "total_citations_in_cc", "citation_density",
            "is_isolated", #, "in_parenthesis", "n_citations_in_ctx",  # 新增

            "is_bge", "is_sr", "is_art",
            "bm25_score",
            "keyword_hit_rate",
            # ── 新增：query–doc 交互 ──────────────────
            "emb_cosine",
            "emb_dot",
            "emb_l2",
            "tfidf_cosine",
            "tfidf_dot",
            
            # 语义子组
            "authority_score",
            "application_score",
            "confirmation_score",
            "negation_score",
            "total_pos_score",
            "net_pos_score",
            "pos_neg_ratio",
            "authority_pre",
            "authority_post",
            "application_pre",
            "application_post",
            "negation_pre",
            "negation_post",
            "application_direction",
        ]

        query_feature_names = [
            # ── query 结构特征 ──────────────────────────
            "q_has_bge", "q_has_sr", "q_has_art",
            "q_n_citations",
            "q_n_tokens", "q_is_short", "q_is_long",
            "q_is_question", "q_n_commas", "q_has_quotes",
            # ── query 意图特征 ──────────────────────────
            "q_is_procedural", "q_is_substantive", "q_is_interpretive",
            "q_domain_criminal", "q_domain_civil", "q_domain_admin",
            # ── query–citation 交互特征 ─────────────────
            "q_mentions_this_cit",
            "q_citation_type_match",
            "q_keyword_hit_pre",
            "q_keyword_hit_post",
            "q_hit_asymmetry",
        ]
        return base + query_feature_names # + [f"tfidf_{w}" for w in sorted(self.vocab, key=self.vocab.get)]



    def _vec(self, inst: CitationInstance) -> list[float]:
        rel_pos = inst.sentence_index / max(inst.total_sentences - 1, 1)
        ctx = (inst.preceding_text + " " + inst.following_text).lower()
        pos_kw = sum(1 for kw in self.POSITIVE_KW if kw in ctx)
        neg_kw = sum(1 for kw in self.NEGATIVE_KW if kw in ctx)
        # d, s, r = _safe_dense(inst.dense_score), _safe_sparse(inst.sparse_score), _safe_rerank(inst.rerank_score)

        d, s, r = inst.dense_score, inst.sparse_score, inst.rerank_score
        base = [
            d, s, r,
            d * r, s * r, d + s, r - d,
            rel_pos,
            float(rel_pos < 0.25),
            float(rel_pos > 0.75),
            float(inst.frequency_in_doc),
            math.log1p(inst.frequency_in_doc),
            inst.frequency_in_doc / 5.0,
            float(pos_kw), float(neg_kw),
            pos_kw / (pos_kw + neg_kw + 1),
            float(len(inst.preceding_text.split())),
            float(len(inst.following_text.split())),
            float(inst.total_citations_in_cc),
            inst.citation_density,
            float(inst.is_isolated),        # 新增
            # float(inst.in_parenthesis),     # 新增
            # float(inst.n_citations_in_ctx), # 新增
            float(inst.is_bge),
            float(inst.is_sr),
            float(inst.is_art),
            inst.bm25_score,
            inst.keyword_hit_rate,
            # ── 新增 ──────────────────────────────────
            inst.emb_cosine,
            inst.emb_dot,
            inst.emb_l2,
            inst.tfidf_cosine,
            inst.tfidf_dot,
        ]

        query_feature = [
            # ── query 结构特征 ──────────────────────────
            float(inst.q_has_bge),
            float(inst.q_has_sr),
            float(inst.q_has_art),
            float(inst.q_n_citations),
            float(inst.q_n_tokens),
            float(inst.q_is_short),
            float(inst.q_is_long),
            float(inst.q_is_question),
            float(inst.q_n_commas),
            float(inst.q_has_quotes),
            # ── query 意图特征 ──────────────────────────
            float(inst.q_is_procedural),
            float(inst.q_is_substantive),
            float(inst.q_is_interpretive),
            float(inst.q_domain_criminal),
            float(inst.q_domain_civil),
            float(inst.q_domain_admin),
            # ── query–citation 交互特征 ─────────────────
            float(inst.q_mentions_this_cit),
            float(inst.q_citation_type_match),
            inst.q_keyword_hit_pre,
            inst.q_keyword_hit_post,
            inst.q_hit_asymmetry,
        ]
        result = base + self._semantic_group_features(inst) + query_feature
        assert len(result) == len(self.feature_names()), \
            f"特征数量不匹配: _vec={len(result)}, feature_names={len(self.feature_names())}"
        return result

    def _build_vocab(self, corpus, max_features):
        from collections import Counter
        tc, df = Counter(), Counter()
        for doc in corpus:
            toks = set(self._tok(doc))
            df.update(toks)
            tc.update(self._tok(doc))
        N = len(corpus)
        scored = {
            w: tc[w] * math.log(N / (df[w] + 1))
            for w in tc if len(w) > 2 and df[w] > 1
        }
        top = sorted(scored, key=scored.get, reverse=True)[:max_features]
        self.vocab = {w: i for i, w in enumerate(top)}

    def _tfidf(self, text):
        if not self.vocab:
            return []
        from collections import Counter
        toks = self._tok(text)
        tf = Counter(toks)
        vec = [0.0] * len(self.vocab)
        for t, c in tf.items():
            if t in self.vocab:
                vec[self.vocab[t]] = c / max(len(toks), 1)
        return vec

    def _tok(self, text):
        return re.findall(r"\b[a-züäöA-ZÜÄÖ]{3,}\b", text.lower())

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import hashlib

class EmbeddingCitationFeatureBuilder(CitationFeatureBuilder):

    def __init__(
        self,
        model_name: str = "deepset/gbert-base",   # 德语专用，比 multilingual 更准
        n_components: int = 128,
        batch_size: int = 64,
        device: str = None,
        cache_path: str = "../data/ml5/emb_cache.pkl"
    ):
        super().__init__()
        self.model_name   = model_name
        self.n_components = n_components
        self.batch_size   = batch_size
        self.device       = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._tokenizer = None
        self._bert      = None
        self._pca: Optional[PCA] = None

        self.cache_path = cache_path
        self._emb_cache: dict[str, np.ndarray] = {}

        # 尝试加载 cache
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    self._emb_cache = pickle.load(f)
                print(f"[EmbeddingCache] loaded {len(self._emb_cache)} entries")
            except Exception:
                print("[EmbeddingCache] load failed, start fresh")

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _save_cache(self):
        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump(self._emb_cache, f)
            print(f"[EmbeddingCache] saved {len(self._emb_cache)} entries")
        except Exception as e:
            print(f"[EmbeddingCache] save failed: {e}")
            
    def fit(self, instances):
        super().fit(instances)   # TF-IDF 词表

        # 初始化模型
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._bert = AutoModel.from_pretrained(self.model_name).to(self.device)
        self._bert.eval()

        # fit PCA
        raw_embs = self._encode(self._to_texts(instances))
        self._pca = PCA(n_components=self.n_components, random_state=42)
        self._pca.fit(raw_embs)

        explained = self._pca.explained_variance_ratio_.sum()
        print(f"[EmbeddingFeatureBuilder] PCA {self.n_components}d "
              f"explained variance: {explained:.3f}")

        # 回填 fit 阶段的交互特征
        self._fill_interaction_features(instances, raw_embs)
        
        # 🔥 保存 cache
        self._save_cache()
    
        return self

    def transform(self, instances):
        tfidf_feats = super().transform(instances)

        raw_embs  = self._encode(self._to_texts(instances))
        emb_feats = self._pca.transform(raw_embs).astype(np.float32)

        # 回填交互特征到 instance（_vec 会读取）
        self._fill_interaction_features(instances, raw_embs)

        if getattr(self, "_cache_dirty", False):  # ← 只在有新 embedding 时写盘
            self._save_cache()
            self._cache_dirty = False

        return super().transform(instances)
        
    def feature_names(self):
        return super().feature_names() # + [f"emb_pca_{i}" for i in range(self.n_components)]

    # ── 内部 ────────────────────────────────────────────────────────────

    @staticmethod
    def _to_texts(instances):
        return [
            inst.preceding_text + " [SEP] " + inst.following_text
            for inst in instances
        ]

    @torch.no_grad()
    def _encode(self, texts: list[str]) -> np.ndarray:
        results = [None] * len(texts)

        uncached_idx = []
        uncached_texts = []

        # ── 1. 查 cache ─────────────────────────
        for i, text in enumerate(texts):
            key = self._hash(text)
            if key in self._emb_cache:
                results[i] = self._emb_cache[key]
            else:
                uncached_idx.append(i)
                uncached_texts.append(text)

        # ── 2. 只计算未缓存部分 ─────────────────
        if uncached_texts:
            new_embs = []
    
            for i in range(0, len(uncached_texts), self.batch_size):
                batch = uncached_texts[i : i + self.batch_size]
    
                enc = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ).to(self.device)
    
                out  = self._bert(**enc)
    
                mask = enc["attention_mask"].unsqueeze(-1).float()
                emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
                emb  = F.normalize(emb, dim=-1)
    
                new_embs.append(emb.cpu().numpy())
    
            new_embs = np.concatenate(new_embs, axis=0)

            self._cache_dirty = True   # ← 标记有新内容
    
            # ── 3. 写回 cache ───────────────────
            for idx, text, emb in zip(uncached_idx, uncached_texts, new_embs):
                key = self._hash(text)
                self._emb_cache[key] = emb
                results[idx] = emb
    
        return np.stack(results).astype(np.float32)
    
        # all_embs = []
        # for i in range(0, len(texts), self.batch_size):
        #     batch = texts[i : i + self.batch_size]
        #     enc = self._tokenizer(
        #         batch,
        #         padding=True,
        #         truncation=True,
        #         max_length=256,
        #         return_tensors="pt",
        #     ).to(self.device)

        #     out  = self._bert(**enc)

        #     # mean pooling（比 [CLS] 对短文本更稳定）
        #     mask = enc["attention_mask"].unsqueeze(-1).float()
        #     emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        #     emb  = F.normalize(emb, dim=-1)   # L2 归一化

        #     all_embs.append(emb.cpu().numpy())

        # return np.concatenate(all_embs, axis=0).astype(np.float32)

        # ── 新增：计算并回填 query–doc 交互特征 ─────────────────────────────

    def _fill_interaction_features(
            self,
            instances: list[CitationInstance],
            ctx_embs: np.ndarray,          # [N, raw_bert_dim]，已 L2 归一化
        ) -> None:
            """
            对每个 instance：
              - 编码 query embedding（有 cache，基本无额外开销）
              - 计算 emb_cosine / emb_dot / emb_l2
              - 计算 tfidf_cosine / tfidf_dot（用父类 _tfidf 方法）
            结果原地写回 instance 字段。
            """
            # 1. 编码所有 query（去重，利用 cache）
            query_texts = [inst.query_text for inst in instances]
            query_embs  = self._encode(query_texts)   # [N, raw_bert_dim]，已归一化
    
            # 2. 构建 tfidf 向量（用父类已 fit 的词表）
            ctx_tfidf_vecs   = np.array(
                [self._tfidf(inst.preceding_text + " " + inst.following_text)
                 for inst in instances],
                dtype=np.float32,
            )   # [N, vocab_size]
    
            query_tfidf_vecs = np.array(
                [self._tfidf(inst.query_text) for inst in instances],
                dtype=np.float32,
            )   # [N, vocab_size]
    
            # 3. 逐样本计算交互并回填
            for i, inst in enumerate(instances):
                q_emb = query_embs[i]    # 已 L2 归一化
                d_emb = ctx_embs[i]      # 已 L2 归一化
    
                inst.emb_cosine = float(np.dot(q_emb, d_emb))          # 归一化后点积 = 余弦
                inst.emb_dot    = float(np.dot(q_emb, d_emb))          # 同上（显式保留语义）
                inst.emb_l2     = float(np.linalg.norm(q_emb - d_emb))
    
                q_tfidf = query_tfidf_vecs[i]
                d_tfidf = ctx_tfidf_vecs[i]
    
                q_norm  = np.linalg.norm(q_tfidf)
                d_norm  = np.linalg.norm(d_tfidf)
    
                inst.tfidf_dot    = float(np.dot(q_tfidf, d_tfidf))
                inst.tfidf_cosine = (
                    float(inst.tfidf_dot / (q_norm * d_norm + 1e-8))
                    if q_norm > 0 and d_norm > 0 else 0.0
                )
        
# ─────────────────────────────────────────────
# 排序评估指标
# ─────────────────────────────────────────────

def _dcg(relevances: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(relevances))


def ranking_metrics(
    groups: dict[str, list[tuple[int, float]]],
    ks: list[int] = None,
) -> dict:
    """
    groups: {group_id: [(is_gold, score), ...]}
    对每个 group 按 score 降序排列后计算指标，最后取均值。
    跳过无正样本的 group（对 MRR/NDCG 无意义）。
    """
    ks = ks or [1, 3, 5]
    recall_at  = {k: [] for k in ks}
    ndcg_at    = {k: [] for k in ks}
    mrr_scores = []

    for gid, items in groups.items():
        if not any(lab for lab, _ in items):
            continue
        sorted_labels = [lab for lab, _ in sorted(items, key=lambda x: -x[1])]
        n_pos = sum(sorted_labels)
        ideal = sorted(sorted_labels, reverse=True)

        mrr = next((1.0 / (r + 1) for r, lab in enumerate(sorted_labels) if lab), 0.0)
        mrr_scores.append(mrr)

        for k in ks:
            top_k = sorted_labels[:k]
            recall_at[k].append(sum(top_k) / max(n_pos, 1))
            idcg = _dcg(ideal[:k])
            ndcg_at[k].append(_dcg(top_k) / idcg if idcg > 0 else 0.0)

    result = {"MRR": float(np.mean(mrr_scores)) if mrr_scores else 0.0}
    for k in ks:
        result[f"Recall@{k}"] = float(np.mean(recall_at[k])) if recall_at[k] else 0.0
        result[f"NDCG@{k}"]   = float(np.mean(ndcg_at[k]))   if ndcg_at[k]   else 0.0
    return result


# ─────────────────────────────────────────────
# 单层 LambdaRank
# ─────────────────────────────────────────────

class CitationRanker:
    """
    单层 LambdaRank：group = query_id

    同一 query 下所有 CC 里的 citations 构成一个排序组，
    模型直接学「哪个 citation 对该 query 是 gold」。
    CC 的检索分数作为 citation 的特征传入。
    """

    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        early_stopping_rounds: int = 50,
        eval_at: list[int] = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_at = eval_at or [1, 3, 5, 200]

        self.model = None
        self.feature_builder = CitationFeatureBuilder()
        self.best_iteration_: Optional[int] = None
        self.valid_metrics_: Optional[dict] = None

    # ── 内部：instances → LambdaRank 所需数组 ───────────────────────────

    def _to_arrays(self, instances: list[CitationInstance]):
        """
        按 query_id 分组，返回 (X, y, groups)。
        groups[i] = 第 i 个 query 下的 citation 数量（LightGBM group 格式）。
        组内顺序：同一 query 的所有 citations 连续排列。
        """
        compute_bm25_scores(instances)  # ← 新增
        compute_keyword_hit_rate(instances)
        compute_query_features(instances)
        
        # 按 query_id 保持稳定顺序
        from collections import OrderedDict
        buckets: dict[str, list[CitationInstance]] = OrderedDict()
        for inst in instances:
            buckets.setdefault(inst.query_id, []).append(inst)

        ordered: list[CitationInstance] = []
        groups: list[int] = []
        for qid, insts in buckets.items():
            ordered.extend(insts)
            groups.append(len(insts))

        X = self.feature_builder.transform(ordered)
        y = np.array([inst.is_gold for inst in ordered], dtype=np.float32)
        return X, y, groups, ordered

    # ── fit ──────────────────────────────────────────────────────────────

    def fit(
        self,
        train_instances: list[CitationInstance],
        valid_instances: Optional[list[CitationInstance]] = None,
    ) -> "CitationRanker":
        # 词表在训练集上 fit
        self.feature_builder.fit(train_instances)

        X_train, y_train, groups_train, _ = self._to_arrays(train_instances)

        import lightgbm as lgb

        callbacks = [lgb.log_evaluation(period=20)]
        if valid_instances:
            callbacks.append(lgb.early_stopping(
                stopping_rounds=self.early_stopping_rounds,
                verbose=True,
            ))

        self.model = lgb.LGBMRanker(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            objective="lambdarank",
            lambdarank_truncation_level=50, # max(self.eval_at),
            verbose=-1,
            seed=43
        )

        feat_names = self.feature_builder.feature_names()
        fit_kwargs = dict(
            X=pd.DataFrame(X_train, columns=feat_names),
            y=y_train,
            group=groups_train,
            callbacks=callbacks,
            eval_metric=[f"ndcg@{k}" for k in self.eval_at],
        )

        if valid_instances:
            X_val, y_val, groups_val, _ = self._to_arrays(valid_instances)
            fit_kwargs["eval_set"]   = [(pd.DataFrame(X_val, columns=feat_names), y_val)]
            fit_kwargs["eval_group"] = [groups_val]

        self.model.fit(**fit_kwargs)
        self.best_iteration_ = getattr(self.model, "best_iteration_", None)

        if valid_instances:
            self.valid_metrics_ = self.evaluate(valid_instances)
            metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in self.valid_metrics_.items())
            print(f"\n[LambdaRank] 早停于第 {self.best_iteration_} 轮")
            print(f"[LambdaRank] Valid  {metrics_str}")

        return self

    # ── 推理 ─────────────────────────────────────────────────────────────

    def predict_scores(self, instances: list[CitationInstance]) -> np.ndarray:
        """返回排序分数，值越大越可能是 gold"""
        X = self.feature_builder.transform(instances)
        feat_names = self.feature_builder.feature_names()
        return self.model.predict(pd.DataFrame(X, columns=feat_names))

    # ── 评估 ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        instances: list[CitationInstance],
        ks: list[int] = None,
    ) -> dict:
        """按 query_id 分组计算 Recall@K, MRR, NDCG@K"""
        scores = self.predict_scores(instances)
        groups: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for inst, score in zip(instances, scores):
            groups[inst.query_id].append((inst.is_gold, float(score)))
        return ranking_metrics(groups, ks or self.eval_at)

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        names = self.feature_builder.feature_names()
        coefs = self.model.feature_importances_
        df = pd.DataFrame({"feature": names[:len(coefs)], "importance": coefs})
        return df.sort_values("importance", ascending=False).head(top_n)


# ─────────────────────────────────────────────
# DataLoader
# ─────────────────────────────────────────────

class DataLoader:
    def __init__(self, context_sentences: int = 2):
        self.extractor = CitationExtractor(context_sentences=context_sentences)
        self._query_map: dict[str, str] = {}   # query_id → query text
        self._raw_query_map: dict[str, str] = {}   # query_id → 德语原文

    def load_query_map(self, 
                       path: str, 
                       query_col: str = "query", 
                       raw_query_col: str = None, # 英语列名，None 表示没有
                      ):
        """
        预加载 query_id → query text 的映射。
        valid 文件的列名是 query2，通过 query_col 参数指定。
        """
        df = pd.read_csv(path)
        self._query_map.update(
            dict(zip(df["query_id"].astype(str), df[query_col].astype(str)))
        )
        if raw_query_col and raw_query_col in df.columns:
            self._raw_query_map.update(
                dict(zip(df["query_id"].astype(str), df[raw_query_col].astype(str)))
            )
        print(f"[DataLoader] 加载 {len(df)} 条 query from {path}")
        return self
        
    @staticmethod
    def _read_jsonl(path: str) -> list[dict]:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSONL 第 {lineno} 行解析失败: {e}")
        return records

    # ── train / valid ─────────────────────────────────────────────────────

    def load_file(self, path: str) -> list[CitationInstance]:
        return self.load(self._read_jsonl(path))

    def load(self, data: list[dict]) -> list[CitationInstance]:
        all_instances: list[CitationInstance] = []
        for entry in data:
            query_id = entry["query_id"]
            gold_citations = entry["gold_citations"]
            for cc in entry["cc_list"]:
                insts = self.extractor.extract(
                    cc_text=cc["text"],
                    cc_id=cc["cc_id"],
                    query_id=query_id,
                    gold_citations=gold_citations,
                    dense_score=cc.get("dense_score", 0.0),
                    sparse_score=cc.get("sparse_score", 0.0),
                    rerank_score=cc.get("rerank_score", 0.0),
                )

                # 回填 query_text
                query_text = self._query_map.get(str(query_id), "")
                raw_query_text = self._raw_query_map.get(str(query_id), query_text)
                for inst in insts:
                    inst.query_text = query_text
                    inst.raw_query_text = raw_query_text
                    
                all_instances.extend(insts)
        return all_instances

    # ── test predict ──────────────────────────────────────────────────────

    def predict_file(
        self,
        path: str,
        ranker: CitationRanker,
        output_path: Optional[str] = None,
    ) -> list[dict]:
        """
        从 test JSONL 文件加载并预测，可选写出结果 JSONL。

        输出格式（每行一个 query）：
          {"query_id": "test_0001", "cc_list": [
            {"cc_id": "cc_t01", "citations": [
              {"citation_id": "BGE 138 IV 1", "score": 1.23, "rank": 1},
              {"citation_id": "SR 311.0",     "score": 0.87, "rank": 2}
            ]}
          ]}

        rank 是该 citation 在所属 query 内所有 citations 中的全局排名。
        """
        data = self._read_jsonl(path)
        results = self.predict_dataset(data, ranker)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                for record in results:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return results

    def predict_dataset(
        self,
        data: list[dict],
        ranker: CitationRanker,
    ) -> list[dict]:
        """
        对 test 数据打分并排序（无 gold_citations 字段）。

        每个 query 下所有 CC 的 citations 统一排序（group = query_id），
        rank 是该 citation 在本 query 内的全局排名（1 = 最可能是 gold）。
        """
        # 1. 抽取所有 instances（gold_citations 传空列表）
        all_instances: list[CitationInstance] = []
        # 记录每个 instance 属于哪个 query / cc，供后续还原结构
        meta: list[tuple[str, str]] = []   # (query_id, cc_id)

        for entry in data:
            query_id = entry["query_id"]
            for cc in entry["cc_list"]:
                insts = self.extractor.extract(
                    cc_text=cc["text"],
                    cc_id=cc["cc_id"],
                    query_id=query_id,
                    gold_citations=[],
                    dense_score=cc.get("dense_score", 0.0),
                    sparse_score=cc.get("sparse_score", 0.0),
                    rerank_score=cc.get("rerank_score", 0.0),
                )
                # 回填 query_text
                query_text = self._query_map.get(str(query_id), "")
                for inst in insts:
                    inst.query_text = query_text
                
                all_instances.extend(insts)
                meta.extend((query_id, cc["cc_id"]) for _ in insts)

        if not all_instances:
            return [{"query_id": e["query_id"], "cc_list": []} for e in data]

        # 2. 统一打分
        scores = ranker.predict_scores(all_instances)

        # 3. 按 query_id 分组，计算全局 rank
        #    同一 query 内所有 citations（跨 CC）按 score 降序排名
        query_items: dict[str, list[tuple[int, float, CitationInstance]]] = defaultdict(list)
        for idx, (inst, score) in enumerate(zip(all_instances, scores)):
            query_items[inst.query_id].append((idx, float(score), inst))

        # global_rank[idx] = 该 citation 在其 query 内的排名（1-based）
        global_rank: dict[int, int] = {}
        for qid, items in query_items.items():
            for rank, (idx, score, inst) in enumerate(
                sorted(items, key=lambda x: -x[1]), start=1
            ):
                global_rank[idx] = rank

        # 4. 还原输出结构（保持输入的 query / cc 顺序）
        inst_ptr = 0
        results = []
        for entry in data:
            query_id = entry["query_id"]
            cc_results = []
            for cc in entry["cc_list"]:
                # 找属于这个 cc 的 instances
                cc_citations = []
                while inst_ptr < len(all_instances) and \
                      all_instances[inst_ptr].cc_id == cc["cc_id"] and \
                      all_instances[inst_ptr].query_id == query_id:
                    inst = all_instances[inst_ptr]
                    score = float(scores[inst_ptr])
                    cc_citations.append({
                        "citation_id": inst.citation_id,
                        "score":       round(score, 4),
                        "rank":        global_rank[inst_ptr],  # query 内全局排名
                    })
                    inst_ptr += 1
                # cc 内按 score 降序排列，方便阅读
                cc_citations.sort(key=lambda x: x["rank"])
                cc_results.append({"cc_id": cc["cc_id"], "citations": cc_citations})
            results.append({"query_id": query_id, "cc_list": cc_results})

        return results

    def sample_instances(
        self,
        instances: list[CitationInstance],
        neg_pos_ratio: int = 10,
        hard_neg_keep: int = 20,
    ) -> list[CitationInstance]:
        """
        按 query_id 做 1:neg_pos_ratio 负采样

        规则：
        - 保留所有正样本
        - 保留 top-K hard negatives（按 rerank_score）
        - 其余负样本随机采样
        """

        from collections import defaultdict
        import random

        buckets = defaultdict(list)
        for inst in instances:
            buckets[inst.query_id].append(inst)

        sampled = []

        for qid, insts in buckets.items():
            pos = [x for x in insts if x.is_gold == 1]
            neg = [x for x in insts if x.is_gold == 0]

            if not pos:
                # 没有正样本的 query 直接跳过（训练无意义）
                continue

            # 🔥 1. 选 hard negatives（按 rerank_score 排）
            neg_sorted = sorted(neg, key=lambda x: -x.rerank_score)
            hard_negs = neg_sorted[:hard_neg_keep]

            # 🔥 2. 剩余 negatives
            remaining_negs = neg_sorted[hard_neg_keep:]

            # 🔥 3. 计算需要多少负样本
            target_neg = neg_pos_ratio * len(pos)
            already = len(hard_negs)

            if already >= target_neg:
                sampled_negs = hard_negs[:target_neg]
            else:
                need = target_neg - already
                random.shuffle(remaining_negs)
                sampled_negs = hard_negs + remaining_negs[:need]

            sampled.extend(pos)
            sampled.extend(sampled_negs)

        return sampled

    def sample_instances_multisources(
        self,
        instances: list[CitationInstance],
        neg_pos_ratio: int = 10,
        hard_neg_keep: int = 20,
    ) -> list[CitationInstance]:
        """
        Hard negative 选取依据：多信号融合分数
        
        score = rerank_score                        # CC 语义相关（粗粒度）
              + bm25_score                          # citation 上下文与 query 的词汇匹配（细粒度）
              + keyword_hit_rate                    # query 关键词覆盖率
              + pos_kw_signal                       # 上下文含权威性关键词（如 massgebend）
              - 0.5 * citation_density              # 惩罚：citation 扎堆的 CC 里随便一个都高分
        
        多信号融合比单一 rerank_score 更能识别「citation 级别的难负例」。
        """
        from collections import defaultdict
        import random
    
        POSITIVE_KW = [
            "grundlegend", "massgebend", "wegweisend",
            "ständige rechtsprechung", "bestätigt", "gemäss",
            "analog", "entsprechend", "gilt", "ist anzuwenden",
            "ergibt sich", "findet anwendung", "leitentscheid",
        ]
    
        def fusion_score(inst: CitationInstance) -> float:
            ctx = (inst.preceding_text + " " + inst.following_text).lower()
            pos_kw = sum(1 for kw in POSITIVE_KW if kw in ctx)
            return (
                inst.rerank_score
                + inst.bm25_score
                + inst.keyword_hit_rate
                + 0.3 * pos_kw
                - 0.5 * inst.citation_density      # 高密度 CC 里的 citation 含金量低
            )
    
        buckets: dict[str, list[CitationInstance]] = defaultdict(list)
        for inst in instances:
            buckets[inst.query_id].append(inst)
    
        sampled: list[CitationInstance] = []
    
        for qid, insts in buckets.items():
            pos = [x for x in insts if x.is_gold == 1]
            neg = [x for x in insts if x.is_gold == 0]
    
            if not pos:
                continue
    
            neg_sorted    = sorted(neg, key=fusion_score, reverse=True)
            hard_negs     = neg_sorted[:hard_neg_keep]
            remaining     = neg_sorted[hard_neg_keep:]
    
            target_neg = neg_pos_ratio * len(pos)
            already    = len(hard_negs)
    
            if already >= target_neg:
                sampled_negs = hard_negs[:target_neg]
            else:
                random.shuffle(remaining)
                sampled_negs = hard_negs + remaining[:target_neg - already]
    
            sampled.extend(pos)
            sampled.extend(sampled_negs)
    
        return sampled
