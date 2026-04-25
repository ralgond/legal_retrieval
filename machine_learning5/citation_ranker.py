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

# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class CitationInstance:
    citation_id: str
    cc_id: str
    query_id: str

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

    total_citations_in_cc: int = 0
    citation_density: float = 0.0

    is_isolated:          int = 0
    # in_parenthesis:       int = 0
    # n_citations_in_ctx:   int = 0

    is_gold: int = 0


# ─────────────────────────────────────────────
# Citation 抽取
# ─────────────────────────────────────────────

class CitationExtractor:
    CITATION_RE = re.compile(
        r"""\b(?:
            SR\s*\d{3}(?:\.\d+)?(?:\s+Art\.?\s*\d+[a-z]?)?
          | BGE\s+\d{1,3}\s+[IVX]+[a-z]?\s+\d+(?:\s+E\.\s*\d+[a-z]?)?
          | Art\.?\s+\d+[a-z]?\s+(?:Abs\.?\s*\d+\s+)?[A-Z]{2,}
        )\b""",
        re.VERBOSE,
    )

    def __init__(self, context_sentences: int = 2):
        self.context_sentences = context_sentences

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
        for c in self.CITATION_RE.findall(cc_text):
            key = self._normalize(c)
            freq_map[key] = freq_map.get(key, 0) + 1

        gold_normalized = {self._normalize(g) for g in gold_citations}
        seen: set[str] = set()
        instances: list[CitationInstance] = []

        total_citations_in_cc = len(freq_map)   # CC内不同citation的种数
        citation_density = total_citations_in_cc / max(total_sents, 1)  # 每句平均citation数


        for sent_idx, sent in enumerate(sentences):
            for m in self.CITATION_RE.finditer(sent):
                cit_id = self._normalize(m.group(0))
                if cit_id in seen:
                    continue
                seen.add(cit_id)
                pre, post = self._context(sentences, sent_idx)

                ctx = pre + " " + sent + " " + post

                # ── citation role features ──────────────────────
                n_cit_in_ctx = len(self.CITATION_RE.findall(ctx))
                is_isolated  = int(n_cit_in_ctx == 1)

                # 检查 citation 是否出现在括号内
                in_paren = False
                for par_m in re.finditer(r'\(([^)]{0,200})\)', ctx):
                    if cit_id[:6].lower() in par_m.group(1).lower():
                        in_paren = True
                        break
                # ────────────────────────────────────────────────

                instances.append(CitationInstance(
                    citation_id=cit_id,
                    cc_id=cc_id,
                    query_id=query_id,
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
        return base + [f"tfidf_{w}" for w in sorted(self.vocab, key=self.vocab.get)]



    def _vec(self, inst: CitationInstance) -> list[float]:
        rel_pos = inst.sentence_index / max(inst.total_sentences - 1, 1)
        ctx = (inst.preceding_text + " " + inst.following_text).lower()
        pos_kw = sum(1 for kw in self.POSITIVE_KW if kw in ctx)
        neg_kw = sum(1 for kw in self.NEGATIVE_KW if kw in ctx)
        d, s, r = _safe_dense(inst.dense_score), _safe_sparse(inst.sparse_score), _safe_rerank(inst.rerank_score)
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
        ]
        return base + self._semantic_group_features(inst) + self._tfidf(inst.preceding_text + " " + inst.following_text)

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

# from transformers import AutoTokenizer, AutoModel
# import torch
# import torch.nn.functional as F

# class BertCitationFeatureBuilder(CitationFeatureBuilder):
#     """
#     用 BERT encoder 替换 TF-IDF。
#     其余手工特征（检索分数、位置、关键词等）保持不变。
#     继承父类只是为了复用 feature_names() 和 _vec() 的非TF-IDF部分。
#     """

#     def __init__(self, model_name: str = "deepset/gbert-base", 
#                  batch_size: int = 64, device: str = None):
#         super().__init__()
#         self.model_name = model_name
#         self.batch_size = batch_size
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.tokenizer = None
#         self.bert = None
#         self._emb_dim = None

#     def fit(self, instances: list[CitationInstance]):
#         # BERT 不需要fit，但要在这里初始化模型
#         # 沿用父类接口，保持调用方不变
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#         self.bert = AutoModel.from_pretrained(self.model_name).to(self.device)
#         self.bert.eval()

#         # 先encode，再fit PCA
#         texts = [
#             inst.preceding_text + " [SEP] " + inst.following_text
#             for inst in instances
#         ]
#         bert_feats = self._encode_texts(texts)        # ← 这行不能省
    
#         self.pca = PCA(n_components=64)
#         self.pca.fit(bert_feats)                      # ← fitOnly，不transform
    
#         # 确认维度
#         self._emb_dim = 64
#         return self

#     def transform(self, instances: list[CitationInstance]) -> np.ndarray:
#         # 手工特征（父类 _vec，不含TF-IDF部分）
#         hand_feats = np.array(
#             [self._vec_no_tfidf(inst) for inst in instances], 
#             dtype=np.float32
#         )
#         # BERT特征
#         texts = [
#             inst.preceding_text + " [SEP] " + inst.following_text 
#             for inst in instances
#         ]
#         bert_feats = self._encode_texts(texts)   # (N, emb_dim)

#         # 第一次调用时fit PCA，之后只transform
#         if not hasattr(self, "pca"):
#             from sklearn.decomposition import PCA
#             self.pca = PCA(n_components=128)
#             bert_feats = self.pca.fit_transform(bert_feats)
#         else:
#             bert_feats = self.pca.transform(bert_feats)

#         return np.concatenate([hand_feats, bert_feats], axis=1)

#     def feature_names(self) -> list[str]:
#         base = [
#             "dense_score", "sparse_score", "rerank_score",
#             "dense_x_rerank", "sparse_x_rerank",
#             "dense_plus_sparse", "rerank_minus_dense",
#             "pos_relative", "pos_in_first_quarter", "pos_in_last_quarter",
#             "freq_raw", "freq_log", "freq_normalized",
#             "ctx_pos_kw", "ctx_neg_kw", "ctx_kw_ratio",
#             "ctx_pre_len", "ctx_post_len",
#             "is_bge",
#         ]
#         bert_names = [f"bert_{i}" for i in range(64)]
#         return base + bert_names

#     # ── 内部方法 ────────────────────────────────────────────────

#     def _vec_no_tfidf(self, inst: CitationInstance) -> list[float]:
#         # 父类 _vec() 去掉末尾的 self._tfidf() 调用
#         # 直接复制父类逻辑，只删最后一行
#         import math, random
#         rel_pos = inst.sentence_index / max(inst.total_sentences - 1, 1)
#         ctx = (inst.preceding_text + " " + inst.following_text).lower()
#         pos_kw = sum(1 for kw in self.POSITIVE_KW if kw in ctx)
#         neg_kw = sum(1 for kw in self.NEGATIVE_KW if kw in ctx)
#         d, s, r = inst.dense_score, inst.sparse_score, inst.rerank_score
#         return [
#             d, s, r,
#             d * r, s * r, d + s, r - d,
#             rel_pos,
#             float(rel_pos < 0.25),
#             float(rel_pos > 0.75),
#             float(inst.frequency_in_doc),
#             math.log1p(inst.frequency_in_doc),
#             inst.frequency_in_doc / 5.0,
#             float(pos_kw), float(neg_kw),
#             pos_kw / (pos_kw + neg_kw + 1),
#             float(len(inst.preceding_text.split())),
#             float(len(inst.following_text.split())),
#             inst.is_bge,
#         ]

#     @torch.no_grad()
#     def _encode_texts(self, texts: list[str]) -> np.ndarray:
#         all_embs = []
#         for i in range(0, len(texts), self.batch_size):
#             batch = texts[i : i + self.batch_size]
#             enc = self.tokenizer(
#                 batch,
#                 padding=True,
#                 truncation=True,
#                 max_length=256,     # 上下文窗口，可调
#                 return_tensors="pt",
#             ).to(self.device)
#             out = self.bert(**enc)
#             # mean pooling（比[CLS]对短文本更稳定）
#             mask = enc["attention_mask"].unsqueeze(-1).float()
#             emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
#             emb  = F.normalize(emb, dim=-1)   # L2归一化
#             all_embs.append(emb.cpu().numpy())
#         return np.concatenate(all_embs, axis=0)
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
            lambdarank_truncation_level=max(self.eval_at),
            verbose=-1,
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
                all_instances.extend(self.extractor.extract(
                    cc_text=cc["text"],
                    cc_id=cc["cc_id"],
                    query_id=query_id,
                    gold_citations=gold_citations,
                    dense_score=cc.get("dense_score", 0.0),
                    sparse_score=cc.get("sparse_score", 0.0),
                    rerank_score=cc.get("rerank_score", 0.0),
                ))
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