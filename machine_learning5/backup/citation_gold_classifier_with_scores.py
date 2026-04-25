"""
Citation Gold Classifier
========================
目标：对CC中每个citation，预测它是否为gold citation（二分类）

输入信息（你实际拥有的）：
  1. citation前后的上下文文本
  2. citation在文档中被引用的次数（文档级频率）
  3. citation在court consideration中的第几句

数据格式（train/valid JSON）：
[
  {
    "query_id": "train_0001",
    "gold_citations": ["BGE 138 III 123", ...],
    "cc_list": [
      {
        "dense_score": 0.1,
        "sparse_score": 0.1,
        "rerank_score": 0.2,
        "text": "..."
      }
    ]
  }
]

架构选择：
  - 'lr'  : LogisticRegression（无早停，用valid做事后评估）
  - 'lgbm': LightGBM（原生早停，监控valid F1）
"""

import json
import re
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class CitationInstance:
    """一个citation实例，包含你实际能拿到的三类信息"""
    citation_id: str
    cc_id: str
    query_id: str

    # ── 你拥有的三类信息 ──────────────────────
    preceding_text: str       # citation前的文本
    following_text: str       # citation后的文本
    sentence_index: int       # 在CC中第几句（0-based）
    total_sentences: int      # CC总句数
    frequency_in_doc: int     # 该citation在本CC中出现次数

    # 🔥 新增：继承自 CC 的检索分数
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: float = 0.0

    # ── label ─────────────────────────────────
    is_gold: int = 0          # 1 = gold citation, 0 = non-gold


# ─────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────

class DataLoader:
    """
    将你的JSON格式数据加载为CitationInstance列表。

    JSON格式：
    [
      {
        "query_id": "train_0001",
        "gold_citations": ["BGE 138 III 123", ...],
        "cc_list": [
          {"cc_id": "cc_001", "dense_score": 0.1, "sparse_score": 0.1, "rerank_score": 0.2, "text": "..."}
        ]
      }
    ]
    """

    def __init__(self, context_sentences: int = 2):
        self.extractor = CitationExtractor(context_sentences=context_sentences)

    def load_file(self, path: str) -> list[CitationInstance]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self.load(data)

    def load(self, data: list[dict]) -> list[CitationInstance]:
        """
        从已解析的list[dict]加载，cc_id直接读取自 cc["cc_id"]
        """
        all_instances: list[CitationInstance] = []

        for entry in data:
            query_id: str = entry["query_id"]
            gold_citations: list[str] = entry["gold_citations"]
            cc_list: list[dict] = entry["cc_list"]

            for cc in cc_list:
                cc_id = cc["cc_id"]
                instances = self.extractor.extract(
                    cc_text=cc["text"],
                    cc_id=cc_id,
                    query_id=query_id,
                    gold_citations=gold_citations,

                    # 🔥 新增
                    dense_score=cc.get("dense_score", 0.0),
                    sparse_score=cc.get("sparse_score", 0.0),
                    rerank_score=cc.get("rerank_score", 0.0),
                )
                all_instances.extend(instances)

        return all_instances

    def load_with_retrieval_scores(self, data: list[dict]) -> tuple[list[CitationInstance], pd.DataFrame]:
        """
        同时返回citation实例 和 CC级别的检索分数DataFrame（供LTR使用）。

        返回：
          instances  : list[CitationInstance]
          cc_df      : DataFrame，列= [query_id, cc_id, dense_score, sparse_score, rerank_score]
        """
        all_instances: list[CitationInstance] = []
        cc_rows: list[dict] = []

        for entry in data:
            query_id = entry["query_id"]
            gold_citations = entry["gold_citations"]

            for cc in entry["cc_list"]:
                cc_id = cc["cc_id"]

                instances = self.extractor.extract(
                    cc_text=cc["text"],
                    cc_id=cc_id,
                    query_id=query_id,
                    gold_citations=gold_citations,

                    # 🔥 新增
                    dense_score=cc.get("dense_score", 0.0),
                    sparse_score=cc.get("sparse_score", 0.0),
                    rerank_score=cc.get("rerank_score", 0.0),
                )
                all_instances.extend(instances)

                cc_rows.append({
                    "query_id":     query_id,
                    "cc_id":        cc_id,
                    "dense_score":  cc.get("dense_score", 0.0),
                    "sparse_score": cc.get("sparse_score", 0.0),
                    "rerank_score": cc.get("rerank_score", 0.0),
                    # CC级别label：是否包含任何gold citation
                    "label": int(any(
                        self.extractor._normalize(g) in {
                            self.extractor._normalize(m.group(0))
                            for m in CitationExtractor.CITATION_RE.finditer(cc["text"])
                        }
                        for g in gold_citations
                    )),
                })

        return all_instances, pd.DataFrame(cc_rows)


# ─────────────────────────────────────────────
# 从CC文本中抽取CitationInstance
# ─────────────────────────────────────────────

class CitationExtractor:
    # 德语瑞士法律citation正则
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

        # 🔥 必须加
        dense_score: float,
        sparse_score: float,
        rerank_score: float,
    ) -> list[CitationInstance]:
        sentences = self._split_sentences(cc_text)
        total_sents = len(sentences)

        all_cit_raw = self.CITATION_RE.findall(cc_text)
        freq_map: dict[str, int] = {}
        for c in all_cit_raw:
            key = self._normalize(c)
            freq_map[key] = freq_map.get(key, 0) + 1

        gold_normalized = {self._normalize(g) for g in gold_citations}
        seen_cit_ids: set[str] = set()
        instances: list[CitationInstance] = []

        for sent_idx, sent in enumerate(sentences):
            for m in self.CITATION_RE.finditer(sent):
                cit_id = self._normalize(m.group(0))
                if cit_id in seen_cit_ids:
                    continue
                seen_cit_ids.add(cit_id)

                pre, post = self._get_context(sentences, sent_idx)
                instances.append(CitationInstance(
                    citation_id=cit_id,
                    cc_id=cc_id,
                    query_id=query_id,
                    preceding_text=pre,
                    following_text=post,
                    sentence_index=sent_idx,
                    total_sentences=total_sents,
                    frequency_in_doc=freq_map.get(cit_id, 1),

                    # 🔥 关键：继承 CC 分数
                    dense_score=dense_score,
                    sparse_score=sparse_score,
                    rerank_score=rerank_score,

                    is_gold=int(cit_id in gold_normalized),
                ))

        return instances

    def _get_context(self, sentences: list[str], idx: int) -> tuple[str, str]:
        n = self.context_sentences
        pre = " ".join(sentences[max(0, idx - n): idx])
        post = " ".join(sentences[idx + 1: idx + 1 + n])
        return pre, post

    def _split_sentences(self, text: str) -> list[str]:
        text = re.sub(r"\b(E|Nr|Abs|Art|Ziff|bzw|ca|vgl|usw|etc)\.", r"\1<DOT>", text)
        sents = re.split(r"(?<=[.!?])\s+(?=[A-ZÜÄÖ\d])", text)
        return [s.replace("<DOT>", ".").strip() for s in sents if s.strip()]

    def _normalize(self, raw: str) -> str:
        return re.sub(r"\s+", " ", raw.strip().upper())


# ─────────────────────────────────────────────
# 特征工程
# ─────────────────────────────────────────────

class CitationFeatureBuilder:
    POSITIVE_KEYWORDS = [
        "grundlegend", "massgebend", "wegweisend", "ständige rechtsprechung",
        "bestätigt", "gemäss", "analog", "entsprechend", "gilt", "ist anzuwenden",
        "ergibt sich", "findet anwendung", "in rechtlicher hinsicht",
        "ratio decidendi", "leitentscheid",
    ]
    NEGATIVE_KEYWORDS = [
        "vgl.", "anders als", "entgegen", "zweifelhaft", "offen gelassen",
        "nicht anwendbar", "abweichend", "kritisch",
    ]

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def fit(self, instances: list[CitationInstance]):
        corpus = [self._context_text(inst) for inst in instances]
        self._build_vocab(corpus, max_features=500)
        return self

    def transform(self, instances: list[CitationInstance]) -> np.ndarray:
        rows = [self._build_feature_vector(inst) for inst in instances]
        return np.array(rows, dtype=np.float32)

    def feature_names(self) -> list[str]:
        return [
            # 🔥 CC-level features
            "dense_score",
            "sparse_score",
            "rerank_score",
            "dense_x_rerank",
            "sparse_x_rerank",
            "dense_plus_sparse",
            "rerank_minus_dense",
            "pos_relative", "pos_in_first_quarter", "pos_in_last_quarter",
            "freq_raw", "freq_log", "freq_normalized",
            "ctx_positive_kw_count", "ctx_negative_kw_count", "ctx_kw_ratio",
            "ctx_pre_len", "ctx_post_len",
        ] + [f"tfidf_{w}" for w in sorted(self.vocab, key=self.vocab.get)]

    def _build_feature_vector(self, inst: CitationInstance) -> list[float]:
        rel_pos = inst.sentence_index / max(inst.total_sentences - 1, 1)
        ctx = (inst.preceding_text + " " + inst.following_text).lower()
        pos_count = sum(1 for kw in self.POSITIVE_KEYWORDS if kw in ctx)
        neg_count = sum(1 for kw in self.NEGATIVE_KEYWORDS if kw in ctx)


        dense = inst.dense_score
        sparse = inst.sparse_score
        rerank = inst.rerank_score

        score_features = [
            dense,
            sparse,
            rerank,
            dense * rerank,
            sparse * rerank,
            dense + sparse,
            rerank - dense,
        ]

        base = score_features + [
            rel_pos,
            float(rel_pos < 0.25),
            float(rel_pos > 0.75),
            float(inst.frequency_in_doc),
            math.log1p(inst.frequency_in_doc),
            inst.frequency_in_doc / 5.0,
            float(pos_count),
            float(neg_count),
            pos_count / (pos_count + neg_count + 1),
            float(len(inst.preceding_text.split())),
            float(len(inst.following_text.split())),
        ]
        return base + self._tfidf_vector(self._context_text(inst))

    def _context_text(self, inst: CitationInstance) -> str:
        return inst.preceding_text + " " + inst.following_text

    def _build_vocab(self, corpus: list[str], max_features: int):
        from collections import Counter
        token_counts: Counter = Counter()
        doc_freq: Counter = Counter()
        for doc in corpus:
            tokens = set(self._tokenize(doc))
            doc_freq.update(tokens)
            token_counts.update(self._tokenize(doc))
        N = len(corpus)
        scored = {
            w: token_counts[w] * math.log(N / (doc_freq[w] + 1))
            for w in token_counts
            if len(w) > 2 and doc_freq[w] > 1
        }
        top = sorted(scored, key=scored.get, reverse=True)[:max_features]
        self.vocab = {w: i for i, w in enumerate(top)}

    def _tfidf_vector(self, text: str) -> list[float]:
        if not self.vocab:
            return []
        from collections import Counter
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        vec = [0.0] * len(self.vocab)
        for token, count in tf.items():
            if token in self.vocab:
                vec[self.vocab[token]] = count / max(len(tokens), 1)
        return vec

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b[a-züäöA-ZÜÄÖ]{3,}\b", text.lower())


# ─────────────────────────────────────────────
# 分类器（含早停）
# ─────────────────────────────────────────────

class CitationGoldClassifier:
    """
    二分类器：预测citation是否为gold

    早停策略：
      - 'lr'  : 无原生早停；fit()接受valid_instances做事后评估，
                不影响训练（LR无需早停，正则C已起到同等作用）
      - 'lgbm': 原生早停，监控valid集的 binary_logloss，
                patience由 early_stopping_rounds 控制
    """

    def __init__(
        self,
        model_type: str = "lgbm",
        # LightGBM超参
        n_estimators: int = 1000,      # 树的上限（早停会自动截断）
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        early_stopping_rounds: int = 50,  # valid无改善多少轮后停止
        # LogisticRegression超参
        C: float = 1.0,
    ):
        assert model_type in ("lr", "lgbm")
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.early_stopping_rounds = early_stopping_rounds
        self.C = C

        self.model = None
        self.feature_builder = CitationFeatureBuilder()
        self.best_iteration_: Optional[int] = None   # lgbm早停后的最佳轮次
        self.valid_metrics_: Optional[dict] = None   # 早停时的valid指标

    def fit(
        self,
        train_instances: list[CitationInstance],
        valid_instances: Optional[list[CitationInstance]] = None,
    ) -> "CitationGoldClassifier":
        """
        train_instances : 训练集citation列表
        valid_instances : 验证集citation列表（lgbm用于早停，lr用于评估打印）
        """
        # 在训练集上fit特征词表
        self.feature_builder.fit(train_instances)

        X_train = self.feature_builder.transform(train_instances)
        y_train = np.array([inst.is_gold for inst in train_instances])

        if self.model_type == "lr":
            self._fit_lr(X_train, y_train, valid_instances)
        else:
            self._fit_lgbm(X_train, y_train, valid_instances)

        return self

    # ── LR ────────────────────────────────────────────────────────────────

    def _fit_lr(self, X_train, y_train, valid_instances):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=self.C,
                class_weight="balanced",
                max_iter=500,
                solver="lbfgs",
            )),
        ])
        self.model.fit(X_train, y_train)

        # LR没有早停，但打印valid集指标供参考
        if valid_instances:
            metrics = self.evaluate(valid_instances)
            self.valid_metrics_ = metrics
            print(f"[LR] Valid → "
                  f"F1={metrics['f1']:.4f}  "
                  f"AUC={metrics['roc_auc']:.4f}  "
                  f"AP={metrics['avg_precision']:.4f}")

    # ── LightGBM ──────────────────────────────────────────────────────────

    def _fit_lgbm(self, X_train, y_train, valid_instances):
        import lightgbm as lgb
        from sklearn.metrics import f1_score

        # 构造valid集
        eval_set = []
        if valid_instances:
            X_val = self.feature_builder.transform(valid_instances)
            y_val = np.array([inst.is_gold for inst in valid_instances])
            eval_set = [(X_val, y_val)]

        # 自定义early stopping metric：valid F1（比logloss更直接）
        # LightGBM callbacks方式（>=3.0 API）
        callbacks = [lgb.log_evaluation(period=20)]
        if valid_instances:
            callbacks.append(lgb.early_stopping(
                stopping_rounds=self.early_stopping_rounds,
                verbose=True,
            ))

        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            class_weight="balanced",
            verbose=-1,
        )

        import pandas as _pd
        feat_names = self.feature_builder.feature_names()
        X_train_df = _pd.DataFrame(X_train, columns=feat_names)

        fit_kwargs = dict(X=X_train_df, y=y_train, callbacks=callbacks)
        if eval_set:
            X_val_df = _pd.DataFrame(eval_set[0][0], columns=feat_names)
            fit_kwargs["eval_set"] = [(X_val_df, eval_set[0][1])]
            fit_kwargs["eval_metric"] = "binary_logloss"

        self.model.fit(**fit_kwargs)

        self.best_iteration_ = getattr(self.model, "best_iteration_", None)

        if valid_instances:
            self.valid_metrics_ = self.evaluate(valid_instances)
            print(
                f"\n[LGBM] 早停于第 {self.best_iteration_} 轮  |  "
                f"Valid F1={self.valid_metrics_['f1']:.4f}  "
                f"AUC={self.valid_metrics_['roc_auc']:.4f}  "
                f"AP={self.valid_metrics_['avg_precision']:.4f}"
            )

    # ── 推理 ──────────────────────────────────────────────────────────────
    def predict_proba(self, instances: list[CitationInstance]) -> np.ndarray:
        # 1. 转换成 numpy 数组
        X = self.feature_builder.transform(instances)
        
        # 2. 获取训练时使用的特征名称
        feat_names = self.feature_builder.feature_names()
        
        # 3. 将 X 包装成 DataFrame (解决 UserWarning)
        import pandas as _pd
        X_df = _pd.DataFrame(X, columns=feat_names)
        
        return self.model.predict_proba(X_df)[:, 1]

    # def predict_proba(self, instances: list[CitationInstance]) -> np.ndarray:
    #     X = self.feature_builder.transform(instances)
    #     return self.model.predict_proba(X)[:, 1]

    def predict(self, instances: list[CitationInstance], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(instances) >= threshold).astype(int)

    # ── 评估 ──────────────────────────────────────────────────────────────

    def evaluate(self, instances: list[CitationInstance]) -> dict:
        from sklearn.metrics import (
            precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score,
        )
        y_true = np.array([inst.is_gold for inst in instances])
        y_prob = self.predict_proba(instances)
        y_pred = (y_prob >= 0.5).astype(int)

        return {
            "precision":     precision_score(y_true, y_pred, zero_division=0),
            "recall":        recall_score(y_true, y_pred, zero_division=0),
            "f1":            f1_score(y_true, y_pred, zero_division=0),
            "roc_auc":       roc_auc_score(y_true, y_prob) if y_true.sum() > 0 else 0.0,
            "avg_precision": average_precision_score(y_true, y_prob) if y_true.sum() > 0 else 0.0,
            "support_pos":   int(y_true.sum()),
            "support_neg":   int((1 - y_true).sum()),
        }

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        names = self.feature_builder.feature_names()
        if self.model_type == "lr":
            coefs = np.abs(self.model.named_steps["clf"].coef_[0])
        else:
            coefs = self.model.feature_importances_
            names = names[:len(coefs)]
        df = pd.DataFrame({"feature": names[:len(coefs)], "importance": coefs})
        return df.sort_values("importance", ascending=False).head(top_n)


# ─────────────────────────────────────────────
# 与LTR的整合接口
# ─────────────────────────────────────────────

class CitationScoredCC:
    def __init__(self, classifier: CitationGoldClassifier, extractor: CitationExtractor):
        self.clf = classifier
        self.ext = extractor

    def score_for_ltr(
        self,
        cc_text: str,
        cc_id: str,
        query_id: str,
        gold_citations: list[str],
    ) -> dict:
        instances = self.ext.extract(cc_text, cc_id, query_id, gold_citations)
        if not instances:
            return {
                "citation_max_gold_prob": 0.0,
                "citation_mean_gold_prob": 0.0,
                "citation_predicted_gold_count": 0,
                "citation_count": 0,
            }

        probs = self.clf.predict_proba(instances)
        preds = (probs >= 0.5).astype(int)

        return {
            "citation_max_gold_prob":        float(probs.max()),
            "citation_mean_gold_prob":       float(probs.mean()),
            "citation_predicted_gold_count": int(preds.sum()),
            "citation_count":                len(instances),
            "_citation_details": [
                {
                    "citation_id":    inst.citation_id,
                    "gold_prob":      float(p),
                    "predicted_gold": int(pred),
                    "is_gold":        inst.is_gold,
                }
                for inst, p, pred in zip(instances, probs, preds)
            ],
        }