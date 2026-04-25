"""
Citation Gold Classifier
========================
目标：对CC中每个citation，预测它是否为gold citation（二分类）

输入信息（你实际拥有的）：
  1. citation前后的上下文文本
  2. citation在文档中被引用的次数（文档级频率）
  3. citation在court consideration中的第几句

数据格式（train/valid JSONL，每行一个JSON对象）：
{"query_id": "train_0001", "gold_citations": ["BGE 138 III 123", ...], "cc_list": [...]}

test JSONL 格式相同，只是缺少 gold_citations 字段：
{"query_id": "test_0001", "cc_list": [...]}

架构选择：
  - 'lr'  : LogisticRegression（无早停，用valid做事后评估）
  - 'lgbm': LightGBM（原生早停，监控valid binary_logloss）
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

    # CC级别的检索分数（继承自所属CC）
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
    将 JSONL 格式数据加载为 CitationInstance 列表。

    train/valid JSONL（每行一个对象，含 gold_citations）：
      {"query_id": "train_0001", "gold_citations": ["BGE 138 III 123"], "cc_list": [
        {"cc_id": "cc_001", "dense_score": 0.1, "sparse_score": 0.1, "rerank_score": 0.2, "text": "..."}
      ]}

    test JSONL（每行一个对象，无 gold_citations）：
      {"query_id": "test_0001", "cc_list": [
        {"cc_id": "cc_001", "dense_score": 0.1, "sparse_score": 0.1, "rerank_score": 0.2, "text": "..."}
      ]}
    """

    def __init__(self, context_sentences: int = 2):
        self.extractor = CitationExtractor(context_sentences=context_sentences)

    # ── 读取工具 ──────────────────────────────────────────────────────────

    @staticmethod
    def _read_jsonl(path: str) -> list[dict]:
        """逐行读取 JSONL 文件，跳过空行"""
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

    # ── train / valid 加载 ────────────────────────────────────────────────

    def load_file(self, path: str) -> list[CitationInstance]:
        """从 train/valid JSONL 文件加载（含 gold_citations）"""
        return self.load(self._read_jsonl(path))

    def load(self, data: list[dict]) -> list[CitationInstance]:
        """
        从已解析的 list[dict] 加载（train/valid，含 gold_citations）。
        cc_id 直接读取自 cc["cc_id"]。
        """
        all_instances: list[CitationInstance] = []

        for entry in data:
            query_id: str = entry["query_id"]
            gold_citations: list[str] = entry["gold_citations"]   # train/valid 必须有

            for cc in entry["cc_list"]:
                instances = self.extractor.extract(
                    cc_text=cc["text"],
                    cc_id=cc["cc_id"],
                    query_id=query_id,
                    gold_citations=gold_citations,
                    dense_score=cc.get("dense_score", 0.0),
                    sparse_score=cc.get("sparse_score", 0.0),
                    rerank_score=cc.get("rerank_score", 0.0),
                )
                all_instances.extend(instances)

        return all_instances

    def load_with_retrieval_scores(self, data: list[dict]) -> tuple[list[CitationInstance], pd.DataFrame]:
        """
        同时返回 citation 实例 和 CC 级别的检索分数 DataFrame（供 LTR 使用）。

        返回：
          instances : list[CitationInstance]
          cc_df     : DataFrame，列 = [query_id, cc_id, dense_score, sparse_score, rerank_score, label]
        """
        all_instances: list[CitationInstance] = []
        cc_rows: list[dict] = []

        for entry in data:
            query_id: str = entry["query_id"]
            gold_citations: list[str] = entry["gold_citations"]

            for cc in entry["cc_list"]:
                cc_id = cc["cc_id"]
                instances = self.extractor.extract(
                    cc_text=cc["text"],
                    cc_id=cc_id,
                    query_id=query_id,
                    gold_citations=gold_citations,
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
                    "label": int(any(
                        self.extractor._normalize(g) in {
                            self.extractor._normalize(m.group(0))
                            for m in CitationExtractor.CITATION_RE.finditer(cc["text"])
                        }
                        for g in gold_citations
                    )),
                })

        return all_instances, pd.DataFrame(cc_rows)

    # ── test 预测 ─────────────────────────────────────────────────────────

    def predict_file(
        self,
        path: str,
        classifier: "CitationGoldClassifier",
        threshold: float = 0.5,
        output_path: Optional[str] = None,
    ) -> list[dict]:
        """
        从 test JSONL 文件加载并预测，可选写出结果 JSONL。

        Args:
            path        : test JSONL 文件路径
            classifier  : 已训练的 CitationGoldClassifier
            threshold   : gold 判定阈值，默认 0.5
            output_path : 若指定，将预测结果写入该 JSONL 文件（每行一个 query）

        Returns:
            list[dict]，格式见 predict_dataset()
        """
        data = self._read_jsonl(path)
        results = self.predict_dataset(data, classifier, threshold)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                for record in results:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return results

    def predict_dataset(
        self,
        data: list[dict],
        classifier: "CitationGoldClassifier",
        threshold: float = 0.5,
    ) -> list[dict]:
        """
        对 test 数据做预测（无 gold_citations 字段）。

        输入格式（JSONL 每行 / list[dict] 每项）：
          {"query_id": "test_0001", "cc_list": [
            {"cc_id": "cc_001", "dense_score": 0.1, "sparse_score": 0.1,
             "rerank_score": 0.2, "text": "..."}
          ]}

        输出格式（list[dict]，与输入 query_id / cc_id 对齐）：
          [
            {
              "query_id": "test_0001",
              "cc_list": [
                {
                  "cc_id": "cc_001",
                  "citations": [
                    {"citation_id": "BGE 138 III 123", "gold_prob": 0.87, "predicted_gold": true},
                    {"citation_id": "BGER 4A_100/2020", "gold_prob": 0.21, "predicted_gold": false}
                  ]
                }
              ]
            }
          ]
        """
        results = []

        for entry in data:
            query_id: str = entry["query_id"]
            # test 集没有 gold_citations，传空列表（is_gold 全为 0，不影响特征）
            cc_results = []

            for cc in entry["cc_list"]:
                cc_id: str = cc["cc_id"]

                instances = self.extractor.extract(
                    cc_text=cc["text"],
                    cc_id=cc_id,
                    query_id=query_id,
                    gold_citations=[],
                    dense_score=cc.get("dense_score", 0.0),
                    sparse_score=cc.get("sparse_score", 0.0),
                    rerank_score=cc.get("rerank_score", 0.0),
                )

                if not instances:
                    cc_results.append({"cc_id": cc_id, "citations": []})
                    continue

                probs = classifier.predict_proba(instances)
                preds = (probs >= threshold).astype(int)

                cc_results.append({
                    "cc_id": cc_id,
                    "citations": [
                        {
                            "citation_id":    inst.citation_id,
                            "gold_prob":      round(float(prob), 4),
                            "predicted_gold": bool(pred),
                        }
                        for inst, prob, pred in zip(instances, probs, preds)
                    ],
                })

            results.append({"query_id": query_id, "cc_list": cc_results})

        return results


# ─────────────────────────────────────────────
# 从CC文本中抽取CitationInstance
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
        # return re.sub(r"\s+", " ", raw.strip().upper())
        return re.sub(r"\s+", " ", raw.strip())


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
            "dense_score", "sparse_score", "rerank_score",
            "dense_x_rerank", "sparse_x_rerank",
            "dense_plus_sparse", "rerank_minus_dense",
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

        dense, sparse, rerank = inst.dense_score, inst.sparse_score, inst.rerank_score

        base = [
            dense, sparse, rerank,
            dense * rerank, sparse * rerank,
            dense + sparse, rerank - dense,
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
    二分类器：预测 citation 是否为 gold

    早停策略：
      - 'lr'  : 无原生早停；fit() 接受 valid_instances 做事后评估
      - 'lgbm': 原生早停，监控 valid 集的 binary_logloss
    """

    def __init__(
        self,
        model_type: str = "lgbm",
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        early_stopping_rounds: int = 50,
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
        self.best_iteration_: Optional[int] = None
        self.valid_metrics_: Optional[dict] = None

    def fit(
        self,
        train_instances: list[CitationInstance],
        valid_instances: Optional[list[CitationInstance]] = None,
    ) -> "CitationGoldClassifier":
        self.feature_builder.fit(train_instances)
        X_train = self.feature_builder.transform(train_instances)
        y_train = np.array([inst.is_gold for inst in train_instances])

        if self.model_type == "lr":
            self._fit_lr(X_train, y_train, valid_instances)
        else:
            self._fit_lgbm(X_train, y_train, valid_instances)
        return self

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

        if valid_instances:
            metrics = self.evaluate(valid_instances)
            self.valid_metrics_ = metrics
            print(f"[LR] Valid → F1={metrics['f1']:.4f}  "
                  f"AUC={metrics['roc_auc']:.4f}  AP={metrics['avg_precision']:.4f}")

    def _fit_lgbm(self, X_train, y_train, valid_instances):
        import lightgbm as lgb

        eval_set = []
        if valid_instances:
            X_val = self.feature_builder.transform(valid_instances)
            y_val = np.array([inst.is_gold for inst in valid_instances])
            eval_set = [(X_val, y_val)]

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

        feat_names = self.feature_builder.feature_names()
        X_train_df = pd.DataFrame(X_train, columns=feat_names)
        fit_kwargs = dict(X=X_train_df, y=y_train, callbacks=callbacks)
        if eval_set:
            X_val_df = pd.DataFrame(eval_set[0][0], columns=feat_names)
            fit_kwargs["eval_set"] = [(X_val_df, eval_set[0][1])]
            fit_kwargs["eval_metric"] = "binary_logloss"

        self.model.fit(**fit_kwargs)
        self.best_iteration_ = getattr(self.model, "best_iteration_", None)

        if valid_instances:
            self.valid_metrics_ = self.evaluate(valid_instances)
            print(f"\n[LGBM] 早停于第 {self.best_iteration_} 轮  |  "
                  f"Valid F1={self.valid_metrics_['f1']:.4f}  "
                  f"AUC={self.valid_metrics_['roc_auc']:.4f}  "
                  f"AP={self.valid_metrics_['avg_precision']:.4f}")

    def predict_proba(self, instances: list[CitationInstance]) -> np.ndarray:
        X = self.feature_builder.transform(instances)
        feat_names = self.feature_builder.feature_names()
        X_df = pd.DataFrame(X, columns=feat_names)
        return self.model.predict_proba(X_df)[:, 1]

    def predict(self, instances: list[CitationInstance], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(instances) >= threshold).astype(int)

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
        dense_score: float = 0.0,
        sparse_score: float = 0.0,
        rerank_score: float = 0.0,
    ) -> dict:
        instances = self.ext.extract(
            cc_text, cc_id, query_id, gold_citations,
            dense_score=dense_score,
            sparse_score=sparse_score,
            rerank_score=rerank_score,
        )
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