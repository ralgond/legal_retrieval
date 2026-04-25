"""
LTR Pipeline for Swiss German Legal Retrieval
==============================================

排序粒度：(query, citation)
  - 训练时：每个 query 下所有 CC 中的所有 citation 混排，label=is_gold
  - 预测时：输出 query_id, cc_id, citation_id, ltr_score, rank

完整数据流：

  JSON (train/valid/test)
        │
        ▼  DataLoader.load_with_retrieval_scores()
        │  ├─ list[CitationInstance]  → CitationGoldClassifier.fit()
        │  └─ cc_df (dense/sparse/rerank, CC级)
        │
        ▼  LTRFeatureBuilder.build()
           每行 = 一个 (query, citation)
           特征 = citation自身特征 + 所属CC的检索分数
        │
        ▼  LTRModel.fit(train_feat, valid_feat)   ← LambdaMART，group=query，valid早停
        │
        ▼  LTRModel.predict(test_feat)
           输出：query_id, cc_id, citation_id, ltr_score, rank
           rank 在 query 内跨所有CC混排
"""

import json
import numpy as np
import pandas as pd
from typing import Optional

from citation_gold_classifier import (
    CitationGoldClassifier,
    CitationExtractor,
    CitationInstance,
    DataLoader,
)


# ─────────────────────────────────────────────
# 特征构建（citation 粒度）
# ─────────────────────────────────────────────

class LTRFeatureBuilder:
    """
    输出每行对应一个 (query_id, cc_id, citation_id)。

    特征分两类：
      A. citation 自身特征（来自 CitationGoldClassifier）
           - gold_prob         : 分类器预测的 gold 概率（核心信号）
           - pos_relative      : 在 CC 中的归一化位置
           - freq_log          : 文档内频率 log(1+freq)
      B. 所属 CC 的检索分数（来自召回阶段）
           - dense_score
           - sparse_score
           - rerank_score
           - score_rrf         : 三路 RRF 融合
    """

    FEATURE_COLS = [
        # A. citation 自身
        "gold_prob",
        "pos_relative",
        "freq_log",
        # B. 所属 CC 检索分数
        "dense_score",
        "sparse_score",
        "rerank_score",
        "score_rrf",
    ]

    def __init__(self, clf: CitationGoldClassifier):
        self.clf = clf
        self.extractor = CitationExtractor(context_sentences=2)

    def build(
        self,
        data: list[dict],
        cc_df: pd.DataFrame,
        has_labels: bool = True,
    ) -> pd.DataFrame:
        """
        返回 citation 粒度的特征 DataFrame。
        每行：query_id, cc_id, citation_id, [label,] + FEATURE_COLS
        """
        # (query_id, cc_id) → retrieval scores
        score_index: dict[tuple, dict] = {
            (row["query_id"], row["cc_id"]): {
                "dense_score":  float(row["dense_score"]),
                "sparse_score": float(row["sparse_score"]),
                "rerank_score": float(row["rerank_score"]),
            }
            for _, row in cc_df.iterrows()
        }

        rows = []
        for entry in data:
            qid  = entry["query_id"]
            gold = entry.get("gold_citations", [])
            gold_normalized = {self.extractor._normalize(g) for g in gold}

            for cc in entry["cc_list"]:
                ccid    = cc["cc_id"]
                cc_text = cc["text"]
                scores  = score_index.get((qid, ccid), {
                    "dense_score": 0.0, "sparse_score": 0.0, "rerank_score": 0.0,
                })

                instances: list[CitationInstance] = self.extractor.extract(
                    cc_text=cc_text,
                    cc_id=ccid,
                    query_id=qid,
                    gold_citations=gold,
                )
                if not instances:
                    continue

                probs = self.clf.predict_proba(instances)

                dense  = scores["dense_score"]
                sparse = scores["sparse_score"]
                rerank = scores["rerank_score"]
                rrf = (
                    1 / (60 + (1 - dense))
                    + 1 / (60 + (1 - sparse))
                    + 1 / (60 + (1 - rerank))
                )

                for inst, prob in zip(instances, probs):
                    row: dict = {
                        "query_id":    qid,
                        "cc_id":       ccid,
                        "citation_id": inst.citation_id,
                        # A. citation 特征
                        "gold_prob":    float(prob),
                        "pos_relative": inst.sentence_index / max(inst.total_sentences - 1, 1),
                        "freq_log":     float(np.log1p(inst.frequency_in_doc)),
                        # B. CC 检索分数
                        "dense_score":  dense,
                        "sparse_score": sparse,
                        "rerank_score": rerank,
                        "score_rrf":    rrf,
                    }
                    if has_labels:
                        row["label"] = int(inst.citation_id in gold_normalized)
                    rows.append(row)

        return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# LTR 模型（LambdaMART via LightGBM）
# ─────────────────────────────────────────────

class LTRModel:
    """
    LambdaMART 排序模型。
    group = query（每个 query 下有多少个 citation）。
    valid 早停，监控 ndcg@10。
    """

    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        early_stopping_rounds: int = 50,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self.best_iteration_: Optional[int] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: Optional[pd.DataFrame] = None,
    ) -> "LTRModel":
        import lightgbm as lgb

        X_train, y_train, g_train = self._unpack(train_df)
        dtrain = lgb.Dataset(
            X_train, label=y_train, group=g_train,
            feature_name=LTRFeatureBuilder.FEATURE_COLS,
        )

        callbacks   = [lgb.log_evaluation(period=20)]
        valid_sets  = [dtrain]
        valid_names = ["train"]

        if valid_df is not None:
            X_val, y_val, g_val = self._unpack(valid_df)
            dval = lgb.Dataset(
                X_val, label=y_val, group=g_val,
                feature_name=LTRFeatureBuilder.FEATURE_COLS,
                reference=dtrain,
            )
            valid_sets.append(dval)
            valid_names.append("valid")
            callbacks.append(lgb.early_stopping(
                stopping_rounds=self.early_stopping_rounds,
                verbose=True,
            ))

        params = {
            "objective":    "lambdarank",
            "metric":       "ndcg",
            "ndcg_eval_at": [5, 10],
            "learning_rate":  self.learning_rate,
            "num_leaves":     self.num_leaves,
            "min_data_in_leaf": 3,
            "verbose": -1,
        }

        self.model = lgb.train(
            params, dtrain,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        self.best_iteration_ = self.model.best_iteration
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        输出列：query_id, cc_id, citation_id, ltr_score, rank
        rank 在 query 内跨所有 CC 混排（rank=1 是该 query 最重要的 citation）
        """
        X = df[LTRFeatureBuilder.FEATURE_COLS].values
        scores = self.model.predict(X, num_iteration=self.best_iteration_)

        out = df[["query_id", "cc_id", "citation_id"]].copy()
        if "label" in df.columns:
            out["label"] = df["label"].values
        out["ltr_score"] = scores
        out["rank"] = (
            out.groupby("query_id")["ltr_score"]
            .rank(ascending=False, method="first")
            .astype(int)
        )
        return out.sort_values(["query_id", "rank"]).reset_index(drop=True)

    def feature_importance(self) -> pd.DataFrame:
        imp = self.model.feature_importance(importance_type="gain")
        return (
            pd.DataFrame({
                "feature":    LTRFeatureBuilder.FEATURE_COLS,
                "importance": imp,
            })
            .sort_values("importance", ascending=False)
        )

    def save(self, path: str):
        self.model.save_model(path)

    def load(self, path: str) -> "LTRModel":
        import lightgbm as lgb
        self.model = lgb.Booster(model_file=path)
        self.best_iteration_ = self.model.best_iteration
        return self

    def _unpack(self, df: pd.DataFrame):
        X = df[LTRFeatureBuilder.FEATURE_COLS].values.astype(np.float32)
        y = df["label"].values.astype(np.float32)
        # group：每个 query 下有多少个 citation
        g = df.groupby("query_id", sort=False).size().values
        return X, y, g


# ─────────────────────────────────────────────
# 评估（citation 粒度）
# ─────────────────────────────────────────────

class Evaluator:
    @staticmethod
    def recall_at_k(ranked_df: pd.DataFrame, k: int) -> float:
        scores = []
        for _, grp in ranked_df.groupby("query_id"):
            top_k = grp[grp["rank"] <= k]["label"].sum()
            total = grp["label"].sum()
            if total > 0:
                scores.append(min(top_k / total, 1.0))
        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def mrr(ranked_df: pd.DataFrame) -> float:
        scores = []
        for _, grp in ranked_df.groupby("query_id"):
            first_hit = grp[grp["label"] == 1]["rank"].min()
            scores.append(1.0 / first_hit if pd.notna(first_hit) else 0.0)
        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def ndcg_at_k(ranked_df: pd.DataFrame, k: int) -> float:
        import math
        def dcg(rels: list) -> float:
            return sum(r / math.log2(i + 2) for i, r in enumerate(rels))
        scores = []
        for _, grp in ranked_df.groupby("query_id"):
            top_k = grp[grp["rank"] <= k].sort_values("rank")["label"].tolist()
            ideal = sorted(grp["label"].tolist(), reverse=True)[:k]
            idcg  = dcg(ideal)
            if idcg > 0:
                scores.append(dcg(top_k) / idcg)
        return float(np.mean(scores)) if scores else 0.0

    @classmethod
    def report(cls, ranked_df: pd.DataFrame) -> dict:
        return {
            "Recall@5":  cls.recall_at_k(ranked_df, 5),
            "Recall@10": cls.recall_at_k(ranked_df, 10),
            "MRR":       cls.mrr(ranked_df),
            "NDCG@5":    cls.ndcg_at_k(ranked_df, 5),
            "NDCG@10":   cls.ndcg_at_k(ranked_df, 10),
        }


# ─────────────────────────────────────────────
# 训练 + 预测入口
# ─────────────────────────────────────────────

def train_and_evaluate(
    train_path: str,
    valid_path: str,
    citation_model_type: str = "lgbm",
) -> tuple[CitationGoldClassifier, LTRFeatureBuilder, LTRModel]:
    """
    Step 1: 训练 CitationGoldClassifier（citation 二分类，带早停）
    Step 2: 构建 citation 粒度 LTR 特征矩阵
    Step 3: 训练 LTRModel（LambdaMART，带早停）
    Step 4: 打印 valid 指标

    返回 (clf, feat_builder, ltr)，三者均需传给 predict_test()。
    """
    loader = DataLoader()

    with open(train_path, encoding="utf-8") as f:
        train_data = json.load(f)
    with open(valid_path, encoding="utf-8") as f:
        valid_data = json.load(f)

    train_instances, train_cc_df = loader.load_with_retrieval_scores(train_data)
    valid_instances, valid_cc_df = loader.load_with_retrieval_scores(valid_data)

    # Step 1
    print("=" * 60)
    print("Step 1: 训练 CitationGoldClassifier")
    print("=" * 60)
    clf = CitationGoldClassifier(model_type=citation_model_type, early_stopping_rounds=30)
    clf.fit(train_instances, valid_instances=valid_instances)

    # Step 2
    print("\n" + "=" * 60)
    print("Step 2: 构建 citation 粒度 LTR 特征矩阵")
    print("=" * 60)
    feat_builder = LTRFeatureBuilder(clf)
    train_feat = feat_builder.build(train_data, train_cc_df, has_labels=True)
    valid_feat = feat_builder.build(valid_data, valid_cc_df, has_labels=True)
    print(f"  train: {len(train_feat)} citations  (gold={int(train_feat['label'].sum())})")
    print(f"  valid: {len(valid_feat)} citations  (gold={int(valid_feat['label'].sum())})")

    # Step 3
    print("\n" + "=" * 60)
    print("Step 3: 训练 LTRModel (LambdaMART，citation 粒度)")
    print("=" * 60)
    ltr = LTRModel(early_stopping_rounds=50)
    ltr.fit(train_feat, valid_feat)
    print(f"\n  最佳迭代轮次: {ltr.best_iteration_}")

    # Step 4
    print("\n" + "=" * 60)
    print("Step 4: Valid 集评估（citation 粒度，跨 CC 混排）")
    print("=" * 60)
    ranked = ltr.predict(valid_feat)
    for k, v in Evaluator.report(ranked).items():
        print(f"  {k:<12}: {v:.4f}")

    return clf, feat_builder, ltr


def predict_test(
    test_path: str,
    clf: CitationGoldClassifier,
    feat_builder: LTRFeatureBuilder,
    ltr: LTRModel,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    对 test 集生成 citation 级排序结果。

    参数：
      test_path    : test JSON（格式同 train，但无 gold_citations 字段）
      clf          : train_and_evaluate() 返回的 CitationGoldClassifier
      feat_builder : train_and_evaluate() 返回的 LTRFeatureBuilder（保证特征对齐）
      ltr          : train_and_evaluate() 返回的 LTRModel
      output_path  : 可选，写入 JSON

    输出列：query_id, cc_id, citation_id, ltr_score, rank
      rank 在 query 内跨所有 CC 混排（rank=1 是该 query 最重要的 citation）
    """
    loader = DataLoader()

    with open(test_path, encoding="utf-8") as f:
        test_data = json.load(f)

    _, test_cc_df = loader.load_with_retrieval_scores(test_data)
    test_feat = feat_builder.build(test_data, test_cc_df, has_labels=False)

    ranked = ltr.predict(test_feat)

    if output_path:
        ranked.to_json(output_path, orient="records", force_ascii=False, indent=2)
        print(f"预测结果已写入 {output_path}  ({len(ranked)} 条 citation)")

    return ranked