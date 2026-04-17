"""
Citation Ranking — LightGBM Training & Prediction
===================================================
依赖上一个文件 citation_scorer.py 提供特征提取。

用法：
    python citation_lgbm.py --mode train  --data data.json --model_out model.pkl
    python citation_lgbm.py --mode predict --data test.json --model_in  model.pkl --out predictions.json
"""

from __future__ import annotations

import json
import math
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit
import lightgbm as lgb

# 从上一个文件导入 scorer
from citation_scorer import (
    CCResult,
    score_citations_for_query,
    evaluate as rule_evaluate,
)


# ─────────────────────────────────────────────
# 1. 特征列定义
# ─────────────────────────────────────────────

FEATURE_COLS = [
    # 关键词信号
    "kw_pos_score",
    "kw_neg_score",
    "kw_net_score",
    "kw_pos_count",
    "kw_neg_count",
    "window_score",
    # 位置
    "position_score",
    # citation 类型
    "type_bonus",
    "is_bge",
    "is_art",
    "is_sr",
    "is_botschaft",
    # cc 级别聚合
    "best_cc_level_score",
    "mean_cc_level_score",
    "max_cc_level_score",
    # rerank 信号
    "best_rerank_score",
    "mean_rerank_score",
    "top1_rerank_score",     # 出现在 rank=1 的 cc 中时的 rerank score，否则 0
    "top3_rerank_score",     # 出现在前 3 的 cc 中时最高 rerank score
    # 频次
    "appearances",
    "appearances_log",
    "appearances_in_top3",
    "appearances_in_top5",
    # 联合特征
    "best_cc_x_rerank",      # best_cc_level_score × best_rerank_score
    "freq_x_rerank",         # appearances × best_rerank_score
]


# ─────────────────────────────────────────────
# 2. 特征提取
# ─────────────────────────────────────────────

def build_feature_row(cit_result: dict, all_cc_scores: list[dict]) -> dict:
    """
    从 score_citations_for_query 的单条输出中提取 LightGBM 特征行。
    all_cc_scores: cit_result["all_cc_scores"]
    """
    citation = cit_result["citation"]

    cc_scores   = [x["cc_level_score"]  for x in all_cc_scores]
    rerank_scores = [x["rerank_score"]  for x in all_cc_scores]
    cc_ranks    = [x["cc_rank"]         for x in all_cc_scores]

    appearances      = len(cc_scores)
    appear_top3      = sum(1 for r in cc_ranks if r <= 3)
    appear_top5      = sum(1 for r in cc_ranks if r <= 5)

    best_rerank = max(rerank_scores)
    top1_rerank = max((r for x, r, rank in zip(cc_scores, rerank_scores, cc_ranks) if rank == 1), default=0.0)
    top3_rerank = max((r for r, rank in zip(rerank_scores, cc_ranks) if rank <= 3), default=0.0)

    best_cc  = cit_result["best_cc_level_score"]
    mean_cc  = sum(cc_scores) / len(cc_scores)
    max_cc   = max(cc_scores)

    return {
        # 关键词
        "kw_pos_score":        cit_result["kw_pos_score"],
        "kw_neg_score":        cit_result["kw_neg_score"],
        "kw_net_score":        cit_result["kw_pos_score"] - cit_result["kw_neg_score"],
        "kw_pos_count":        sum(1 for x in all_cc_scores),   # proxy: appearances
        "kw_neg_count":        cit_result["kw_neg_score"] > 0,
        "window_score":        cit_result["window_score"],
        # 位置
        "position_score":      cit_result["position_score"],
        # citation 类型
        "type_bonus":          cit_result["type_bonus"],
        "is_bge":              int("BGE" in citation),
        "is_art":              int("Art." in citation),
        "is_sr":               int("SR"  in citation),
        "is_botschaft":        int("Botschaft" in citation),
        # cc 级别
        "best_cc_level_score": best_cc,
        "mean_cc_level_score": mean_cc,
        "max_cc_level_score":  max_cc,
        # rerank
        "best_rerank_score":   best_rerank,
        "mean_rerank_score":   sum(rerank_scores) / len(rerank_scores),
        "top1_rerank_score":   top1_rerank,
        "top3_rerank_score":   top3_rerank,
        # 频次
        "appearances":         appearances,
        "appearances_log":     math.log1p(appearances),
        "appearances_in_top3": appear_top3,
        "appearances_in_top5": appear_top5,
        # 联合
        "best_cc_x_rerank":    best_cc * best_rerank,
        "freq_x_rerank":       appearances * best_rerank,
    }


def dataset_to_dataframe(
    dataset: list[dict],
    window_size: int = 1,
) -> pd.DataFrame:
    """
    将原始 dataset 转换为 LightGBM 训练用 DataFrame。

    dataset 格式（与 citation_scorer.process_dataset 相同）：
    [
        {
            "query_id": "q001",
            "query": "...",
            "cc_list": [{"cc_id":..., "text":..., "rerank_score":..., "rank":...}, ...],
            "gold_citations": ["Art. 41 OR", ...]   ← 训练时必须提供
        },
        ...
    ]
    """
    rows = []
    for sample in dataset:
        qid   = sample["query_id"]
        gold  = set(sample.get("gold_citations", []))

        cc_list = [
            CCResult(
                cc_id=cc["cc_id"],
                text=cc["text"],
                rerank_score=cc["rerank_score"],
                rank=cc["rank"],
            )
            for cc in sample["cc_list"]
        ]

        ranked = score_citations_for_query(cc_list, window_size=window_size)

        for cit_result in ranked:
            feats = build_feature_row(cit_result, cit_result["all_cc_scores"])
            feats["qid"]      = qid
            feats["citation"] = cit_result["citation"]
            feats["label"]    = int(cit_result["citation"] in gold)
            rows.append(feats)

    df = pd.DataFrame(rows)
    return df


# ─────────────────────────────────────────────
# 3. 训练
# ─────────────────────────────────────────────

def train(
    df: pd.DataFrame,
    objective: str = "lambdarank",   # "lambdarank" | "binary"
    val_ratio: float = 0.2,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
    verbose_eval: int = 50,
    lgbm_params: dict | None = None,
) -> lgb.Booster:
    """
    训练 LightGBM 模型。

    objective:
        "lambdarank" — Learning to Rank，推荐（直接优化 NDCG）
        "binary"     — 二分类，样本少时更稳定
    """
    # 按 query 分割 train/val（保证同一 query 不拆开）
    qids = df["qid"].values
    unique_qids = np.unique(qids)
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
    train_idx, val_idx = next(splitter.split(df, groups=qids))

    df_train = df.iloc[train_idx].copy()
    df_val   = df.iloc[val_idx].copy()

    X_train = df_train[FEATURE_COLS].values
    y_train = df_train["label"].values
    X_val   = df_val[FEATURE_COLS].values
    y_val   = df_val["label"].values

    print(f"Train queries : {df_train['qid'].nunique()}, samples: {len(df_train)}, pos: {y_train.sum()}")
    print(f"Val   queries : {df_val['qid'].nunique()},  samples: {len(df_val)},  pos: {y_val.sum()}")

    if objective == "lambdarank":
        # LambdaRank 需要按 query 分组
        groups_train = df_train.groupby("qid", sort=False).size().values
        groups_val   = df_val.groupby("qid",   sort=False).size().values

        train_data = lgb.Dataset(X_train, label=y_train, group=groups_train,
                                 feature_name=FEATURE_COLS)
        val_data   = lgb.Dataset(X_val,   label=y_val,   group=groups_val,
                                 feature_name=FEATURE_COLS, reference=train_data)

        params = {
            "objective":       "lambdarank",
            "metric":          "ndcg",
            "ndcg_eval_at":    [1, 3, 5],
            "learning_rate":   0.05,
            "num_leaves":      31,
            "min_data_in_leaf": 5,
            "lambda_l1":       0.1,
            "lambda_l2":       0.1,
            "verbose":         -1,
        }

    else:  # binary
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
        val_data   = lgb.Dataset(X_val,   label=y_val,   feature_name=FEATURE_COLS,
                                 reference=train_data)

        params = {
            "objective":        "binary",
            "metric":           "auc",
            "learning_rate":    0.05,
            "num_leaves":       31,
            "min_data_in_leaf": 5,
            "scale_pos_weight": neg_count / max(pos_count, 1),
            "lambda_l1":        0.1,
            "lambda_l2":        0.1,
            "verbose":          -1,
        }

    # 允许外部覆盖参数
    if lgbm_params:
        params.update(lgbm_params)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=True),
            lgb.log_evaluation(verbose_eval),
        ],
    )

    print(f"\nBest iteration: {model.best_iteration}")
    return model


# ─────────────────────────────────────────────
# 4. 预测
# ─────────────────────────────────────────────

def predict(
    model: lgb.Booster,
    dataset: list[dict],
    window_size: int = 1,
    top_k: int = 10,
) -> list[dict]:
    """
    对新数据进行预测，返回每个 query 的 citation 排序结果。
    """
    df = dataset_to_dataframe(dataset, window_size=window_size)

    scores = model.predict(df[FEATURE_COLS].values)
    df["lgbm_score"] = scores

    results = []
    for sample in dataset:
        qid  = sample["query_id"]
        gold = set(sample.get("gold_citations", []))

        sub = df[df["qid"] == qid].copy()
        sub = sub.sort_values("lgbm_score", ascending=False).reset_index(drop=True)

        ranked_citations = []
        for _, row in sub.iterrows():
            ranked_citations.append({
                "rank":        int(row.name) + 1,
                "citation":    row["citation"],
                "lgbm_score":  round(float(row["lgbm_score"]), 4),
                "is_gold":     row["citation"] in gold,
            })

        results.append({
            "query_id":          qid,
            "query":             sample.get("query", ""),
            "gold_citations":    list(gold),
            "ranked_citations":  ranked_citations[:top_k],
        })

    return results


# ─────────────────────────────────────────────
# 5. 评估
# ─────────────────────────────────────────────

def evaluate_lgbm(results: list[dict], top_k_list: list[int] = [1, 3, 5, 10]) -> dict:
    """计算 Recall@K 和 MAP。"""
    recall_at_k = {k: [] for k in top_k_list}
    aps = []

    for sample in results:
        gold   = set(sample["gold_citations"])
        ranked = [r["citation"] for r in sample["ranked_citations"]]
        if not gold:
            continue

        for k in top_k_list:
            hit = len(gold & set(ranked[:k]))
            recall_at_k[k].append(hit / len(gold))

        hits, ap = 0, 0.0
        for i, cit in enumerate(ranked, 1):
            if cit in gold:
                hits += 1
                ap  += hits / i
        aps.append(ap / len(gold) if hits else 0.0)

    metrics = {f"Recall@{k}": round(np.mean(v), 4) for k, v in recall_at_k.items() if v}
    metrics["MAP"] = round(np.mean(aps), 4) if aps else 0.0
    return metrics


# ─────────────────────────────────────────────
# 6. 特征重要性
# ─────────────────────────────────────────────

def print_feature_importance(model: lgb.Booster, top_n: int = 20) -> None:
    importance = pd.DataFrame({
        "feature": model.feature_name(),
        "gain":    model.feature_importance(importance_type="gain"),
        "split":   model.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)

    print(f"\n── Top-{top_n} Feature Importance (by gain) ──")
    print(f"{'Feature':<30} {'Gain':>10} {'Split':>8}")
    print("-" * 52)
    for _, row in importance.head(top_n).iterrows():
        print(f"{row['feature']:<30} {row['gain']:>10.1f} {row['split']:>8.0f}")


# ─────────────────────────────────────────────
# 7. 保存 / 加载
# ─────────────────────────────────────────────

def save_model(model: lgb.Booster, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved → {path}")


def load_model(path: str) -> lgb.Booster:
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded ← {path}")
    return model


# ─────────────────────────────────────────────
# 8. CLI 入口
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Citation LightGBM Pipeline")
    parser.add_argument("--mode",       choices=["train", "predict", "demo"], default="demo")
    parser.add_argument("--data",       type=str, help="Input JSON data path")
    parser.add_argument("--model_out",  type=str, default="citation_model.pkl")
    parser.add_argument("--model_in",   type=str, default="citation_model.pkl")
    parser.add_argument("--out",        type=str, default="predictions.json")
    parser.add_argument("--objective",  choices=["lambdarank", "binary"], default="lambdarank")
    parser.add_argument("--top_k",      type=int, default=10)
    return parser.parse_args()


# ─────────────────────────────────────────────
# 9. Demo（内置模拟数据）
# ─────────────────────────────────────────────

def make_demo_dataset(n_queries: int = 80) -> list[dict]:
    """生成模拟数据集用于演示（实际使用时替换为真实数据）。"""
    import random
    random.seed(42)

    CITATION_POOL = [
        "Art. 97 Abs. 1 OR", "Art. 41 OR", "Art. 42 OR", "Art. 101 OR",
        "Art. 97 OR", "Art. 41 Abs. 1 OR", "BGE 130 III 182", "BGE 133 III 295",
        "BGE 115 II 440", "BGE 145 II 100", "SR 220", "SR 210",
        "Art. 8 ZGB", "Art. 2 ZGB", "BGE 132 III 100", "Art. 18 OR",
    ]

    POSITIVE_SENT_TEMPLATES = [
        "Gemäss {cit} haftet der Schuldner für den entstandenen Schaden.",
        "Gestützt auf {cit} ist die Klage gutzuheissen.",
        "Das Bundesgericht hat in {cit} namentlich festgehalten, dass die Beweislast beim Kläger liegt.",
        "Somit findet {cit} unmittelbar Anwendung auf den vorliegenden Fall.",
        "{cit} i.V.m. Art. 2 ZGB regelt ausdrücklich die Haftungsvoraussetzungen.",
        "Folglich ergibt sich aus {cit}, dass der Beklagte schadensersatzpflichtig ist.",
    ]

    NEGATIVE_SENT_TEMPLATES = [
        "Es ist fraglich, ob {cit} hier anwendbar ist.",
        "Entgegen der h.M. wird in {cit} u.a. ein anderer Ansatz vertreten.",
        "Der genaue Anwendungsbereich von {cit} ist umstritten (str.).",
        "Die Frage, ob {cit} einschlägig ist, wurde offen gelassen.",
        "Vgl. etwa {cit} für eine abweichende Meinung (a.M.).",
    ]

    NEUTRAL_SENT_TEMPLATES = [
        "Vgl. {cit}.",
        "Siehe auch {cit} und weitere Entscheide.",
        "Diese Frage wird allgemein nach {cit} beurteilt.",
    ]

    dataset = []
    for i in range(n_queries):
        # 为每个 query 选 2-3 个 gold citation
        gold = random.sample(CITATION_POOL, k=random.randint(2, 3))

        cc_list = []
        for rank in range(1, random.randint(6, 11)):
            rerank_score = max(0.3, 1.0 - (rank - 1) * 0.08 + random.uniform(-0.05, 0.05))
            # 在 cc 中放若干句子，gold citation 多出现在高排名 cc
            sentences = []
            for g in gold:
                if rank <= 3 or random.random() < 0.4:
                    tmpl = random.choice(POSITIVE_SENT_TEMPLATES)
                    sentences.append(tmpl.format(cit=g))
            # 加一些噪音 citation
            for _ in range(random.randint(1, 3)):
                noise_cit = random.choice(CITATION_POOL)
                if noise_cit not in gold:
                    tmpl = random.choice(NEGATIVE_SENT_TEMPLATES + NEUTRAL_SENT_TEMPLATES)
                    sentences.append(tmpl.format(cit=noise_cit))

            random.shuffle(sentences)
            text = " ".join(sentences) if sentences else "Keine relevanten Bestimmungen gefunden."
            cc_list.append({
                "cc_id":        f"q{i:03d}_cc{rank:02d}",
                "text":         text,
                "rerank_score": round(rerank_score, 4),
                "rank":         rank,
            })

        dataset.append({
            "query_id":       f"q{i:03d}",
            "query":          f"Rechtsfrage {i}: Haftung und Schadenersatz",
            "cc_list":        cc_list,
            "gold_citations": gold,
        })

    return dataset


def run_demo(objective: str = "lambdarank") -> None:
    print("=" * 60)
    print("Citation LightGBM Pipeline — Demo")
    print("=" * 60)

    # 1. 生成模拟数据
    print("\n[1/5] Generating demo dataset...")
    dataset = make_demo_dataset(n_queries=120)
    train_data = dataset[:80]
    test_data  = dataset[80:]
    print(f"  Train: {len(train_data)} queries, Test: {len(test_data)} queries")

    # 2. 构建特征矩阵
    print("\n[2/5] Building feature matrix...")
    df_train = dataset_to_dataframe(train_data, window_size=1)
    print(f"  Shape: {df_train.shape}, Positive ratio: {df_train['label'].mean():.3f}")

    # 3. 训练
    print(f"\n[3/5] Training LightGBM ({objective})...")
    model = train(df_train, objective=objective, num_boost_round=300, verbose_eval=100)

    # 4. 特征重要性
    print_feature_importance(model, top_n=15)

    # 5. 预测 & 评估
    print("\n[4/5] Predicting on test set...")
    predictions = predict(model, test_data, top_k=10)

    print("\n[5/5] Evaluation:")
    metrics = evaluate_lgbm(predictions, top_k_list=[1, 3, 5, 10])
    print(f"\n  {'Metric':<15} {'Score':>6}")
    print("  " + "-" * 22)
    for k, v in metrics.items():
        print(f"  {k:<15} {v:>6.4f}")

    # 展示一个样例
    print("\n── Sample Prediction ──")
    sample = predictions[0]
    print(f"Query: {sample['query']}")
    print(f"Gold:  {sample['gold_citations']}")
    print(f"{'Rank':<5} {'Citation':<35} {'Score':>8} {'Gold?':>6}")
    print("-" * 58)
    for r in sample["ranked_citations"]:
        gold_mark = "✓" if r["is_gold"] else ""
        print(f"{r['rank']:<5} {r['citation']:<35} {r['lgbm_score']:>8.4f} {gold_mark:>6}")


# ─────────────────────────────────────────────
# 10. Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    if args.mode == "demo":
        run_demo(objective="lambdarank")

    elif args.mode == "train":
        assert args.data, "--data 必须指定"
        with open(args.data) as f:
            dataset = json.load(f)
        df = dataset_to_dataframe(dataset)
        model = train(df, objective=args.objective)
        print_feature_importance(model)
        save_model(model, args.model_out)

    elif args.mode == "predict":
        assert args.data and args.model_in, "--data 和 --model_in 必须指定"
        with open(args.data) as f:
            dataset = json.load(f)
        model = load_model(args.model_in)
        predictions = predict(model, dataset, top_k=args.top_k)
        metrics = evaluate_lgbm(predictions)
        print("Metrics:", metrics)
        with open(args.out, "w") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        print(f"Predictions saved → {args.out}")