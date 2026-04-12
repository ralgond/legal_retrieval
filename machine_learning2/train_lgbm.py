"""
train_lgbm.py
使用 LightGBM lambdarank 训练排序模型。

依赖：
  pip install lightgbm optuna matplotlib
"""

from __future__ import annotations
import os
import json
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR   = "../data/ml2/lgbm_data"
OUTPUT_DIR = "../data/ml2/lgbm_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 加载数据 ──────────────────────────────────────────────────────────────────
X_tr = np.load(f"{DATA_DIR}/train_features.npy")
y_tr = np.load(f"{DATA_DIR}/train_labels.npy")
g_tr = np.load(f"{DATA_DIR}/train_groups.npy")

X_va = np.load(f"{DATA_DIR}/valid_features.npy")
y_va = np.load(f"{DATA_DIR}/valid_labels.npy")
g_va = np.load(f"{DATA_DIR}/valid_groups.npy")

with open(f"{DATA_DIR}/feature_names.txt") as f:
    feature_names = [l.strip() for l in f.readlines()]

print(f"Train: X={X_tr.shape}  queries={len(g_tr)}  pos_rate={y_tr.mean():.4f}")
print(f"Valid: X={X_va.shape}  queries={len(g_va)}  pos_rate={y_va.mean():.4f}")

train_ds = lgb.Dataset(X_tr, label=y_tr, group=g_tr, feature_name=feature_names)
valid_ds = lgb.Dataset(X_va, label=y_va, group=g_va, feature_name=feature_names,
                       reference=train_ds)

# ── 基础参数（固定） ──────────────────────────────────────────────────────────
BASE_PARAMS = dict(
    # objective        = "binary",
    # metric           = 'AUC',
    # is_unbalance     = True,
    objective        = "lambdarank",
    metric           = "ndcg",
    ndcg_eval_at     = [1, 5, 10],
    label_gain       = [0, 1],          # 二值标签: 0 irrelevant, 1 relevant
    boosting_type    = "gbdt",
    n_jobs           = -1,
    verbose          = 2,
    seed             = 42,
    feature_pre_filter = False,
)

# ══════════════════════════════════════════════════════════════════════════════
# 方案 A：直接训练（快速验证）
# ══════════════════════════════════════════════════════════════════════════════
def train_default() -> lgb.Booster:
    params = {
        **BASE_PARAMS,
        "learning_rate":    0.01,
        "num_leaves":       32,
        "max_depth":        -1,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "lambda_l1":        0.1,
        "lambda_l2":        0.1,
        "lambdarank_truncation_level": 200,
    }
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=20),
    ]
    booster = lgb.train(
        params,
        train_ds,
        num_boost_round   = 1,
        valid_sets        = [valid_ds],
        valid_names       = ["valid"],
        callbacks         = callbacks,
    )
    return booster


# ══════════════════════════════════════════════════════════════════════════════
# 方案 B：Optuna 超参搜索
# ══════════════════════════════════════════════════════════════════════════════
def optuna_objective(trial: optuna.Trial) -> float:
    params = {
        **BASE_PARAMS,
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 15, 255),
        "max_depth":        trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq":     trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1":        trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
        "lambda_l2":        trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True),
        "lambdarank_truncation_level": trial.suggest_int("lambdarank_truncation_level", 5, 30),
    }
    callbacks = [
        lgb.early_stopping(stopping_rounds=30, verbose=False),
        lgb.log_evaluation(period=9999),   # 静音
    ]
    booster = lgb.train(
        params,
        train_ds,
        num_boost_round = 500,
        valid_sets      = [valid_ds],
        valid_names     = ["valid"],
        callbacks       = callbacks,
    )
    # 优化目标：valid NDCG@20
    score = booster.best_score["valid"]["ndcg@20"]
    return score


def train_with_optuna(n_trials: int = 30) -> lgb.Booster:
    study = optuna.create_study(direction="maximize",
                                study_name="lgbm_lambdarank")
    study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest trial NDCG@20: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # 用最优超参重新训练（更多轮次）
    best_params = {**BASE_PARAMS, **study.best_params}
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=20),
    ]
    booster = lgb.train(
        best_params,
        train_ds,
        num_boost_round = 2000,
        valid_sets      = [valid_ds],
        valid_names     = ["valid"],
        callbacks       = callbacks,
    )
    # 保存最优超参
    with open(f"{OUTPUT_DIR}/best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)

    return booster


# ══════════════════════════════════════════════════════════════════════════════
# 可视化
# ══════════════════════════════════════════════════════════════════════════════
def plot_importance(booster: lgb.Booster):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, imp_type in zip(axes, ["gain", "split"]):
        importance = booster.feature_importance(importance_type=imp_type)
        names      = booster.feature_name()
        order      = np.argsort(importance)
        ax.barh(np.array(names)[order], importance[order])
        ax.set_title(f"Feature Importance ({imp_type})")
        ax.set_xlabel(imp_type)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/feature_importance.png"
    plt.savefig(path, dpi=150)
    print(f"Saved feature importance plot → {path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["default", "optuna"], default="default",
                        help="default: 直接训练; optuna: 超参搜索后训练")
    parser.add_argument("--n_trials", type=int, default=30,
                        help="Optuna 搜索次数")
    args = parser.parse_args()

    if args.mode == "optuna":
        print(f"=== Optuna 超参搜索 ({args.n_trials} trials) ===")
        booster = train_with_optuna(n_trials=args.n_trials)
    else:
        print("=== 默认参数训练 ===")
        booster = train_default()

    # 保存模型
    model_path = f"{OUTPUT_DIR}/lgbm_ranker.txt"
    booster.save_model(model_path)
    print(f"Model saved → {model_path}")
    print(f"Best iteration : {booster.best_iteration}")
    print(f"Best valid NDCG: {booster.best_score}")

    plot_importance(booster)
