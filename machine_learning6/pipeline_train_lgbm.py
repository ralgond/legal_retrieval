"""
pipeline_train_lgbm.py
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

DATA_DIR   = "../data/ml6/lgbm_data"
OUTPUT_DIR = "../data/ml6/lgbm_model"
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
        "max_depth":        6,
        "min_data_in_leaf": 1,
        # "feature_fraction": 0.8,
        # "bagging_fraction": 0.8,
        # "bagging_freq":     5,
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