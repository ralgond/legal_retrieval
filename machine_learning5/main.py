"""
Demo：用你的JSON格式数据 + valid早停 的完整流程
"""

import json
import tempfile
import os
from citation_gold_classifier import DataLoader, CitationGoldClassifier

# ── 构造合成train/valid数据（模拟你的JSON格式）──────────────────────────────

TRAIN_DATA = []
with open('../data/ml5/train.jsonl', encoding='utf-8') as inf:
    for line in inf:
        TRAIN_DATA.append(json.loads(line.strip("\n")))

VALID_DATA = []
with open('../data/ml5/valid.jsonl', encoding='utf-8') as inf:
    for line in inf:
        VALID_DATA.append(json.loads(line.strip("\n")))



def write_tmp_json(data: list[dict]) -> str:
    """将数据写入临时JSON文件，返回路径"""
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def demo_lgbm_early_stopping():
    print("=" * 65)
    print("DEMO: LightGBM 早停  (监控 valid binary_logloss)")
    print("=" * 65)

    loader = DataLoader(context_sentences=2)

    # ── 从文件加载（模拟你的真实用法）
    train_path = write_tmp_json(TRAIN_DATA)
    valid_path = write_tmp_json(VALID_DATA)

    train_instances = loader.load_file(train_path)
    valid_instances = loader.load_file(valid_path)

    os.unlink(train_path)
    os.unlink(valid_path)

    print(f"\n训练集: {len(train_instances)} citations  "
          f"(gold={sum(i.is_gold for i in train_instances)})")
    print(f"验证集: {len(valid_instances)} citations  "
          f"(gold={sum(i.is_gold for i in valid_instances)})\n")

    clf = CitationGoldClassifier(
        model_type="lgbm",
        n_estimators=300,          # 上限，早停会提前截断
        learning_rate=0.05,
        num_leaves=31,
        early_stopping_rounds=30,  # valid 30轮无改善则停止
    )
    clf.fit(train_instances, valid_instances=valid_instances)

    print(f"\n最佳迭代轮次: {clf.best_iteration_}")
    print("\n── Valid集最终指标 ──")
    for k, v in clf.valid_metrics_.items():
        if not k.startswith("support"):
            print(f"  {k:<20}: {v:.4f}")

    print("\n── Valid集逐citation预测 ──")
    probs = clf.predict_proba(valid_instances)
    print(f"  {'Citation ID':<35} {'Prob':>6}  {'Pred':>5}  {'True':>5}")
    print("  " + "-" * 58)
    for inst, prob in zip(valid_instances, probs):
        pred = "gold" if prob >= 0.5 else "neg"
        true = "gold" if inst.is_gold else "neg"
        mark = "✓" if pred == true else "✗"
        print(f"  {inst.citation_id:<35} {prob:>6.3f}  {pred:>5}  {true:>5} {mark}")

    return clf


def demo_lr_with_valid():
    print("\n" + "=" * 65)
    print("DEMO: LogisticRegression + valid评估（无需早停）")
    print("=" * 65)

    loader = DataLoader(context_sentences=2)
    train_instances = loader.load(TRAIN_DATA)
    valid_instances = loader.load(VALID_DATA)

    clf = CitationGoldClassifier(model_type="lr", C=1.0)
    clf.fit(train_instances, valid_instances=valid_instances)
    # fit() 内部已打印valid指标


def demo_load_with_retrieval_scores():
    print("\n" + "=" * 65)
    print("DEMO: load_with_retrieval_scores（同时获取CC级检索分数）")
    print("=" * 65)

    loader = DataLoader()
    instances, cc_df = loader.load_with_retrieval_scores(TRAIN_DATA)

    print(f"\nCitation实例数: {len(instances)}")
    print(f"\nCC级DataFrame (供LTR):")
    print(cc_df[["query_id", "cc_id", "dense_score", "rerank_score", "label"]].to_string(index=False))


if __name__ == "__main__":
    clf = demo_lgbm_early_stopping()
    print(clf.feature_importance(top_n=100))
    # demo_lr_with_valid()
    # demo_load_with_retrieval_scores()