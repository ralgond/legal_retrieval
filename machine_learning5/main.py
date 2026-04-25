"""
Demo：用你的JSON格式数据 + valid早停 的完整流程
"""

import json
import tempfile
import os
from citation_gold_classifier_with_scores import DataLoader, CitationGoldClassifier, CitationExtractor
from collections import defaultdict
import pandas as pd

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


def predict_test_file(extractor: CitationExtractor, 
                      classifier: CitationGoldClassifier, 
                      test_path: str, 
                      output_path: str, 
                      threshold: float = 0.5):
    """
    读取 test.jsonl，预测每个 citation 是否为 gold，并将结果保存。
    
    输入格式 (test.jsonl): 每行一个 JSON, 包含 query_id 和 cc_list
    输出格式: 每行一个 JSON, 包含 query_id, cc_id, citation_id 以及 predicted_gold (0/1)
    """
    results = []
    
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            query_id = entry["query_id"]
            cc_list = entry["cc_list"]
            
            for cc in cc_list:
                cc_id = cc.get("cc_id", "unknown_cc")
                # 提取该 CC 中的所有 citation 实例
                # 注意：test 集没有 gold_citations，传入空列表即可
                instances = extractor.extract(
                    cc_text=cc["text"],
                    cc_id=cc_id,
                    query_id=query_id,
                    gold_citations=[], 
                    dense_score=cc.get("dense_score", 0.0),
                    sparse_score=cc.get("sparse_score", 0.0),
                    rerank_score=cc.get("rerank_score", 0.0)
                )
                
                if not instances:
                    continue
                
                # 批量预测概率
                probs = classifier.predict_proba(instances)
                
                for inst, prob in zip(instances, probs):
                    results.append({
                        "query_id": query_id,
                        "cc_id": cc_id,
                        "citation_id": inst.citation_id,
                        "gold_prob": round(float(prob), 4),
                        "is_gold_pred": int(prob >= threshold)
                    })
    
    # 保存结果
    # with open(output_path, "w", encoding="utf-8") as f:
    #     for res in results:
    #         f.write(json.dumps(res, ensure_ascii=False) + "\n")
    
    query_id_2_predicted_citation_l_d = defaultdict(list)
    for res in results:
        if res['is_gold_pred']:
            query_id_2_predicted_citation_l_d[res['query_id']].append(res['citation_id'])

    query_id_l = sorted(list(query_id_2_predicted_citation_l_d.keys()))
    predicted_citations_l = []
    for query_id in query_id_l:
        predicted_citations_l.append(';'.join(query_id_2_predicted_citation_l_d[query_id]))

    result_df = pd.DataFrame({'query_id':query_id_l, "predicted_citations":predicted_citations_l})
    result_df.to_csv(output_path, index=False)

    print(f"预测完成，结果已保存至: {output_path}")
    return results

# ─────────────────────────────────────────────
# 使用示例
# ─────────────────────────────────────────────
"""
# 假设你已经训练好了 clf
# clf = CitationGoldClassifier(model_type='lgbm').fit(train_instances, valid_instances)

# 1. 确保 feature_builder 绑定了 extractor (如果之前没绑定，可以在训练后手动绑定)
clf.feature_builder.extractor = CitationExtractor(context_sentences=2)

# 2. 执行预测
predictions = predict_test_file(
    classifier=clf, 
    test_path="test.jsonl", 
    output_path="citation_predictions.jsonl",
    threshold=0.5  # 你可以根据 valid 集的 PR 曲线调整此阈值
)
"""

if __name__ == "__main__":
    clf = demo_lgbm_early_stopping()
    predictions = predict_test_file(
        extractor=CitationExtractor(context_sentences=2),
        classifier=clf, 
        test_path="../data/ml5/test.jsonl", 
        output_path="../data/ml5/citation_predictions.csv",
        threshold=0.5  # 你可以根据 valid 集的 PR 曲线调整此阈值
    )
    print(clf.feature_importance(top_n=100))
    # demo_lr_with_valid()
    # demo_load_with_retrieval_scores()