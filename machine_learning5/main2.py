"""
Demo：完整流程
  train_and_evaluate()  →  CitationGoldClassifier + LTRModel（均含早停）
  predict_test()        →  test 集排序结果
"""
 
import json, tempfile, os
from ltr_pipeline import train_and_evaluate, predict_test
 

 
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


TEST_DATA = []
with open('../data/ml5/test.jsonl', encoding='utf-8') as inf:
    for line in inf:
        TEST_DATA.append(json.loads(line.strip("\n")))

def write_json(data, suffix=".json") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path

if __name__ == "__main__":
    train_path = write_json(TRAIN_DATA)
    valid_path = write_json(VALID_DATA)
    test_path  = write_json(TEST_DATA)
    out_path   = "/tmp/test_predictions.json"
 
    # ── 训练
    clf, feat_builder, ltr = train_and_evaluate(train_path, valid_path, citation_model_type="lgbm")
 
    # ── 预测 test
    print("\n" + "=" * 60)
    print("Step 5: 预测 Test 集")
    print("=" * 60)
    ranked = predict_test(test_path, clf, feat_builder, ltr, output_path=out_path)
 
    print("\n── Test 排序结果 ──")
    print("\n── Test 排序结果（query 内跨 CC 混排，每行一个 citation）──")
    print(ranked[["query_id", "cc_id", "citation_id", "ltr_score", "rank"]].to_string(index=False))
 
    print(f"\n── 写入文件预览 ({out_path}) ──")
    with open(out_path, encoding="utf-8") as f:
        print(json.dumps(json.load(f), indent=2, ensure_ascii=False)[:600] + "\n...")
 
    for p in [train_path, valid_path, test_path]:
        os.unlink(p)