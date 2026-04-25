"""
Demo：JSONL 格式的完整流程
  train.jsonl → fit（含早停）
  valid.jsonl → 早停监控
  test.jsonl  → predict_file / predict_dataset（无 gold_citations）
"""

import json
import os
import tempfile
from citation_gold_classifier_with_scores_2 import DataLoader, CitationGoldClassifier

# ─────────────────────────────────────────────────────────────────────────────
# 合成数据（模拟 JSONL 每行内容）
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_DATA = []
with open('../data/ml5/s3_train.jsonl', encoding='utf-8') as inf:
# with open('../data/ml5/train.jsonl', encoding='utf-8') as inf:
    for line in inf:
        TRAIN_DATA.append(json.loads(line.strip("\n")))

VALID_DATA = []
with open('../data/ml5/s3_valid.jsonl', encoding='utf-8') as inf:
    for line in inf:
        VALID_DATA.append(json.loads(line.strip("\n")))


TEST_RECORDS = []
with open('../data/ml5/test.jsonl', encoding='utf-8') as inf:
    for line in inf:
        TEST_RECORDS.append(json.loads(line.strip("\n")))


def write_jsonl(records: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    # ── 写临时 JSONL 文件（模拟你的真实文件）
    train_path  = tempfile.mktemp(suffix="_train.jsonl")
    valid_path  = tempfile.mktemp(suffix="_valid.jsonl")
    test_path   = tempfile.mktemp(suffix="_test.jsonl")
    output_path = "../data/ml5/predictions.jsonl"

    write_jsonl(TRAIN_DATA, train_path)
    write_jsonl(VALID_DATA, valid_path)
    write_jsonl(TEST_RECORDS,  test_path)

    loader = DataLoader(context_sentences=2)

    # ── 1. 加载 train / valid ─────────────────────────────────────────────
    train_instances = loader.load_file(train_path)
    valid_instances = loader.load_file(valid_path)

    print("=" * 60)
    print("训练 & 早停")
    print("=" * 60)
    print(f"训练集: {len(train_instances)} citations  "
          f"(gold={sum(i.is_gold for i in train_instances)})")
    print(f"验证集: {len(valid_instances)} citations  "
          f"(gold={sum(i.is_gold for i in valid_instances)})\n")

    clf = CitationGoldClassifier(
        model_type="lgbm",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        early_stopping_rounds=30,
    )
    clf.fit(train_instances, valid_instances=valid_instances)

    # ── 2. predict test（方式1：从文件，写出结果）────────────────────────
    print("\n" + "=" * 60)
    print("Test 预测（predict_file，写出 JSONL）")
    print("=" * 60)

    results = loader.predict_file(
        path=test_path,
        classifier=clf,
        threshold=0.5,
        output_path=output_path,   # 同时写出预测结果
    )

    # print(f"结果已写入: {output_path}")
    # print("\n── 预测结果 ──")
    # for entry in results:
    #     print(f"\n  query_id: {entry['query_id']}")
    #     for cc in entry["cc_list"]:
    #         print(f"    cc_id: {cc['cc_id']}")
    #         for cit in cc["citations"]:
    #             flag = "★ gold" if cit["predicted_gold"] else "  neg "
    #             print(f"      {flag}  {cit['citation_id']:<30}  prob={cit['gold_prob']:.4f}")

    # # ── 验证写出的 JSONL 格式正确 ─────────────────────────────────────────
    # print("\n── 输出文件内容（原始 JSONL） ──")
    # with open(output_path, encoding="utf-8") as f:
    #     for line in f:
    #         obj = json.loads(line)
    #         print(f"  query_id={obj['query_id']}  "
    #               f"cc数={len(obj['cc_list'])}  "
    #               f"citation总数={sum(len(cc['citations']) for cc in obj['cc_list'])}")

    # ── 3. predict test（方式2：直接传 list[dict]）───────────────────────
    print("\n── predict_dataset（直接传数据，不读文件）──")
    results2 = loader.predict_dataset(TEST_RECORDS, clf, threshold=0.5)
    assert len(results2) == len(TEST_RECORDS), "结果条数应与输入一致"
    print(f"  输入 {len(TEST_RECORDS)} 条 query，输出 {len(results2)} 条  ✓")

    # ── 清理临时文件
    for p in [train_path, valid_path, test_path]:
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()