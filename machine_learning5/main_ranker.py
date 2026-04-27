"""
Demo：单层 LambdaRank（group = query_id）完整流程
"""

import json
import os
import tempfile
from citation_ranker import DataLoader, CitationRanker, EmbeddingCitationFeatureBuilder

TRAIN_RECORDS = []
with open('../data/ml5/s3_train.jsonl', encoding='utf-8') as inf:
    for line in inf:
        TRAIN_RECORDS.append(json.loads(line.strip("\n")))

VALID_RECORDS = []
with open('../data/ml5/s3_valid.jsonl', encoding='utf-8') as inf:
    for line in inf:
        VALID_RECORDS.append(json.loads(line.strip("\n")))

TEST_RECORDS = []
with open('../data/ml5/test.jsonl', encoding='utf-8') as inf:
    for line in inf:
        TEST_RECORDS.append(json.loads(line.strip("\n")))


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    train_path  = tempfile.mktemp(suffix="_train.jsonl")
    valid_path  = tempfile.mktemp(suffix="_valid.jsonl")
    test_path   = tempfile.mktemp(suffix="_test.jsonl")
    output_path = "../data/ml5/predictions.jsonl"

    write_jsonl(TRAIN_RECORDS, train_path)
    write_jsonl(VALID_RECORDS, valid_path)
    write_jsonl(TEST_RECORDS,  test_path)

    loader = DataLoader(context_sentences=2)

    loader.load_query_map("../data/train_rewrite_001.csv", query_col="query2")
    loader.load_query_map("../data/valid_rewrite_001.csv", query_col="query2")
    loader.load_query_map("../data/test_rewrite_001.csv",  query_col="query")

    train_instances = loader.load_file(train_path)
    train_instances = loader.sample_instances(train_instances, neg_pos_ratio=10, hard_neg_keep=30)
    valid_instances = loader.load_file(valid_path)

    # ── 训练集统计 ────────────────────────────────────────────────────────
    from collections import Counter
    q_counts = Counter(inst.query_id for inst in train_instances)
    print("=" * 60)
    print("训练集统计（group = query_id）")
    print("=" * 60)
    print(f"  总 citations : {len(train_instances)}")
    print(f"  gold         : {sum(i.is_gold for i in train_instances)}")
    print(f"  queries      : {len(q_counts)}")
    print(f"  citations/query（均值）: {len(train_instances)/len(q_counts):.1f}")
    print()

    # ── 训练（含早停）────────────────────────────────────────────────────
    ranker = CitationRanker(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        early_stopping_rounds=50,
        eval_at=[1, 3, 5, 25, 200],
    )
    ranker.feature_builder = EmbeddingCitationFeatureBuilder(
        model_name="/root/.cache/modelscope/hub/models/ralgond/legal-swiss-roberta-large",
        batch_size=64,                    # A100可以开到256
        device="cuda",
    )
    ranker.fit(train_instances, valid_instances=valid_instances)

    # ── Valid 指标 ────────────────────────────────────────────────────────
    print("\n── Valid 排序指标 ──")
    for k, v in ranker.valid_metrics_.items():
        print(f"  {k:<12}: {v:.4f}")

    # ── 特征重要性 ────────────────────────────────────────────────────────
    print("\n── 特征重要性（Top 10）──")
    fi = ranker.feature_importance(top_n=10)
    for _, row in fi.iterrows():
        bar = "█" * int(row["importance"] * 20 / (fi["importance"].max() + 1e-9))
        print(f"  {row['feature']:<35} {row['importance']:>6.0f}  {bar}")

    # ── Test 预测 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Test 预测（group=query_id，rank 是 query 内全局排名）")
    print("=" * 60)

    results = loader.predict_file(test_path, ranker, output_path=output_path)

    # for entry in results:
    #     print(f"\n  query_id: {entry['query_id']}")
    #     for cc in entry["cc_list"]:
    #         print(f"    cc_id: {cc['cc_id']}")
    #         for cit in cc["citations"]:
    #             print(f"      rank={cit['rank']:>3}  score={cit['score']:>7.4f}  {cit['citation_id']}")

    print(f"\n结果已写入: {output_path}")

    for p in [train_path, valid_path, test_path]:
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()