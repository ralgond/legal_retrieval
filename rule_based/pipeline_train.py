import pandas as pd
import os
import numpy as np

DATA_DIR = "../data/rule_based"
OUTPUT_DIR = "../data/rule_based"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 加载数据 ──────────────────────────────────────────────────────────────────
print("Loading data...")
court_consideration_df = pd.read_csv("../data/court_considerations_maped.csv")
court_consideration_d = dict(zip(court_consideration_df['citation'].tolist(), court_consideration_df['text'].tolist()))

import common
test_candidate_d = common.read_candidate(f"{DATA_DIR}/raw_test_candidate.pkl", court_consideration_d)


def build_test_dataframe(query_id, cc_text_list):

    dfs = []

    for cc_id, cc_text in enumerate(cc_text_list):

        df_cc = extract_features_from_text(cc_text)

        if len(df_cc) == 0:
            continue

        df_cc["query_id"] = query_id
        df_cc["cc_id"] = cc_id

        dfs.append(df_cc)

    if len(dfs) == 0:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)

def f1_at_k(df, k=5):

    f1s = []

    for qid, group in df.groupby("query_id"):
        group = group.sort_values("score", ascending=False)

        topk = group.head(k)

        y_true = group["label"].values
        y_pred = np.zeros_like(y_true)

        y_pred[:k] = 1  # topK预测为1

        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()

        if tp == 0:
            f1s.append(0)
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1s.append(2 * precision * recall / (precision + recall))

    return sum(f1s) / len(f1s)

def prepare_lgb_data(df):

    feature_cols = [
        c for c in df.columns
        if c not in ["query_id", "citation", "label"]
    ]

    X = df[feature_cols]
    y = df["label"]

    # group（每个 query 的样本数）
    group = df.groupby("query_id").size().values

    return X, y, group, feature_cols

def print_feature_importance(model, feature_names, top_k=30):

    gain = model.feature_importance(importance_type="gain")
    split = model.feature_importance(importance_type="split")

    df = pd.DataFrame({
        "feature": feature_names,
        "gain": gain,
        "split": split
    })

    # 归一化（方便看）
    df["gain_pct"] = df["gain"] / df["gain"].sum()
    df["split_pct"] = df["split"] / df["split"].sum()

    # 按 gain 排序（最重要）
    df = df.sort_values("gain", ascending=False)

    print("\n=== Feature Importance (Top {}) ===".format(top_k))
    print(df.head(top_k).to_string(index=False))

    return df

def inference(model, test_data, feature_cols, K=5):
    all_results = []

    for qid, item in test_data.items():

        # 1. build features
        df = build_test_dataframe(
            query_id=qid,
            cc_text_list=item["cc_text_list"]
        )

        if len(df) == 0:
            continue

        # 2. predict
        df = predict_scores(model, df, feature_cols)

        # 3. aggregate (critical)
        df = aggregate_citations(df)

        # 4. topK
        topk = df.sort_values("score", ascending=False).head(K)

        all_results.append({
            "query_id": qid,
            "top_citations": topk["citation"].tolist()
        })

    return all_results

import lightgbm as lgb

def train():
    df_train = pd.read_csv("../data/rule_based/train_df.csv")
    X_train, y_train, group_train, feature_cols = prepare_lgb_data(df_train)
    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        group=group_train
    )

    df_valid = pd.read_csv("../data/rule_based/valid_df.csv")
    X_valid, y_valid, group_valid, feature_cols = prepare_lgb_data(df_valid)
    valid_data = lgb.Dataset(
        X_valid,
        label=y_valid,
        group=group_valid
    )

    
    model = lgb.train(
        {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1, 3, 5],
            "learning_rate": 0.01,
            "num_leaves": 32,
            "max_depth": 6
        },
        train_data,
        num_boost_round=200,
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=10)
        ]
    )

    print_feature_importance(model, feature_cols, top_k=20)

    return model

def validate(model):
    df_valid = pd.read_csv("../data/rule_based/valid_df.csv")
    X_valid, y_valid, group_valid, feature_cols = prepare_lgb_data(df_valid)
    valid_data = lgb.Dataset(
        X_valid,
        label=y_valid,
        group=group_valid
    )

    df_valid["score"] = model.predict(df_valid[feature_cols])

    df_valid_agg = (
        df_valid
        .groupby(["query_id", "citation"], as_index=False)
        .agg({
            "score": "max",
            "label": "max"
        })
    )

    for k in [5,7,10,12,15,17,20,22,25,27,30,32,35,37,40,42,45,47,50]:
        print(f"F1@{k}:", f1_at_k(df_valid_agg, k))


def predict(model):
    test_data = {}
    for query_id, d in test_candidate_d.items():
        rerank_l = d['rerank']
        item = dict()
        cc_text_list = [hit['text'] for hit, score in rerank_l]
        item['cc_text_list'] = cc_text_list
        test_data[query_id] = item

    results = inference(model, test_data, feature_cols, K=5)

    print(results)

def main():
    model = train()
    validate(model)
    predict(model)

if __name__ == "__main__":
    main()