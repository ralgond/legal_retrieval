import common
import pandas as pd

# ── 加载数据 ──────────────────────────────────────────────────────────────────
print("Loading data...")
court_consideration_df = pd.read_csv("../data/court_considerations.csv")
court_consideration_d = dict(zip(court_consideration_df['citation'].tolist(), court_consideration_df['text'].tolist()))

train_candidate_d = common.read_candidate("../data/ml5/raw_train_candidate.pkl", court_consideration_d)
valid_candidate_d = common.read_candidate("../data/ml5/raw_valid_candidate.pkl", court_consideration_d)
test_candidate_d = common.read_candidate("../data/ml5/raw_test_candidate.pkl", court_consideration_d)


train_df = pd.read_csv("../data/train_rewrite_001.csv")
train_qid_2_gold_citations = {}
for query_id, gold_citations in zip(train_df['query_id'], train_df['gold_citations']):
    train_qid_2_gold_citations[query_id] = set(gold_citations.split(";"))

for query_id, d in train_candidate_d.items():
    gold_citation_set = set(train_qid_2_gold_citations[query_id])
    
    rerank_l = d['rerank']
    
    cc_contains_gold_cnt = 0
    for cc,score in rerank_l:
        for gold in gold_citation_set:
            if gold in cc['text']:
                cc_contains_gold_cnt += 1
    print(f"{cc_contains_gold_cnt}/{len(rerank_l)}, {cc_contains_gold_cnt/len(rerank_l)}")
                