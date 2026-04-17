import pandas as pd
print("Loading data...")
court_consideration_df = pd.read_csv("../data/court_considerations.csv")
court_consideration_d = dict(zip(court_consideration_df['citation'].tolist(), court_consideration_df['text'].tolist()))

import common
test_candidate_d = common.read_candidate("../data/rule_based/raw_test_candidate.pkl", court_consideration_d)

def build_dataset():
    
    ret = []
    for query_id, data_d in test_candidate_d.items():
        d = {}
        d['query_id'] = query_id
        d['query'] = query_id

        cc_list = []
        cc_list_base = data_d['rerank']
        for rank, cc in enumerate(cc_list_base,start=1):
            hit, score = cc
            cc_id = hit['citation']
            text = hit['text']
            d2 = {}
            d2["cc_id"] = cc_id
            d2['text'] = text
            d2['rerank_score'] = score
            d2['rank'] = rank
            cc_list.append(d2)

        d['cc_list'] = cc_list

        ret.append(d)

    return ret

ds = build_dataset()

import json
with open("../data/rule_based/test.json", "w+") as of:
    json.dump(ds, of)