import json
import sys
import pandas as pd

TOPK = int(sys.argv[1])

l = None
with open("../data/rule_based/predictions.json") as inf:
    l = json.load(inf)

query_id_l = []
predicted_citations_l = []
for d in l:
    query_id = d['query_id']
    c_l = []
    ranked_citations = d['ranked_citations']
    for c in ranked_citations:
        c_l.append(c['citation'])
    c_l = c_l[:TOPK]
    predicted_citations = ';'.join(c_l)

    query_id_l.append(query_id)
    predicted_citations_l.append(predicted_citations)

df = pd.DataFrame({"query_id":query_id_l, "predicted_citations":predicted_citations_l})
df.to_csv("../data/rule_based/predictions.csv", index=False)

