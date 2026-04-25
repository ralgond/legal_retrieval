import json
import pandas as pd

l = []

def parse(d):
    query_id = d['query_id']
    l = []
    for cc in d['cc_list']:
        for citation in cc['citations']:
            l.append((citation['citation_id'], citation['rank']))

    l.sort(key=lambda x: x[1])
    return query_id, [citation for citation,_ in l]
    

query_id_l = []
predicted_citations_l = []
for line in open("../data/ml5/predictions.jsonl"):
    query_id, l = parse(json.loads(line.strip()))
    query_id_l.append(query_id)
    predicted_citations_l.append(';'.join(l[:30]))

df = pd.DataFrame({'query_id': query_id_l, "predicted_citations": predicted_citations_l})
df.to_csv("../data/ml5/citation_predictions.csv", index=False)