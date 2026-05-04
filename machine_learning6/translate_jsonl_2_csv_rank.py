import json
import pandas as pd

print("Loading data...")
court_consideration_df = pd.read_csv("../data/court_considerations.csv")
court_consideration_d = dict(zip(court_consideration_df['citation'].tolist(), court_consideration_df['text'].tolist()))

law_df = pd.read_csv("../data/laws_de.csv")
law_d = dict(zip(law_df['citation'].tolist(), law_df['text'].tolist()))


l = []

def parse(d):
    query_id = d['query_id']
    l = []
    for citation in d['global_ranked_citations']:
        l.append((citation['citation_id'], citation['global_rank']))

    l.sort(key=lambda x: x[1])
    return query_id, [citation for citation,_ in l if citation in law_d or citation in court_consideration_d]
    

query_id_l = []
predicted_citations_l = []
for line in open("../data/ml6/output.jsonl"):
    query_id, l = parse(json.loads(line.strip()))
    query_id_l.append(query_id)
    predicted_citations_l.append(';'.join(l[:25]))

df = pd.DataFrame({'query_id': query_id_l, "predicted_citations": predicted_citations_l})
df.to_csv("../data/ml6/citation_predictions.csv", index=False)
