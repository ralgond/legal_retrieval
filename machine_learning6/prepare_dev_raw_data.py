import json

query_id_set = set()
for line in open("../data/ml6/dev.jsonl"):
    d = json.loads(line.strip())
    query_id_set.add(d['query_id'])

with open("../data/ml6/dev_raw.jsonl", "w+", encoding='utf-8') as of:
    for line in open("../data/ml6/total.jsonl"):
        d = json.loads(line.strip())
        if d['query_id'] in query_id_set:
            of.write(line)