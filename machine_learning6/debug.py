import json
import pprint

# for line in open("../data/ml6/total.jsonl"):
# for line in open("../data/ml6/train.jsonl"):
# for line in open("../data/ml6/predict.jsonl"):
# for line in open("../data/ml6/dev.jsonl"):
for line in open("../data/ml6/output.jsonl"):
    d = json.loads(line.strip())
    pprint.pprint(d)
    break