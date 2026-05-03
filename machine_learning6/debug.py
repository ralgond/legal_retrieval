# import json

# with open("../data/ml6/train_samples.jsonl") as f:
#     for i, line in enumerate(f):
#         s = json.loads(line)
#         print(f"--- sample {i} ---")
#         print(f"label: {s['label']}, neg_type: {s['neg_type']}")
#         print(f"citation: {s['citation']}")
#         print(f"cc_text: {s['cc_text'][:300]}")
#         print()
#         if i >= 5:
#             break

# =============================================================================

from transformers import AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained("/root/.cache/modelscope/hub/models/ralgond/legal-swiss-roberta-large")

with open("../data/ml6/train_samples.jsonl") as f:
    samples = [json.loads(l) for l in f]

# 找一对同 cc_id 的正负例
from collections import defaultdict
groups = defaultdict(lambda: {"pos": [], "neg": []})
for s in samples:
    key = (s["query_id"], s["cc_id"])
    groups[key]["pos" if s["label"] == 1 else "neg"].append(s)

for key, g in groups.items():
    if g["pos"] and g["neg"]:
        p, n = g["pos"][0], g["neg"][0]
        print("=== POSITIVE ===")
        print(f"citation : {p['citation']}")
        print(f"cc_text  : {p['cc_text']}")
        print(f"input    : Query: {p['query']} Context: {p['cc_text']} Citation: {p['citation']}")
        print()
        print("=== NEGATIVE ===")
        print(f"citation : {n['citation']}")
        print(f"cc_text  : {n['cc_text']}")
        print(f"input    : Query: {n['query']} Context: {n['cc_text']} Citation: {n['citation']}")
        print()
        # 检查 cc_text 是否相同
        print(f"cc_text 相同: {p['cc_text'] == n['cc_text']}")
        # 检查 citation 是否出现在 cc_text 里
        print(f"pos citation 在 cc_text 里: {p['citation'] in p['cc_text'] or p['citation'].replace('_',' ') in p['cc_text']}")
        print(f"neg citation 在 cc_text 里: {n['citation'] in n['cc_text'] or n['citation'].replace('_',' ') in n['cc_text']}")
        break