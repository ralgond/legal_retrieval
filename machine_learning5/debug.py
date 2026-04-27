import common
import re
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

CITATION_RE = re.compile(
        r"""(?:
            SR\s*\d{3}(?:\.\d+)?(?:\s+Art\.?\s*\d+[a-z]?)?
          | BGE\s+\d{1,3}\s+[IVX]+[a-z]?\s+\d+(?:\s+E\.\s*\d+[a-z]?)?
          | Art\.?\s+\d+[a-z]?\s+(?:Abs\.?\s*\d+\s+)*(?:[A-Z][a-zA-ZäöüÄÖÜß0-9]*)
        )""",
        re.VERBOSE,
    )

has_citation_d = {}

for query_id, d in train_candidate_d.items():
    rerank_l = d['rerank']
    
    all_citation = []
    for cc, score in rerank_l:
        l = CITATION_RE.findall(cc['text'])
        for c in l:
            all_citation.append(c)

    has_citation_d[query_id] = set(all_citation)

count = 0

for query_id, citation_set in has_citation_d.items():
    if len(citation_set & train_qid_2_gold_citations[query_id]) > 0:
        count += 1

print(count)


for qid, gold_citation_set in train_qid_2_gold_citations.items():
    for c in gold_citation_set:
        if len(CITATION_RE.findall(c)) == 0:
            print(c)

    