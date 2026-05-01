import re
import os
from openai import OpenAI
import pandas as pd
from tqdm import tqdm

def word_count(text):
    return len(re.split(r"\s+", text))

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key="sk-71d0d11bec274377b20a14c5a93f2f0c",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

train_df = pd.read_csv("../data/val.csv")
rewrite_l = []
query_id_l = []

PROMPT = '''You are a legal search query rewriting system for Swiss/German legal retrieval (BM25-based search).

Your task:
Given an English query, generate 10 diverse German search queries for legal document retrieval.

---

IMPORTANT REQUIREMENTS:

1. Translate the meaning into German legal language (not literal translation).
2. Each query MUST preserve the original legal intent.
3. Ensure high diversity across rewrites:
   - Use different verbs (e.g., kündigen, aufheben, beenden, widerrufen, lösen)
   - Use different syntactic forms:
     * nominalization (e.g., "Kündigung des Mietvertrags")
     * passive voice (e.g., "Der Mietvertrag wird beendet")
     * infinitive / noun phrase form
   - Use different legal phrasing styles:
     * formal legal terminology
     * colloquial legal search terms
     * institutional/legal drafting style

4. Each rewrite MUST:
   - Contain at least one key original concept (entity preservation)
   - Avoid introducing new legal concepts not implied in the original query
   - Be suitable as a BM25 search query

5. Diversity constraint:
   - Any two outputs should NOT share more than 70% of their meaningful tokens
   - Avoid simple synonym substitution only

6. Coverage goal:
   - Maximize lexical and structural coverage of the legal semantic space

---

OUTPUT FORMAT:
Return 10 queries only, numbered 1–10.

---

EXAMPLE INPUT:
"terminate rental contract early"

EXAMPLE OUTPUT STYLE:
1. ...
2. ...
...
10. ...'''


def parse_queries(text):
    lines = text.split("\n")
    queries = []
    for l in lines:
        if "." in l:
            q = l.split(".", 1)[1].strip()
            queries.append(q)
    return queries

def rewrite_query(query_en):
    resp = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": query_en}
        ],
        temperature=0.9  # 很重要：提高多样性
    )

    text = resp.choices[0].message.content
    return parse_queries(text)


valid_df = pd.read_csv("../data/val.csv")
qid_l = []
qct_l = []
qct_raw_l = []
for query_id, query_en in tqdm(zip(valid_df['query_id'], valid_df['query']), total=len(valid_df)):
    queries_de = rewrite_query(query_en)
    for qct in queries_de:
        qid_l.append(query_id)
        qct_l.append(qct)
        qct_raw_l.append(query_en)

df = pd.DataFrame({"query_id":qid_l, "query2": qct_l, "query_en":qct_raw_l})
df.to_csv("../data/valid_rewrite_003_10v.csv", index=False)

    