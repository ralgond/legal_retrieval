import re
import os
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import json

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key="sk-71d0d11bec274377b20a14c5a93f2f0c",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

valid_df = pd.read_csv("../data/val.csv")

query_id_l = []
query_l = []

for query_id, query in tqdm(zip(valid_df['query_id'], valid_df['query']), total=len(valid_df)):
    prompt=f'''You are an expert in German law, legal translation, and German judicial language.

Your task: Given a legal search query in English, generate 10 German query variants 
that are lexically close to how "court considerations" (Entscheidungsgründe / 
Erwägungen des Gerichts) are actually written in German court decisions.

The goal is to maximize BM25 recall over a corpus of German court decisions 
by producing variants that match the surface forms, terminology, and phrasing 
patterns found in authentic German judicial writing.

Apply the following 10 strategies — one per variant:

1. **Direct legal translation**: Translate the core legal concept using standard 
   German legal terminology (e.g., "breach of duty" → "Pflichtverletzung").

2. **Statutory reference form**: Use the form courts cite when applying a norm 
   (e.g., "gemäß § 242 BGB", "im Sinne des § 823 Abs. 1 BGB").

3. **Nominalization variant**: Convert verbal phrases to noun-heavy German legal 
   style (e.g., "failed to disclose" → "Verletzung der Offenbarungspflicht").

4. **Latin ↔ German equivalent**: Use the Latin legal term if commonly cited in 
   German courts, or vice versa (e.g., "good faith" → "Treu und Glauben" or 
   "bona fides").

5. **Court-tier specific phrasing**: Use terminology characteristic of a specific 
   court level — BGH (e.g., "revisionsrechtlich"), OLG, or BVerfG 
   (e.g., "verfassungsrechtlich geboten").

6. **Procedural action form**: Phrase as a procedural act or judicial finding 
   (e.g., "the court held that" → "das Gericht hat festgestellt, dass" / 
   "es ist davon auszugehen, dass").

7. **Synonymous legal concept**: Use a related but distinct German legal concept 
   that courts treat similarly (e.g., "negligence" → "Fahrlässigkeit" or 
   "Sorgfaltspflichtverletzung" or "Verkehrspflichtverletzung").

8. **Compound noun expansion**: Exploit German compounding to build domain-specific 
   terms (e.g., "liability" → "Schadensersatzpflicht", "Haftungsgrundlage", 
   "Haftungsmaßstab").

9. **Passive/impersonal judicial voice**: Rephrase in the passive or impersonal 
   constructions typical of German court reasoning 
   (e.g., "ist zu berücksichtigen", "bleibt außer Betracht", "war zu prüfen, ob").

10. **Abbreviated ↔ full form + common collocations**: Expand abbreviations or add 
    collocations courts use around the term 
    (e.g., "GmbH-Geschäftsführer", "im Rahmen der Abwägung", 
    "unter Berücksichtigung der Umstände des Einzelfalls").

Input:
<query>{query}</query>

Output strictly as JSON, no commentary, no markdown:
{{
  "original_en": "...",
  "variants_de": [
    {{"strategy": 1, "label": "Direct legal translation",          "query": "..."}},
    {{"strategy": 2, "label": "Statutory reference form",          "query": "..."}},
    {{"strategy": 3, "label": "Nominalization variant",            "query": "..."}},
    {{"strategy": 4, "label": "Latin ↔ German equivalent",        "query": "..."}},
    {{"strategy": 5, "label": "Court-tier specific phrasing",      "query": "..."}},
    {{"strategy": 6, "label": "Procedural action form",            "query": "..."}},
    {{"strategy": 7, "label": "Synonymous legal concept",          "query": "..."}},
    {{"strategy": 8, "label": "Compound noun expansion",           "query": "..."}},
    {{"strategy": 9, "label": "Passive/impersonal judicial voice", "query": "..."}},
    {{"strategy": 10,"label": "Abbreviated form + collocations",   "query": "..."}}
  ]
}}'''
    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )

    d = json.loads(completion.choices[0].message.content)
    variants_de = d['variants_de']
    for v in variants_de:
        query_id_l.append(query_id)
        query_l.append(v['query'])
    

df = pd.DataFrame({'query_id':query_id_l, 'query':query_l})
df.to_csv("../data/valid_10variants.csv", index=False)