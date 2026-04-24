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
    prompt=f'''You are imitating the language of German court considerations 
(Entscheidungsgründe). Rewrite the following English legal query 
into German exactly as a judge would phrase it in the reasoning 
section of a decision. Generate 10 variants. Each variant must 
introduce at least one key noun not used in any previous variant.

<query>{query}</query>

Output as JSON array of 10 strings, no commentary.'''

    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8
    )

    variants_de = json.loads(completion.choices[0].message.content)
    for v in variants_de:
        query_id_l.append(query_id)
        query_l.append(v)
    

df = pd.DataFrame({'query_id':query_id_l, 'query':query_l})
df.to_csv("../data/valid_10variants_2.csv", index=False)