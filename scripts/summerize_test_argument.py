

import os
from openai import OpenAI
import pandas as pd
from tqdm import tqdm

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key="sk-71d0d11bec274377b20a14c5a93f2f0c",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

query_id_l = []
rewrite_l = []

test_df = pd.read_csv("./data/test.csv")
for query_id, query_en in tqdm(zip(test_df['query_id'].tolist(), test_df['query_en'].tolist()), total=len(test_df)):
    prompt=f'''你是一位精通英语和德语的法律检索优化专家。请将以下用户的问题，改写成一个精炼的、适合在法律判决书数据库中进行关键词搜索的查询语句。
              要求：
              1、保留核心法律关键词（如罪名、法律行为、涉案主体）。
              2、去除语气词，疑问词和口语化表达。
              3、如果可能，补充该行为的法律专业术语（例如将“偷拿”改为“盗窃”或“职务侵占”）。
              4、改写前的查询语句是英语，改写后的查询语句是德语。

              原始问题：{query_en}
              改写后的查询语句：'''

    for _ in range(4):
        completion = client.chat.completions.create(
            # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,
        )
        query_id_l.append(query_id)
        rewrite_l.append(completion.choices[0].message.content)

new_test_df = pd.DataFrame({"query_id":query_id_l, 'query':rewrite_l})
new_test_df.to_csv("data/test_rewrite_002.csv", index=False)

