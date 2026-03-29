
import re
import os
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import json

def word_count(text):
    return len(re.split(r"\s+", text))

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key="sk-71d0d11bec274377b20a14c5a93f2f0c",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

test_df = pd.read_csv("../data/test.csv")
rewrite_l = ['' for _ in range(len(test_df))]
all_data = []
for idx, (query_id, query) in tqdm(enumerate(zip(test_df['query_id'], test_df['query_en'].tolist())), total=len(test_df)):
    prompt=f'''你是一位精通德语的法律检索优化专家。请将以下用户的问题，改写成一个精炼的、适合在法律判决书数据库中进行关键词搜索的查询语句。
              ## 要求：
              1、保留核心法律关键词（如罪名、法律行为、涉案主体）。
              2、去除语气词，疑问词和口语化表达。
              3、如果可能，补充该行为的法律专业术语（例如将“偷拿”改为“盗窃”或“职务侵占”）。
              4、改写前的查询语句是英语，改写后的查询语句是德语。
              5、原始问题的前半部分是事实陈述(context)，后半段是问题，问题有一个或多个；你应该具备识别这个模式的能力。

              ## 输出格式要求（json格式）：
              [
              {{'context':..., 'question':'question 1...'}}
              {{'context':..., 'question':'question 2...'}}
              ...
              {{'context':..., 'question':'question N...'}}
              ]

              ## 原始问题：{query}
              
              ## 输出：'''

    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )

    new_query = completion.choices[0].message.content
    l = json.loads(new_query)
    for d in l:
        d['query_id'] = query_id
        d['query'] = d['context']+"\n\n"+d['question']
        del d['context']
        del d['question']
        all_data.append(d)
    # print(all_data)
    # break
query_id_l = []
query_l = []

for d in all_data:
    query_id_l.append(d['query_id'])
    query_l.append(d['query'])

result_df = pd.DataFrame({'query_id': query_id_l, "query":query_l})
result_df.to_csv("../data/test_rewrite_split_question_001.csv", index=False)
    

