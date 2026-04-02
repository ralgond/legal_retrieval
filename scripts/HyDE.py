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
rewrite_l = ['' for _ in range(len(train_df))]
for idx, query in tqdm(enumerate(train_df['query'].tolist()), total=len(train_df)):
    prompt=f'''你是一名德国法律专家，精通德国民法、刑法及行政法体系。
用户提出了一个法律问题或描述了一段事实情况。请你根据这些信息，生成一段假设性的德国法院判决理由（Entscheidungsgründe），模拟法院在类似案件中可能写下的考量（Court Consideration）。
要求：

使用正式的德语法律文体写作
长度控制在 150–250 词
必须引用具体的法律条文（如 § 823 BGB、§ 242 BGB 等）
包含法律推理结构：事实认定 → 法律适用 → 结论
不要解释你在做什么，直接输出判决理由文本

用户输入：
{query}'''

    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )

    new_query = completion.choices[0].message.content
    print("query.wc:", word_count(query), "new_query.wc:", word_count(new_query))
    rewrite_l[idx] = new_query
    train_df['query2'] = rewrite_l
    train_df.to_csv("../data/valid_hyde_001.csv", index=False)