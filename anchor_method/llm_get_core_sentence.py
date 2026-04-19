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


court_consideration_df = pd.read_csv("../data/court_considerations.csv")
print("cc loaded.")

sampled_cc_df = court_consideration_df.sample(100)

citation_l = []
result_l = []

for idx, (citation, text) in tqdm(enumerate(zip(sampled_cc_df['citation'], sampled_cc_df['text'])), total=len(sampled_cc_df)):
    prompt=f'''你是一位精通德语/法语/意大利语的瑞士法律检索优化专家。
              
              ## 要求
              在下面这段法律文本中，哪些句子具有判决性质同时有法典引用？请直接抽取句子，不要添加或减少token。

              ## 输出为json格式
              ['sentence1', 'sentence2', ...]

              ## 原始问题
              {text}

              ## 抽取结果'''

    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )

    sentences = json.loads(completion.choices[0].message.content)
    for sentence in sentences:
        citation_l.append(citation)
        result_l.append(sentence)

df = pd.DataFrame({"citation":citation_l, "sentence":result_l})
df.to_csv("../data/anchor_method/core_sentence.csv", index=False)
