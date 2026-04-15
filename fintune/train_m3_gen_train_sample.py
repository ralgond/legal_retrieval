import pandas as pd
import json

print("Loading models...")
court_consideration_df = pd.read_csv("../data/court_considerations.csv")

text_l = []
for text in court_consideration_df['text'].tolist():
    text_l.append(text)

# with open("../ft_data/bge-m3_unsupervised_data.jsonl", "w+", encoding='utf-8') as of:
#     for text in text_l:
#         of.write(json.dumps([text], ensure_ascii=False)+"\n")
with open("../ft_data/bge-m3_unsupervised_data.txt", "w+", encoding='utf-8') as of:
    for text in text_l:
        of.write(text+"\n")