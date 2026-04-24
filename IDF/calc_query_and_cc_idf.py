import pandas as pd

df = pd.read_csv("../data/court_considerations.csv")

N = len(df)

for text in df['text']:
    text.sps