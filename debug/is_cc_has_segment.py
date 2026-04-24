import pandas as pd

cc_df = pd.read_csv("../data/court_considerations.csv")
for text in cc_df['text'].tolist()[:10]:
    print(text)
    print("="*80)