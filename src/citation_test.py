import pandas as pd

import citation_utils
import citation_p_utils

df = pd.read_csv('../data/court_considerations.csv')

text = df['text'].to_list()[11]

print(text)

print(citation_utils.extract_citations_from_text(text))

del df

df = pd.read_csv('../data/court_considerations_maped.csv')

text = df['text'].to_list()[11]

print(text)

print(citation_p_utils.p_extract_citations_from_text(text))
