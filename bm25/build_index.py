import pandas as pd
import bm25s
import Stemmer  # optional: for stemming

# Create your corpus here
corpus = []
cc_df = pd.read_csv("../data/court_considerations.csv")
print("cc_df loaded.")
court_doc = [{'citation':citation, 'text':text} for citation,text in zip(cc_df['citation'], cc_df['text'])]
for d in court_doc:
    corpus.append(d['text'])

# optional: create a stemmer
stemmer = Stemmer.Stemmer("german")

# Tokenize the corpus and only keep the ids (faster and saves memory)
corpus_tokens = bm25s.tokenize(corpus, stopwords="de", stemmer=stemmer)

# Create the BM25 model and index the corpus
retriever = bm25s.BM25()
retriever.index(corpus_tokens)

# You can save the arrays to a directory...
retriever.save("../data/bm25/cc")


# Query the corpus
query = "does the fish purr like a cat?"
query_tokens = bm25s.tokenize(query, stemmer=stemmer)

# Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k).
# To return docs instead of IDs, set the `corpus=corpus` parameter.
results, scores = retriever.retrieve(query_tokens, k=2)

for i in range(results.shape[1]):
    doc, score = results[0, i], scores[0, i]
    print(f"Rank {i+1} (score: {score:.2f}): {doc}")


# # You can save the corpus along with the model
# retriever.save("animal_index_bm25", corpus=corpus)

# # ...and load them when you need them
# import bm25s
# reloaded_retriever = bm25s.BM25.load("animal_index_bm25", load_corpus=True)
# # set load_corpus=False if you don't need the corpus