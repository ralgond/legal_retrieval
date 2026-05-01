import pickle

with open("../data/ml3/raw_train_candidate.pkl", "rb") as inf:
    ret_d = {}
    l = pickle.load(inf)
    for query_id, hits1_strip_text, hits2_strip_text, hits3_strip_text in l:
        print(query_id, len(hits3_strip_text))