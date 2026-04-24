import pickle

def assign_text_to_hits(hits, consideration_d):
    ret = []
    for hit,score in hits:
        citation = hit['citation']
        ret.append(({'citation':citation, 'text':consideration_d[citation]}, score))
    return ret

def read_candidate(path, consideration_d):
    with open(path, "rb") as inf:
        ret_d = {}
        l = pickle.load(inf)
        for query_id, hits1_strip_text, hits2_strip_text, hits3_strip_text in l:

            ret_d[query_id] = {
                'dense':assign_text_to_hits(hits1_strip_text,consideration_d),
                'sparse':assign_text_to_hits(hits2_strip_text,consideration_d), 
                'rerank':assign_text_to_hits(hits3_strip_text, consideration_d),
                }
        return ret_d
