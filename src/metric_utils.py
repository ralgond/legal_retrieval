import numpy as np

def cal_recall(all_hits_l, gold_citations_l, truncate_method=None):
    recalls = []
    for all_hits, gold_citations in zip(all_hits_l, gold_citations_l):   
        all_citation = []
        if truncate_method is None:
            for hit in all_hits:
                all_citation.append(hit['citation'])
        else:
            __limit = truncate_method(all_hits)
            for hit in all_hits[:__limit]:
                all_citation.append(hit['citation'])

        hits = len(set(all_citation) & set(gold_citations))

        # print("gold_citations.len:", len(gold_citations), ", hits.len:", hits)
        recall = hits / len(gold_citations)
        recalls.append(recall)
        
    mean_recall = np.mean(recalls)
    return mean_recall

def cal_precision(all_hits_l, gold_citations_l, truncate_method=None):
    precisions = []
    for all_hits, gold_citations in zip(all_hits_l, gold_citations_l):
        all_citation = []
        if truncate_method is None:
            for hit in all_hits:
                all_citation.append(hit['citation'])
        else:
            __limit = truncate_method(all_hits)
            for hit in all_hits[:__limit]:
                all_citation.append(hit['citation'])
        
        predicted = set(all_citation)
        hits = len(predicted & set(gold_citations))
        
        if len(predicted) == 0:
            precision = 0.0
        else:
            precision = hits / len(predicted)
        
        precisions.append(precision)
        
    mean_precision = np.mean(precisions)
    return mean_precision

def cal_f1(r, p):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0