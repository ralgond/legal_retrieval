import numpy as np

def compute(ranked_l_l : list[list[str]]):
    count = len(ranked_l_l)
    freq_d = {}
    rank_d = {}
    for l in ranked_l_l:
        for term in l:
            if term in freq_d:
                freq_d[term] += 1
            else:
                freq_d[term] = 1

    for term, freq in freq_d.items():
        if freq != count:
            raise ValueError(f'{term}, {freq} != {count}')


    for idx, l in enumerate(ranked_l_l):
        for rank, term in enumerate(l, start=1):
            if term in rank_d:
                rank_d[term].append(rank)
            else:
                rank_d[term] = [rank]
                
    # print(rank_d)
    
    term_socre_l = []
    for term, rank_l in rank_d.items():
        score = 0.
        for rank in rank_l:
            score += 1/(60.+rank)
        term_socre_l.append((term, score))

    return sorted(term_socre_l, key=lambda x: x[1], reverse=True)


def compute2(ranked_l_l: list[list[str]], k: int = 60, top_k: int = 100):
    from collections import defaultdict

    scores = defaultdict(float)

    for ranked_list in ranked_l_l:
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] += 1 / (k + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in ranked[:top_k]]

def compute2_with_score(ranked_l_l: list[list[str]], k: int = 60, top_k: int = 100):
    from collections import defaultdict

    scores = defaultdict(float)

    for ranked_list in ranked_l_l:
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] += 1 / (k + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [(doc, score) for doc, score in ranked[:top_k]]

if __name__ == "__main__":
    print(compute([['A', 'B', 'C'],['C','B','A']]))
        


    
        