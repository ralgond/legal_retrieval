
def merge_hits_with_score_l_by_max(l1, l2):
    d = {}
    d_score = {}
    for hit,score in l1:
        citation = hit['citation']
        d[citation] = hit
        if citation not in d_score:
            d_score[citation] = score
        elif d_score[citation] < score:
            d_score[citation] = score
    for hit,score in l2:
        citation = hit['citation']
        d[citation] = hit
        if citation not in d_score:
            d_score[citation] = score
        elif d_score[citation] < score:
            d_score[citation] = score

    l = [(d[citation], score) for citation, score in d_score.items()]
    return sorted(l, key=lambda x: x[1], reverse=True)

def merge_hits_with_score_l_by_weighted_add(l1, l2, w1, w2):
    d = {}
    d_score = {}
    for hit,score in l1:
        citation = hit['citation']
        d[citation] = hit
        if citation not in d_score:
            d_score[citation] = score * w1
            
    for hit,score in l2:
        citation = hit['citation']
        d[citation] = hit
        if citation not in d_score:
            d_score[citation] = score * w2
        else:
            d_score[citation] += score * w2

    l = [(d[citation], score) for citation, score in d_score.items()]
    return sorted(l, key=lambda x: x[1], reverse=True)