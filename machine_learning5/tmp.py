def gen_test_jsonl():
    test_cc_dense_score_d = {}
    test_cc_sparse_score_d = {}
    test_cc_rerank_score_d = {}

    for test_query_id, d in test_candidate_d.items():
        dense_l = d['dense']
        for cc, dense_score in dense_l:
            test_cc_dense_score_d[cc['citation']] = dense_score
        sparse_l = d['sparse']
        for cc, sparse_score in sparse_l:
            test_cc_sparse_score_d[cc['citation']] = sparse_score
        rerank_l = d['rerank']
        for cc, rerank_score in rerank_l:
            test_cc_rerank_score_d[cc['citation']] = rerank_score

    l = []
    for test_query_id, d_raw in test_candidate_d.items():
        d = {}
        d['query_id'] = test_query_id
        # d['gold_citations'] = test_query_id_2_gold_citation_d[test_query_id]
        cc_list = []
        rerank_list = d_raw['rerank']
        for cc,_ in rerank_list:
            d2 = {}
            d2['cc_id'] = cc['citation']
            d2['text'] = cc['text']
            d2['dense_score'] = float(test_cc_dense_score_d.get(cc['citation'], 0.))
            d2['sparse_score'] = float(test_cc_sparse_score_d.get(cc['citation'], 0.))
            d2['rerank_score'] = float(test_cc_rerank_score_d.get(cc['citation'], 0.))
            cc_list.append(d2)
        d['cc_list'] = cc_list
        l.append(d)
    return l

l = gen_test_jsonl()
with open(f"{OUTPUT}/test.jsonl", "w+", encoding="utf-8") as of:
    for j in l:
        # print(j)
        of.write(json.dumps(j, ensure_ascii=False) + '\n')