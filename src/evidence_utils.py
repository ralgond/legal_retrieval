import citation_utils
import reranker_utils

def build_evidence(sentences, citation_idx, left_sentence_cnt, right_sentence_cnt):
    left = max(0, citation_idx - left_sentence_cnt)
    right = min(len(sentences), citation_idx + right_sentence_cnt+1)
    return " ".join(sentences[left:right])

def extract_evidences(reranker, query, cc_text):
    
    parsed_cc = citation_utils.parse_cc_output_citations_and_sentences(cc_text) # parsed_cc 格式为 {'sentences':sentences, 'citations':citations}

    result_l = []
    for citation in parsed_cc['citations']:
        # 寻找包含citation的句子
        for idx, sentence in enumerate(parsed_cc['sentences']):
            if citation in sentence:
                #result_l.append([citation, build_evidence(parsed_cc['sentences'], idx, 1, 1)])
                result_l.append([citation, build_evidence(parsed_cc['sentences'], idx, 3, 3)])
                # result_l.append([citation, build_evidence(parsed_cc['sentences'], idx, 2, 1)])
                break # 只处理第一个citation

    reranked_result = reranker_utils.rerank_by_batch_chunked_simple(reranker, query, [{'citation':c, "text":evidence_text} for c, evidence_text in result_l])
    result_l = []
    for result,score in reranked_result:
        result['score'] = score
        result_l.append(result)

    citation_evidence_d = {}
    for result in result_l:
        citation, evidence_text, score = result['citation'], result['text'], result['score']
        # print("====>", score)
        if citation not in citation_evidence_d:
            citation_evidence_d[citation] = {'citation':citation, 'text': evidence_text, 'score': score}
        elif citation_evidence_d[citation]['score'] < score:
            citation_evidence_d[citation]['score'] = score

    return citation_evidence_d