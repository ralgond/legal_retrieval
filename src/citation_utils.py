import re
from typing import List, Tuple
import math
from collections import defaultdict
import citation_utils2

def extract_citations_from_text(text: str) -> list[str]:
    """Extract citations from any text (tool output or final answer)."""
    citations = []
    
    # SR pattern: SR followed by number (optionally with article)
    sr_matches = re.findall(
        r"SR\s*\d{3}(?:\.\d+)?(?:\s+Art\.?\s*\d+[a-z]?)?",
        text,
        re.IGNORECASE
    )
    citations.extend(sr_matches)
    
    # BGE pattern: BGE volume section page
    bge_matches = re.findall(
        r"BGE\s+\d{1,3}\s+[IVX]+[a-z]?\s+\d+(?:\s+E\.\s*\d+[a-z]?)?",
        text,
        re.IGNORECASE
    )
    citations.extend(bge_matches)
    
    # Art. pattern: Art. X LAW (e.g., Art. 1 ZGB, Art. 41 OR)
    art_matches = re.findall(
        r"Art\.?\s+\d+[a-z]?\s+(?:Abs\.?\s*\d+\s+)?[A-Z]{2,}",
        text,
        re.IGNORECASE
    )
    citations.extend(art_matches)
    
    return list(set(citations))


def extract_citations_from_text_with_span(text: str) -> List[Tuple[str, int, int]]:
    """
    Extract citations with span information.

    Returns:
        List of (citation_text, start_idx, end_idx)
    """
    results = []
    seen = set()  # 用于去重 (text, start, end)

    patterns = [
        # SR pattern
        r"SR\s*\d{3}(?:\.\d+)?(?:\s+Art\.?\s*\d+[a-z]?)?",
        
        # BGE pattern
        r"BGE\s+\d{1,3}\s+[IVX]+[a-z]?\s+\d+(?:\s+E\.\s*\d+[a-z]?)?",
        
        # Art pattern
        r"Art\.?\s+\d+[a-z]?\s+(?:Abs\.?\s*\d+\s+)?[A-Z]{2,}"
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            span = match.span()
            citation = match.group().strip()

            key = (citation, span[0], span[1])
            if key not in seen:
                seen.add(key)
                results.append((citation, span[0], span[1]))

    return results

def extract_citations_and_type_from_text(text: str) -> list[str]:
    """Extract citations from any text (tool output or final answer)."""
    citations = []
    
    # SR pattern: SR followed by number (optionally with article)
    sr_matches = re.findall(
        r"SR\s*\d{3}(?:\.\d+)?(?:\s+Art\.?\s*\d+[a-z]?)?",
        text,
        re.IGNORECASE
    )
    citations.extend([(m,'sr') for m in sr_matches])
    
    # BGE pattern: BGE volume section page
    bge_matches = re.findall(
        r"BGE\s+\d{1,3}\s+[IVX]+[a-z]?\s+\d+(?:\s+E\.\s*\d+[a-z]?)?",
        text,
        re.IGNORECASE
    )
    citations.extend([(m,'bge') for m in bge_matches])
    
    # Art. pattern: Art. X LAW (e.g., Art. 1 ZGB, Art. 41 OR)
    art_matches = re.findall(
        r"Art\.?\s+\d+[a-z]?\s+(?:Abs\.?\s*\d+\s+)?[A-Z]{2,}",
        text,
        re.IGNORECASE
    )
    citations.extend([(m, 'art') for m in art_matches])
    
    return list(set(citations))

def normalized_sr(text):
    citations = extract_citations_and_type_from_text(text)
    for c,type in citations:
        if type == 'sr':
            parsed = citation_utils2.parse_citation(c)
            c2 = citation_utils2.normalize(parsed)
            text = text.replace(c,c2)
    return text
            
def split_sentences(text: str) -> list[str]:
    """
    正确断句：先将citation中的句号保护起来，断句后再还原。
    """
    citations = extract_citations_from_text(text)
    
    # 1. 用占位符替换所有citation，避免其中的句号干扰断句
    placeholder_map = {}
    protected = text
    for i, citation in enumerate(citations):
        placeholder = f"__CITATION_{i}__"
        placeholder_map[placeholder] = citation
        # 替换文本中所有该citation的出现
        protected = protected.replace(citation, placeholder)
    
    # 2. 在保护后的文本上断句
    # 匹配句末标点：.  !  ? 后跟空白或结尾
    raw_sentences = re.split(r'(?<=[.!?])\s+', protected.strip())
    
    # 3. 还原每个句子中的citation占位符
    sentences = []
    for s in raw_sentences:
        for placeholder, original in placeholder_map.items():
            s = s.replace(placeholder, original)
        s = s.strip()
        if s:
            sentences.append(s)
    
    return sentences

def compute_citation_score_with_sentence_pos(candidates_with_scores, decay="reciprocal"):
    """
    candidates_with_scores: [(consideration_text, reranker_score), ...]
    返回: {law_citation: aggregated_score}
    """
    law_scores = {}
    
    decay_fn = {
        "reciprocal": lambda p: 1 / (p + 1),
        "log":        lambda p: 1 / math.log(p + 2),
        "exp":        lambda p: math.exp(-0.3 * p),
    }[decay]
    
    for doc, reranker_score in candidates_with_scores:
        text = normalized_sr(doc['text'])
        if text != doc['text']:
            pass 
        sentences = split_sentences(text)
        cited_laws = extract_citations_from_text(text)  # 你的citation抽取函数, 这里就没有sr开头的art了
        
        # 建立每个法条首次出现的句子位置
        law_first_pos = {}
        for i, sent in enumerate(sentences):
            for law in cited_laws:
                if law in sent and law not in law_first_pos:
                    law_first_pos[law] = i
        
        for law, pos in law_first_pos.items():
            position_weight = decay_fn(pos)
            if law not in law_scores:
                law_scores[law] = reranker_score * position_weight
            # elif law_scores[law] < reranker_score * position_weight:
            #     law_scores[law] = reranker_score * position_weight
            else:
                law_scores[law] += reranker_score * position_weight
    
    return sorted(law_scores.items(), key=lambda x: -x[1])

import court_consideration_utils

def compute_citation_score_with_court_consideration_sector_pos(candidates_with_scores):

    law_scores = {}
    
    for doc, reranker_score in candidates_with_scores:
        text = normalized_sr(doc['text'])
        cited_laws = extract_citations_from_text(text)  # 你的citation抽取函数, 这里就没有sr开头的art了

        sectors = court_consideration_utils.split_court_document(text)

        law_max_weight_pos = {}
        for i, s in enumerate(sectors):
            for law in cited_laws:
                if law in s.text:
                    if law not in law_max_weight_pos:
                        law_max_weight_pos[law] = s
                    elif law_max_weight_pos[law].weight < s.wight:
                        law_max_weight_pos[law] = s
                    
        for law, s in law_max_weight_pos.items():
            position_weight = s.weight
            if law not in law_scores:
                law_scores[law] = reranker_score * position_weight
            # elif law_scores[law] < reranker_score * position_weight:
            else:
                law_scores[law] += reranker_score * position_weight

    return sorted(law_scores.items(), key=lambda x: -x[1])


def parse_cc_output_citations_and_sentences(text):
    text = normalized_sr(text)
    sentences = split_sentences(text)
    cited_laws = extract_citations_from_text(text)  # 你的citation抽取函数, 这里就没有sr开头的art了

    # 建立每个法条首次出现的句子位置
    law_first_pos = {}
    for i, sent in enumerate(sentences):
        for law in cited_laws:
            if law in sent and law not in law_first_pos:
                law_first_pos[law] = i
                
    return {'sentences':sentences, 'citations':[(law,idx) for law,idx in law_first_pos.items()]}

def parse_cc_output_citations_and_sentences_2(text):
    text = normalized_sr(text)
    sentences = split_sentences(text)
    cited_laws = extract_citations_from_text(text)  # 你的citation抽取函数, 这里就没有sr开头的art了

    _l = []
    for i, sent in enumerate(sentences):
        for law in cited_laws:
            if law in sent:
                _l.append((law, i))
                
    return {'sentences':sentences, 'citations':_l}

def build_evidence(sentences, citation_idx, window_size=3):
    """
    sentences: List[str]
    citation_idx: citation first occurs sentence index
    """
    half = window_size // 2
    left = max(0, citation_idx - half)
    right = min(len(sentences), citation_idx + half + 1)
    return " ".join(sentences[left:right])

def remove_citation_from_text(text):
    citations = extract_citations_from_text(text)
    for c in citations:
        text = text.replace(c, "")
    return text

def map_citation_2_pcitation(citation):
    return "CITE_START_" + citation.replace(" ", "_")+"_CITE_END"

def map_pcitation_2_citation(pcitation):
    assert pcitation.startswith("CITE_START_")
    return pcitation.replace("CITE_START_", "").replace("_CITE_END", "").replace("_", " ")

def extract_pcitations_from_text_with_span(text: str) -> List[Tuple[str, int, int]]:
    """
    Extract citations with span information.

    Returns:
        List of (citation_text, start_idx, end_idx)
    """
    results = []
    seen = set()  # 用于去重 (text, start, end)

    patterns = [
        r"CITE_START_((?:(?!CITE_START_).)*?)_CITE_END"
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            span = match.span()
            citation = match.group().strip()

            key = (citation, span[0], span[1])
            if key not in seen:
                seen.add(key)
                results.append((citation, span[0], span[1]))

    return results