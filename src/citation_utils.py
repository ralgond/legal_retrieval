import re
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
            print("====>sr") 
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
            elif law_scores[law] < reranker_score * position_weight:
                law_scores[law] = reranker_score * position_weight

    # laws = [law for law,_ in law_scores.items()]

    # dedup_laws = set(citation_utils2.deduplicate(laws))

    # for law in laws:
    #     if law not in dedup_laws:
    #         del law_scores[law]
    
    return sorted(law_scores.items(), key=lambda x: -x[1])