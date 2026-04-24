'''
将citation映射成CITE_START_XXX._XXX_CITE_END
'''
import re
from typing import List, Tuple
import math
from collections import defaultdict

def p_extract_citations_from_text(text: str) -> list[str]:
    """Extract pcitations from any text (tool output or final answer)."""
    citations = []
    
    # SR pattern: SR followed by number (optionally with article)
    matches = re.findall(
        r"CITE_START_((?:(?!CITE_START_).)*?)_CITE_END",
        text,
        re.IGNORECASE
    )

    citations.extend(matches)
    
    return list(set(citations))

def p_split_sentences(text: str) -> list[str]:
    """
    正确断句：先将citation中的句号保护起来，断句后再还原。
    """
    citations = p_extract_citations_from_text(text)
    
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

def citation_2_p_citation(citation):
    return "CITE_START_" + citation.replace(" ", "_")+"_CITE_END"

def p_citation_2_citation(pcitation):
    assert pcitation.startswith("CITE_START_")
    return pcitation.replace("CITE_START_", "").replace("_CITE_END", "").replace("_", " ")


def test():

    pass

if __name__ == "__main__":
    test()
