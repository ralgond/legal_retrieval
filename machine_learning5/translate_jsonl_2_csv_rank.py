import json
import pandas as pd


def filter_shadowed_article_citations(citations: list[str]) -> list[str]:
    """
    在一个 citation 列表里，移除被更具体的段落级 citation 所覆盖的条文级 citation，
    并保留原始顺序。
    
    规则：如果集合中存在 `Art. 11 Abs. 2 OR`，则移除 `Art. 11 OR`。
    """
    import re

    ART_RE = re.compile(
        r"^Art\.?\s*(\d+[a-z]?)\s*(Abs\.?\s*\d+)?\s*([A-Z]{2,})$",
        re.IGNORECASE,
    )

    # 第一遍：找出哪些 (art_num, law) 存在带 Abs. 的版本
    has_paragraph: set[tuple[str, str]] = set()
    for cit in citations:
        m = ART_RE.match(cit.strip())
        if m and m.group(2):  # 有 Abs. 部分
            has_paragraph.add((m.group(1).lower(), m.group(3).upper()))

    # 第二遍：按原始顺序过滤
    result = []
    for cit in citations:
        m = ART_RE.match(cit.strip())
        if m and not m.group(2):  # 是纯条文级（无 Abs.）
            key = (m.group(1).lower(), m.group(3).upper())
            if key in has_paragraph:  # 被更具体版本覆盖，跳过
                continue
        result.append(cit)

    return result

print("Loading data...")
court_consideration_df = pd.read_csv("../data/court_considerations.csv")
court_consideration_d = dict(zip(court_consideration_df['citation'].tolist(), court_consideration_df['text'].tolist()))

law_df = pd.read_csv("../data/laws_de.csv")
law_d = dict(zip(law_df['citation'].tolist(), law_df['text'].tolist()))


l = []

def parse(d):
    query_id = d['query_id']
    l = []
    for cc in d['cc_list']:
        for citation in cc['citations']:
            l.append((citation['citation_id'], citation['rank']))

    l.sort(key=lambda x: x[1])
    return query_id, [citation for citation,_ in l if citation in law_d or citation in court_consideration_d]
    

query_id_l = []
predicted_citations_l = []
for line in open("../data/ml5/predictions.jsonl"):
    query_id, l = parse(json.loads(line.strip()))
    query_id_l.append(query_id)
    l = filter_shadowed_article_citations(l)
    predicted_citations_l.append(';'.join(l[:25]))

df = pd.DataFrame({'query_id': query_id_l, "predicted_citations": predicted_citations_l})
df.to_csv("../data/ml5/citation_predictions.csv", index=False)
