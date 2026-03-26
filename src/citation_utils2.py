import re
from collections import defaultdict

# -------------------------------
# 1️⃣ Regex patterns
# -------------------------------

# Art 在前 (最常见)
ART_SR_PATTERN = re.compile(r"""
    Art\.?\s*(?P<art>\d+[a-zA-Z]?)           # Art 条号
    (?:\s*Abs\.?\s*(?P<abs>\d+))?           # Abs 款
    (?:\s*lit\.?\s*(?P<lit>[a-z]))?         # lit 字母
    (?:\s*Ziff\.?\s*(?P<ziff>\d+))?         # Ziff 编号
    \s*(?P<law>[A-Z]{2,}|\d{3}(?:\.\d+)?)  # OR/ZGB 或 SR 编号
""", re.VERBOSE | re.IGNORECASE)

# SR 在前 (官方 / 法典格式)
SR_ART_PATTERN = re.compile(r"""
    SR\s*(?P<law>\d{3}(?:\.\d+)?)           # SR 编号
    (?:\s+Art\.?\s*(?P<art>\d+[a-zA-Z]?))? # Art 条号可选
    (?:\s*Abs\.?\s*(?P<abs>\d+))?
    (?:\s*lit\.?\s*(?P<lit>[a-z]))?
    (?:\s*Ziff\.?\s*(?P<ziff>\d+))?
""", re.VERBOSE | re.IGNORECASE)

# BGE 判例
BGE_PATTERN = re.compile(r"""
    BGE\s*\d+\s*[I]{1,3}\s*\d+              # 卷+分部+页
    (?:\s*E\.?\s*\d+(?:\.\d+)*)?           # 段落 E. 1.2.3
""", re.VERBOSE | re.IGNORECASE)

# docket style 判例
DOCKET_PATTERN = re.compile(r"\b\d+[A-Z]_?\d+/\d+\s+E\.?\s*\d+(?:\.\d+)*\b", re.IGNORECASE)

# -------------------------------
# 2️⃣ Parser
# -------------------------------
def parse_citation(text):
    text = text.strip()
    
    # 先匹配 Art 在前
    m = ART_SR_PATTERN.search(text)
    if m:
        return {
            "type": "law",
            "art": m.group("art"),
            "abs": m.group("abs"),
            "lit": m.group("lit"),
            "ziff": m.group("ziff"),
            "law": m.group("law"),
            "raw": text
        }
    
    # 再匹配 SR 在前
    m = SR_ART_PATTERN.search(text)
    if m:
        return {
            "type": "law",
            "art": m.group("art"),
            "abs": m.group("abs"),
            "lit": m.group("lit"),
            "ziff": m.group("ziff"),
            "law": m.group("law"),
            "raw": text
        }
    
    # BGE 判例
    if BGE_PATTERN.match(text):
        return {"type": "bge", "normalized": text, "raw": text}
    
    # Docket
    if DOCKET_PATTERN.match(text):
        return {"type": "docket", "normalized": text, "raw": text}
    
    return None

# -------------------------------
# 3️⃣ Normalization
# -------------------------------
def normalize(parsed):
    if parsed["type"] == "law":
        parts = [f"Art. {parsed['art']}"] if parsed.get("art") else []
        if parsed.get("abs"):
            parts.append(f"Abs. {parsed['abs']}")
        if parsed.get("lit"):
            parts.append(f"lit. {parsed['lit']}")
        if parsed.get("ziff"):
            parts.append(f"Ziff. {parsed['ziff']}")
        parts.append(parsed["law"])
        return " ".join(parts)
    else:
        return parsed["normalized"]

# -------------------------------
# 4️⃣ Deduplicate + 颗粒度控制
# -------------------------------
def is_sr(law):
    """判断是否为 SR 编号"""
    return law.replace(".", "").isdigit()

def granularity_score(parsed):
    """粒度评分：越详细（Abs/lit/Ziff）分数越高"""
    score = 0
    if parsed.get("abs"):
        score += 2
    if parsed.get("lit"):
        score += 1
    if parsed.get("ziff"):
        score += 1
    return score

def deduplicate(citations):
    parsed_list = []
    
    # parse + normalize
    for c in citations:
        parsed = parse_citation(c)
        if not parsed:
            continue
        parsed["normalized"] = normalize(parsed)
        parsed_list.append(parsed)
    
    # 分组：按 Art+Abs+Lit+Ziff
    groups = defaultdict(list)
    for p in parsed_list:
        if p["type"] == "law":
            key = (p["art"], p.get("abs"), p.get("lit"), p.get("ziff"))
        else:
            key = (p["type"], p["normalized"])
        groups[key].append(p)
    
    results = []
    for key, group in groups.items():
        if group[0]["type"] != "law":
            results.extend({p["normalized"] for p in group})
            continue
        
        # 优先 SR
        sr_candidates = [p for p in group if is_sr(p["law"])]
        if sr_candidates:
            best = max(sr_candidates, key=granularity_score)
        else:
            best = max(group, key=granularity_score)
        results.append(best["normalized"])
    
    # -------------------------------
    # 颗粒度控制：同 Art，保留最高粒度
    # -------------------------------
    art_dict = defaultdict(list)
    for p in results:
        if isinstance(p, str) and p.startswith("Art"):
            art_num = re.match(r"Art\.?\s*(\d+)", p).group(1)
            art_dict[art_num].append(p)
    
    final_results = []
    for art_num, lst in art_dict.items():
        if lst:
            # 选择粒度最高的
            lst_sorted = sorted(lst, key=lambda x: (
                2 if "Abs." in x else 0,
                1 if "lit." in x else 0,
                1 if "Ziff." in x else 0
            ), reverse=True)
            final_results.append(lst_sorted[0])
    
    # 添加非 Art 类型（BGE / docket）
    for p in results:
        if not (isinstance(p, str) and p.startswith("Art")):
            final_results.append(p)
    
    return sorted(set(final_results))

def delete_citation(citation_score_l):
    d = {}
    l = []
    for citation,score in citation_score_l:
        d[citation] = score
        l.append(citation)
    s2 = set(deduplicate(l))
    for citation in l:
        if citation not in s2:
            del d[citation]
    return sorted([(c,s) for c,s in d.items()], key=lambda x:x[1], reverse=True)


def dedup_with_score(pairs, parse_fn, normalize_fn):
    seen = {}
    result = []
    
    for citation, score in pairs:
        parsed = parse_fn(citation)
        if not parsed:
            continue
        
        norm = normalize_fn(parsed)
        
        if parsed["type"] != "law":
            key = ("case", norm)
            if key not in seen:
                seen[key] = (norm, score)
                result.append((norm, score))
            continue
        
        key = (parsed["law"], parsed["art"])
        g = granularity_score(parsed)
        
        if key not in seen:
            seen[key] = (norm, score, g)
            result.append((norm, score))
        else:
            # ⭐ 如果新的是更细粒度 → 替换
            _, _, old_g = seen[key]
            if g > old_g:
                seen[key] = (norm, score, g)
                
                # 替换 result 中旧值
                for i, (c, s) in enumerate(result):
                    if c.startswith(f"Art. {parsed['art']}"):
                        result[i] = (norm, score)
                        break
    
    return result

# -------------------------------
# 5️⃣ 测试示例
# -------------------------------
if __name__ == "__main__":
    citations = [
        "Art. 47 OR",
        "Art. 47 131.211",
        "Art. 47 Abs. 2 OR",
        "Art. 47 Abs. 2 131.211",
        "Art. 47 Abs. 2 lit. a 131.211",
        "SR 220 Art. 47",
        "SR 131.211 Art. 47 Abs. 2",
        "BGE 145 II 32 E. 3.1",
        "5A_800/2019 E 2.1",
        "Art. 48 131.211",
        "Art. 48 Abs. 1 131.211"
    ]
    
    deduped = deduplicate(citations)
    print(deduped)


    pairs = [
    ("Art. 47 OR", 0.95),
    ("Art. 47 Abs. 2 OR", 0.90),
    ("Art. 48 OR", 0.85),
    ("Art. 48 Abs. 1 OR", 0.80),
    ]

    print(dedup_with_score(pairs, parse_citation, normalize))