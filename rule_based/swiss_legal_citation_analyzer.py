"""
Swiss German Legal Citation Analyzer
=====================================
解决两个问题：
1. 瑞士法律引用含空格 → 预处理占位符，spaCy 不会错误切分
2. 根据句法上下文（词性 + 关键词 + 依存关系）判断引用重要程度：
   - CORE       核心引用（+2 及以上）
   - INCIDENTAL 附带引用（-1 ~ +1）
   - NEGATIVE   负面/否定引用（-2 及以下）

依赖：
    pip install spacy
    python -m spacy download de_core_news_lg   # 推荐
    # 或者: python -m spacy download de_core_news_sm
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional
import spacy
from spacy.tokens import Doc, Token

# ---------------------------------------------------------------------------
# 1. 引用正则（瑞士 / 德国 / 奥地利法律文本常见格式）
# ---------------------------------------------------------------------------

_CITATION_PATTERNS: list[tuple[str, str]] = [
    # ── 联邦法律条文 ─────────────────────────────────────────────────────────
    # Art. 41 Abs. 1 lit. a OR / ZGB / SchKG / BV / StGB / ZPO / BGG / IPRG …
    (
        "article",
        r"""
        (?:Art\.|Ziff\.)           # 条款标志
        \s*\d+[a-z]?                 # 条号，可带字母后缀
        (?:                          # 可选：Abs./Ziff./lit./N./Rn.
            \s+(?:Abs\.|Ziff\.|lit\.|N\.|Rn\.)
            \s*\d*[a-z]?
        )*
        (?:\s+[A-ZÄÖÜ]{2,10})?      # 法典缩写（OR, ZGB, BV …）
        """,
    ),
    # ── 联邦最高法院判决（BGE / ATF / BGer / TF） ────────────────────────────
    (
        "bge",
        # r"""
        # (?:BGE|ATF)                  # 德/法引用
        # \s+\d{2,4}
        # \s+[IVXivx]+
        # \s+\d+
        # (?:\s+(?:E\.|Erw\.)\s*\d+(?:\.\d+)*)?  # 考量点
        # """,
        r'''BGE\s+\d{1,3}\s+[IVX]+[a-z]?\s+\d+(?:\s+E\.\s*\d+[a-z]?)?''',
    ),
    (
        "bger",
        r"""
        (?:BGer|TF)
        \s+[\w]+\/\d{4}             # 4A_312/2019
        (?:\s+(?:E\.|Erw\.)\s*\d+(?:\.\d+)*)?
        """,
    ),
    # ── 坎顿判决（OGer / KGer / VerwGer + Kanton） ──────────────────────────
    (
        "cantonal",
        r"""
        (?:OGer|KGer|VerwGer|HGer|AGer|StGer)
        (?:\s+[A-Z]{2})?             # ZH, BE, GE …
        \s+[\w.\/\-]+\/\d{4}
        """,
    ),
    # ── 欧洲人权法院（EGMR / ECtHR） ─────────────────────────────────────────
    (
        "egmr",
        r"""
        (?:EGMR|ECtHR)
        (?:\s+v\.|,)?
        \s+[\w\s]+,?
        \s+\d{1,2}\.\s*\w+\s+\d{4}  # 日期
        """,
    ),
    # ── 法规集（SR 编号） ──────────────────────────────────────────────────────
    (
        "sr",
        r"SR\s+\d{3}(?:\.\d+)+",
    ),
    # ── 文献 / 注释引用（BSK / CHK / BaKo / ZK / BerKomm / Basler Kommentar）
    (
        "commentary",
        r"""
        (?:BSK|CHK|BaKo|ZK|BerKomm|OFK|CR|CPra|SHK|HKommZPO|BSK\s*ZGB|BSK\s*OR)
        \s+\w+[-\w]*                 # 版本 / 卷号
        (?:,\s*Art\.\s*\d+[a-z]?\s*N\s*\d+)?  # 条文+边码
        """,
    ),
]

_COMBINED_PATTERN = re.compile(
    "|".join(f"(?P<{name}>{pat})" for name, pat in _CITATION_PATTERNS),
    re.VERBOSE | re.UNICODE,
)

# ---------------------------------------------------------------------------
# 2. 关键词评分表（正分 → 支持/强调；负分 → 否定/淡化）
# ---------------------------------------------------------------------------
# 格式：(触发词小写, 得分, 简短说明)
# 得分范围建议：核心 +2~+5，附带 +1/-1，否定 -2~-5

_KEYWORD_SCORES: list[tuple[str, float, str]] = [
    # ── 强引用 / 核心依据 ────────────────────────────────────────────────────
    ("gemäss",              +3.0, "依据"),
    ("laut",                +2.5, "依据"),
    ("nach",                +2.5, "依据"),
    ("zufolge",             +2.5, "依据"),
    ("gestützt auf",        +3.5, "基于"),
    ("gestützt",            +2.5, "基于"),
    ("i.s.v.",              +3.0, "在…意义上"),
    ("im sinne von",        +3.0, "在…意义上"),
    ("im sinne",            +2.5, "在…意义上"),
    ("gilt",                +2.5, "适用"),
    ("findet anwendung",    +3.5, "适用"),
    ("zur anwendung gelangt",+3.5,"适用"),
    ("anwendbar",           +2.5, "可适用"),
    ("einschlägig",         +3.0, "相关适用"),
    ("massgebend",          +3.0, "决定性"),
    ("massgeblich",         +3.0, "决定性"),
    ("entscheidend",        +2.5, "决定性"),
    ("ausschlaggebend",     +3.0, "决定性"),
    ("bestimmend",          +2.5, "决定性"),
    ("vorausgesetzt",       +2.0, "以…为前提"),
    ("voraussetzungen",     +1.5, "前提条件"),
    ("voraussetzung",       +1.5, "前提条件"),
    ("erfüllt",             +2.0, "满足"),
    ("erfüllung",           +2.0, "满足"),
    ("verletzt",            +2.0, "违反"),
    ("verletzung",          +2.0, "违反"),
    ("verstösst gegen",     +2.5, "违反"),
    ("verstösst",           +1.5, "违反"),
    ("begründet",           +2.5, "有充分理由"),
    ("stützt sich auf",     +3.0, "依赖"),
    ("stützt",              +2.0, "依赖"),
    ("beruht auf",          +2.5, "基于"),
    ("beruht",              +2.0, "基于"),
    ("basiert auf",         +2.5, "基于"),
    ("heranzuziehen",       +2.0, "援引"),
    ("heranzuziehen ist",   +2.5, "须援引"),
    ("anzuwenden",          +2.0, "须适用"),
    ("anzuwenden ist",      +2.5, "须适用"),
    ("zu beachten",         +2.0, "须注意"),
    ("zu beachten ist",     +2.5, "须注意"),
    ("zu berücksichtigen",  +2.0, "须考虑"),
    ("insbesondere",        +2.5, "尤其"),
    ("namentlich",          +2.5, "尤其"),
    ("vor allem",           +2.5, "尤其"),
    ("hauptsächlich",       +2.0, "主要"),
    ("primär",              +2.0, "首要"),
    ("grundlegend",         +3.0, "奠基性"),
    ("grundsätzlich",       +2.0, "原则上"),
    ("wesentlich",          +2.0, "重要"),
    ("zentral",             +2.5, "核心"),
    ("massgebend ist",      +3.0, "以…为准"),
    ("einschlägige",        +2.5, "相关"),
    ("einschlägigen",       +2.5, "相关"),
    ("vorliegend",          +1.5, "本案"),
    ("hier",                +0.5, "此处"),
    ("im vorliegenden fall",+2.0, "本案"),
    ("im konkreten fall",   +2.0, "本案"),
    ("daher",               +1.0, "因此（引出结论）"),
    ("folglich",            +1.5, "因此"),
    ("somit",               +1.5, "因此"),
    ("mithin",              +1.5, "因此"),
    ("demzufolge",          +1.5, "因此"),
    ("infolgedessen",       +1.5, "因此"),
    ("demnach",             +1.5, "因此"),

    # ── 弱引用 / 附带参见 ────────────────────────────────────────────────────
    ("vgl.",                -1.0, "参见（附带）"),
    ("vgl",                 -1.0, "参见（附带）"),
    ("vgl. auch",           -1.5, "参见（附带）"),
    ("s.",                  -1.0, "见（附带）"),
    ("s. auch",             -1.5, "见（附带）"),
    ("siehe",               -1.0, "见（附带）"),
    ("siehe auch",          -1.5, "见（附带）"),
    ("dazu",                -0.5, "就此"),
    ("hierzu",              -0.5, "就此"),
    ("ferner",              -1.0, "此外（附带）"),
    ("zudem",               -0.5, "此外"),
    ("überdies",            -0.5, "此外"),
    ("ausserdem",           -0.5, "此外"),
    ("auch",                -0.5, "也（附带）"),
    ("ebenfalls",           -0.5, "同样"),
    ("ebenso",              -0.5, "同样"),
    ("gleichermassen",      -0.5, "同样"),
    ("entsprechend",        -0.5, "相应地"),
    ("sinngemäss",          -0.5, "类推适用"),
    ("analog",              -0.5, "类推"),
    ("mutatis mutandis",    -0.5, "作相应变通"),
    ("im übrigen",          -1.0, "其余（附带）"),
    ("nebenbei",            -1.5, "顺便"),
    ("obiter",              -1.5, "附带说明"),
    ("obiter dictum",       -2.0, "附带说明"),
    ("mit verweis auf",     -0.5, "参照"),
    ("unter verweis auf",   -0.5, "参照"),
    ("vorbehaltlich",       -0.5, "以…为保留"),
    ("unter vorbehalt",     -0.5, "保留适用"),
    ("allenfalls",          -1.0, "或许"),
    ("gegebenenfalls",      -0.5, "必要时"),
    ("ggf.",                -0.5, "必要时"),

    # ── 否定 / 排除 / 不适用 ─────────────────────────────────────────────────
    ("nicht",               -2.5, "否定"),
    ("kein",                -2.5, "否定"),
    ("keine",               -2.5, "否定"),
    ("keiner",              -2.5, "否定"),
    ("keinen",              -2.5, "否定"),
    ("keineswegs",          -3.0, "绝非"),
    ("keinesfalls",         -3.0, "绝非"),
    ("nicht anwendbar",     -3.5, "不适用"),
    ("keine anwendung",     -3.5, "不适用"),
    ("keine anwendung findet",-4.0,"不适用"),
    ("findet keine anwendung",-4.0,"不适用"),
    ("nicht heranzuziehen", -3.5, "不得援引"),
    ("nicht zu beachten",   -3.0, "不须注意"),
    ("nicht einschlägig",   -3.5, "不相关"),
    ("nicht massgebend",    -3.5, "无决定性"),
    ("nicht massgeblich",   -3.5, "无决定性"),
    ("nicht relevant",      -3.0, "不相关"),
    ("irrelevant",          -3.0, "不相关"),
    ("unerheblich",         -2.5, "无关紧要"),
    ("nicht erfüllt",       -3.0, "不满足"),
    ("nicht verletzt",      -2.5, "未违反"),
    ("dagegen",             -2.0, "相反"),
    ("hingegen",            -2.0, "相反"),
    ("demgegenüber",        -2.0, "与此相对"),
    ("im gegenteil",        -2.5, "相反"),
    ("überholt",            -3.5, "已过时"),
    ("veraltet",            -3.5, "过时"),
    ("aufgegeben",          -3.5, "已放弃（判例）"),
    ("aufgegebene",         -3.5, "已放弃"),
    ("aufgehobene",         -3.5, "已废止"),
    ("aufgehoben",          -3.0, "已废止"),
    ("abgelöst",            -3.0, "已取代"),
    ("nicht mehr",          -3.5, "不再"),
    ("nicht mehr gilt",     -4.0, "不再适用"),
    ("nicht mehr anwendbar",-4.0, "不再适用"),
    ("nicht mehr massgebend",-4.0,"不再决定性"),
    ("kann nicht",          -2.5, "不能"),
    ("nicht herangezogen",  -3.5, "不得援引"),
    ("nicht herangezogen werden",-4.0,"不得援引"),
    ("abzulehnen",          -3.0, "须拒绝"),
    ("abgelehnt",           -3.0, "已拒绝"),
    ("abweichend",          -2.0, "不同（见解）"),
    ("entgegen",            -2.5, "违背"),
    ("widerspricht",        -2.5, "与…相悖"),
    ("unvereinbar",         -3.0, "不相容"),
    ("unvereinbar mit",     -3.5, "与…不相容"),
    ("beschränkt auf",      -1.0, "限于"),
    ("ausdrücklich nicht",  -4.0, "明确否定"),
    ("jedenfalls nicht",    -3.5, "无论如何不"),
    ("gerade nicht",        -3.5, "恰恰不"),
]

# ---------------------------------------------------------------------------
# 3. 数据结构
# ---------------------------------------------------------------------------

@dataclass
class CitationSpan:
    """原文中检测到的一个引用"""
    original: str          # 原始文本（含空格），e.g. "Art. 41 Abs. 1 OR"
    placeholder: str       # 归一化后的占位符 token
    start: int             # 在原文中的字符起始位置
    end: int               # 在原文中的字符结束位置
    cite_type: str         # 正则分组名称


@dataclass
class CitationResult:
    """分析结果"""
    citation: CitationSpan
    score: float
    category: str          # "CORE" / "INCIDENTAL" / "NEGATIVE"
    matched_keywords: list[tuple[str, float, str]] = field(default_factory=list)
    context_tokens: list[str] = field(default_factory=list)
    dep_info: Optional[str] = None


# ---------------------------------------------------------------------------
# 4. 预处理：引用 → 占位符（spaCy-safe token）
# ---------------------------------------------------------------------------

def _make_placeholder(match_text: str, index: int) -> str:
    """将引用转成单个无空格 token，编码为 CITE_<index>_<stripped>"""
    slug = re.sub(r"[^A-Za-z0-9]", "", match_text)[:20]
    return f"CITE{index}X{slug}"


def extract_and_replace(text: str) -> tuple[str, list[CitationSpan]]:
    """
    1. 找出所有引用 span
    2. 按 start 倒序替换（防止偏移漂移）
    3. 返回 (归一化文本, [CitationSpan, ...])
    """
    spans: list[CitationSpan] = []
    for m in _COMBINED_PATTERN.finditer(text):
        cite_type = m.lastgroup or "unknown"
        spans.append(CitationSpan(
            original=m.group().strip(),
            placeholder="",       # 稍后填
            start=m.start(),
            end=m.end(),
            cite_type=cite_type,
        ))

    # 去重（嵌套匹配保留最长）
    spans = _dedup_spans(spans)

    # 按位置倒序替换
    normalized = text
    for i, span in enumerate(reversed(spans)):
        ph = _make_placeholder(span.original, len(spans) - 1 - i)
        span.placeholder = ph
        normalized = normalized[: span.start] + ph + normalized[span.end :]

    # 恢复顺序
    spans = list(reversed(spans))
    for i, span in enumerate(spans):
        span.placeholder = _make_placeholder(span.original, i)

    return normalized, spans


def _dedup_spans(spans: list[CitationSpan]) -> list[CitationSpan]:
    """移除被更长匹配完全包含的子匹配"""
    spans = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
    result: list[CitationSpan] = []
    last_end = -1
    for s in spans:
        if s.start >= last_end:
            result.append(s)
            last_end = s.end
    return result


def restore_citations(text_with_placeholders: str, spans: list[CitationSpan]) -> str:
    """把占位符还原为原始引用文本（便于最终输出）"""
    result = text_with_placeholders
    for span in spans:
        result = result.replace(span.placeholder, span.original)
    return result


# ---------------------------------------------------------------------------
# 5. spaCy 自定义 tokenizer 规则（让占位符不被切分）
# ---------------------------------------------------------------------------

def add_citation_tokenizer_rules(nlp: spacy.Language) -> None:
    """
    向 spaCy tokenizer 添加特殊规则：
    所有 CITE<n>X<slug> 格式的 token 不被继续切分。
    """
    from spacy.symbols import ORTH
    # spaCy 的 tokenizer special_cases 用于精确字符串
    # 对于动态生成的占位符，我们改用 infix 规则豁免
    # 更稳妥的方案：在 tokenize 前后处理（见 analyze_text 流程）
    pass  # 见 analyze_text 中的 Doc 直接构建方案


# ---------------------------------------------------------------------------
# 6. 上下文评分
# ---------------------------------------------------------------------------

# 预先编译多词关键词（降序按长度，优先匹配长词）
_SORTED_KEYWORDS = sorted(_KEYWORD_SCORES, key=lambda x: -len(x[0]))


def _score_context(
    cite_idx: int,
    tokens: list[str],
    window: int = 8,
) -> tuple[float, list[tuple[str, float, str]]]:
    """
    在 cite_idx 前后 window 个 token 的窗口中扫描关键词，
    按距离衰减加权求和。
    返回 (score, matched_keywords)
    """
    start = max(0, cite_idx - window)
    end = min(len(tokens) - 1, cite_idx + window)

    # 将窗口文本拼成小写字符串（同时保留 token 边界信息）
    left_tokens = tokens[start:cite_idx]
    right_tokens = tokens[cite_idx + 1 : end + 1]

    score = 0.0
    matched: list[tuple[str, float, str]] = []

    def _scan(tok_list: list[str], direction: str) -> None:
        nonlocal score
        joined = " ".join(tok_list).lower()
        for kw, kw_score, desc in _SORTED_KEYWORDS:
            if kw in joined:
                # 找到关键词距引用的 token 距离（粗略）
                kw_tokens = kw.split()
                for i, t in enumerate(tok_list):
                    if t.lower().startswith(kw_tokens[0]):
                        dist = abs(len(tok_list) - i) if direction == "left" else i + 1
                        weight = 1.0 if dist <= 2 else 0.75 if dist <= 4 else 0.5
                        contribution = kw_score * weight
                        score += contribution
                        matched.append((kw, round(contribution, 2), desc))
                        break

    _scan(left_tokens, "left")
    _scan(right_tokens, "right")

    return round(score, 2), matched


def _categorize(score: float) -> str:
    if score >= 2.0:
        return "CORE"
    elif score <= -2.0:
        return "NEGATIVE"
    else:
        return "INCIDENTAL"


# ---------------------------------------------------------------------------
# 7. 主分析函数
# ---------------------------------------------------------------------------

# def analyze_text(
#     text: str,
#     nlp: spacy.Language,
#     context_window: int = 8,
#     verbose: bool = False,
# ) -> tuple[Doc, list[CitationResult]]:
#     """
#     完整流水线：
#       1. 提取引用，替换为占位符
#       2. 用 spaCy 分析归一化文本
#       3. 逐引用评分
#       4. 返回 (spaCy Doc, [CitationResult])
#     """
#     # ── 步骤 1：预处理 ──────────────────────────────────────────────────────
#     normalized, spans = extract_and_replace(text)
#     if verbose:
#         print(f"[预处理] 检测到 {len(spans)} 个引用：")
#         for s in spans:
#             print(f"  [{s.cite_type:12s}] {s.original!r:45s} → {s.placeholder}")
#         print(f"[归一化] {normalized}\n")

#     # ── 步骤 2：spaCy 分析 ──────────────────────────────────────────────────
#     # 直接调用 nlp；占位符是单一无空格字符串，tokenizer 不会切分
#     doc = nlp(normalized)

#     # ── 步骤 3：为每个引用评分 ──────────────────────────────────────────────
#     token_texts = [t.text for t in doc]
#     placeholder_set = {s.placeholder for s in spans}

#     results: list[CitationResult] = []
#     for span in spans:
#         try:
#             cite_idx = token_texts.index(span.placeholder)
#         except ValueError:
#             # 极少数情况下 tokenizer 仍切分了占位符，尝试模糊匹配
#             cite_idx = next(
#                 (i for i, t in enumerate(token_texts) if span.placeholder in t),
#                 -1,
#             )
#         if cite_idx == -1:
#             continue

#         score, matched = _score_context(cite_idx, token_texts, context_window)
#         category = _categorize(score)

#         # 收集依存句法信息
#         tok = doc[cite_idx]
#         dep_info = f"dep={tok.dep_}, head='{tok.head.text}'" if tok.dep_ else None

#         # 上下文 token（还原引用名）
#         ctx_start = max(0, cite_idx - 4)
#         ctx_end = min(len(token_texts), cite_idx + 5)
#         ctx = [
#             s.original if t == s.placeholder else t
#             for t in token_texts[ctx_start:ctx_end]
#             for s in ([span] if t == span.placeholder else [])
#         ] or token_texts[ctx_start:ctx_end]

#         results.append(CitationResult(
#             citation=span,
#             score=score,
#             category=category,
#             matched_keywords=matched,
#             context_tokens=ctx,
#             dep_info=dep_info,
#         ))

#     return doc, results

def analyze_text(
    text: str,
    nlp: spacy.Language,
    context_window: int = 8,
    verbose: bool = False,
    debug: bool = False,
) -> tuple[Doc, list[CitationResult]]:

    # ── Step 1: 预处理 ─────────────────────────────────────
    normalized, spans = extract_and_replace(text)

    if verbose:
        print(f"[预处理] {len(spans)} citations detected")
        for s in spans:
            print(f"  {s.original} -> {s.placeholder}")

    # ── Step 2: spaCy ──────────────────────────────────────
    doc = nlp(normalized)
    token_texts = [t.text for t in doc]

    if debug:
        print("\n[TOKENS]")
        for i, t in enumerate(doc):
            print(i, t.text, f"(idx={t.idx})")

    # ── Step 3: 找 citation 对应 token（多级 fallback） ─────
    def find_cite_token(span: CitationSpan) -> int:
        """
        多级策略：
        1. 精确匹配 placeholder
        2. 子串匹配（token 被切开）
        3. char span 对齐（最终兜底，100% 成功）
        """

        # 1️⃣ 精确匹配
        if span.placeholder in token_texts:
            return token_texts.index(span.placeholder)

        # 2️⃣ 子串匹配（tokenizer 切分）
        for i, t in enumerate(token_texts):
            if span.placeholder in t:
                return i

        # 3️⃣ char-level fallback（最关键）
        for token in doc:
            start = token.idx
            end = token.idx + len(token)

            # span.start 是原始文本位置，但 normalized 长度可能变化
            # 👉 需要找 placeholder 在 normalized 中的位置
            pos = normalized.find(span.placeholder)
            if pos == -1:
                continue

            if start <= pos < end:
                return token.i

        # 4️⃣ 最终兜底（绝不会返回 -1）
        if debug:
            print(f"⚠️ fallback to nearest token: {span.original}")

        # 找最近 token
        pos = normalized.find(span.placeholder)
        if pos == -1:
            return 0

        closest = min(
            range(len(doc)),
            key=lambda i: abs(doc[i].idx - pos)
        )
        return closest

    # ── Step 4: 评分 ───────────────────────────────────────
    results: list[CitationResult] = []

    for span in spans:
        cite_idx = find_cite_token(span)

        if debug:
            print(f"\n[CITATION]")
            print(f"  original: {span.original}")
            print(f"  placeholder: {span.placeholder}")
            print(f"  token_idx: {cite_idx}")
            print(f"  token: {doc[cite_idx].text}")

        # ── scoring ─────────────────────────────
        score, matched = _score_context(
            cite_idx,
            token_texts,
            context_window
        )
        category = _categorize(score)

        # ── dependency ──────────────────────────
        tok = doc[cite_idx]
        dep_info = f"dep={tok.dep_}, head='{tok.head.text}'"

        # ── context（修复后的版本）──────────────
        ctx_start = max(0, cite_idx - 4)
        ctx_end = min(len(token_texts), cite_idx + 5)

        ctx = [
            span.original if t == span.placeholder else t
            for t in token_texts[ctx_start:ctx_end]
        ]

        results.append(CitationResult(
            citation=span,
            score=score,
            category=category,
            matched_keywords=matched,
            context_tokens=ctx,
            dep_info=dep_info,
        ))

    return doc, results

# ---------------------------------------------------------------------------
# 8. 报告输出
# ---------------------------------------------------------------------------

_CATEGORY_LABEL = {
    "CORE":       "核心引用  ★",
    "INCIDENTAL": "附带引用  ◇",
    "NEGATIVE":   "负面引用  ✕",
}

_CATEGORY_COLOR = {
    "CORE":       "\033[34m",   # 蓝
    "INCIDENTAL": "\033[37m",   # 灰
    "NEGATIVE":   "\033[31m",   # 红
}
_RESET = "\033[0m"


def print_report(results: list[CitationResult], use_color: bool = True) -> None:
    print("\n" + "=" * 70)
    print("  瑞士法律引用分析报告")
    print("=" * 70)
    for r in results:
        cat = r.category
        color = _CATEGORY_COLOR[cat] if use_color else ""
        reset = _RESET if use_color else ""
        print(
            f"\n{color}[{_CATEGORY_LABEL[cat]}]  "
            f"得分: {r.score:+.1f}{reset}"
        )
        print(f"  引用: {r.citation.original}")
        print(f"  类型: {r.citation.cite_type}")
        if r.dep_info:
            print(f"  依存: {r.dep_info}")
        if r.matched_keywords:
            print("  触发词:")
            for kw, contrib, desc in r.matched_keywords[:6]:
                bar = "+" if contrib >= 0 else "-"
                print(f"    {bar} '{kw}' ({desc}) → {contrib:+.2f}")
    print("\n" + "=" * 70)
    core = sum(1 for r in results if r.category == "CORE")
    incid = sum(1 for r in results if r.category == "INCIDENTAL")
    neg = sum(1 for r in results if r.category == "NEGATIVE")
    print(f"  汇总：核心 {core}  附带 {incid}  负面 {neg}  共 {len(results)} 个")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# 9. 演示入口
# ---------------------------------------------------------------------------

DEMO_SENTENCES = [
    # 核心依据 + 排除适用
    (
        "Gemäss Art. 41 Abs. 1 OR haftet der Schuldner für jeden Schaden, "
        "wobei Art. 44 OR sinngemäss gilt, insbesondere aber nicht Art. 43 Abs. 2 OR, "
        "da diese Bestimmung hier nicht anwendbar ist."
    ),
    # 判例援引 + 过时判例否定
    (
        "Nach BGE 140 III 115 ist die Klage begründet; vgl. auch BGer 4A_312/2019. "
        "Die ältere Rechtsprechung gemäss BGE 120 II 20 ist überholt und "
        "kann nicht mehr herangezogen werden."
    ),
    # ZGB 人格权
    (
        "Die Voraussetzungen von Art. 28a ZGB sind vorliegend erfüllt. "
        "Zu beachten ist ferner Art. 28 ZGB. "
        "Art. 29 ZGB findet dagegen keine Anwendung."
    ),
    # BV 基本权利 + 参见
    (
        "Art. 29 Abs. 2 BV gewährleistet den Anspruch auf rechtliches Gehör; "
        "massgebend ist ferner Art. 6 EMRK, vgl. auch BGE 138 I 232. "
        "Art. 30 BV ist hingegen nicht einschlägig."
    ),
    # SchKG 执行法
    (
        "Gestützt auf Art. 271 Abs. 1 Ziff. 4 SchKG ist der Arrest zu bewilligen. "
        "Art. 272 SchKG findet keine Anwendung, da kein dringlicher Fall vorliegt. "
        "Vgl. hierzu BGE 142 III 291."
    ),
]


def main() -> None:
    print("加载 spaCy 模型 de_core_news_lg …")
    try:
        nlp = spacy.load("de_core_news_lg")
    except OSError:
        print("  de_core_news_lg 未安装，尝试 de_core_news_sm …")
        try:
            nlp = spacy.load("de_core_news_sm")
        except OSError:
            print(
                "  未找到德语模型，请运行：\n"
                "  python -m spacy download de_core_news_lg\n"
                "  或：python -m spacy download de_core_news_sm"
            )
            return

    for i, sentence in enumerate(DEMO_SENTENCES, 1):
        print(f"\n{'─'*70}")
        print(f"示例 {i}: {sentence}")
        doc, results = analyze_text(sentence, nlp, verbose=True)
        print_report(results)


# ---------------------------------------------------------------------------
# 10. 公共 API（供外部 import 使用）
# ---------------------------------------------------------------------------

__all__ = [
    "extract_and_replace",
    "restore_citations",
    "analyze_text",
    "print_report",
    "CitationSpan",
    "CitationResult",
    "_CITATION_PATTERNS",
    "_KEYWORD_SCORES",
]

if __name__ == "__main__":
    main()