"""
Swiss German Legal Document Section Splitter
Splits court considerations (Erwägungen) into standardized sections.

Section weights (confidence scores):
    sachverhalt  : 0.70  — 事实陈述 (Facts)
    erwaegungen  : 0.60  — 法律考量 (Legal reasoning, core)
    dispositiv   : 0.90  — 判决主文 (Operative part, highest weight)
    rechtsmittel : 0.35  — 上诉告知 (Appeal notice)
    unterschrift : 0.15  — 日期/签名 (Date/Signature)
"""

from __future__ import annotations

import re
import sys
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Section configuration
# ---------------------------------------------------------------------------
SECTION_PATTERNS: dict[str, dict] = {
    "sachverhalt": {
        "weight": 0.2,
        "label_de": "Sachverhalt",
        "label_zh": "事实陈述",
        # Exact header patterns (matched against the trimmed line, case-insensitive)
        "header_re": [
            r"^sachverhalt\s*[:\.]?\s*$",
            r"^tatbestand\s*[:\.]?\s*$",
            r"^tatsachen\s*[:\.]?\s*$",
            r"^in\s+tatsächlicher\s+hinsicht\s*[:\.]?\s*$",
            r"^sachverhaltsdarstellung\s*[:\.]?\s*$",
            r"^sachverhaltsfeststellung\s*[:\.]?\s*$",
            # Numbered: "A. Sachverhalt"
            r"^[A-Z]\.\s+sachverhalt",
            r"^[IVX]+\.\s+sachverhalt",
        ],
        # Keywords that raise score if found in non-header context
        "keywords": [
            "sachverhalt", "tatbestand", "tatsachenfeststellung",
            "ergibt sich folgender sachverhalt", "in tatsächlicher hinsicht",
        ],
    },
    "erwaegungen": {
        "weight": 1.5,
        "label_de": "Erwägungen",
        "label_zh": "法律考量",
        "header_re": [
            r"^erwägungen\s*[:\.]?\s*$",
            r"^erwaegungen\s*[:\.]?\s*$",
            r"^aus\s+den\s+erwägungen\s*[:\.]?\s*$",
            r"^in\s+rechtlicher\s+hinsicht\s*[:\.]?\s*$",
            r"^rechtliche\s+beurteilung\s*[:\.]?\s*$",
            r"^in\s+erwägung\s*[:\.]?\s*$",
            r"^das\s+gericht\s+erwägt\s*[:\.]?\s*$",
            r"^[A-Z]\.\s+erwägungen",
            r"^[IVX]+\.\s+erwägungen",
        ],
        "keywords": [
            "erwägungen", "erwaegungen", "rechtliche beurteilung",
            "in rechtlicher hinsicht", "aus den erwägungen",
        ],
    },
    "dispositiv": {
        "weight": 0.5,
        "label_de": "Dispositiv",
        "label_zh": "判决主文",
        "header_re": [
            r"^dispositiv\s*[:\.]?\s*$",
            r"^demnach\s+erkennt\s*",
            r"^das\s+gericht\s+erkennt\s*",
            r"^wird\s+erkannt\s*[:\.]?\s*$",
            r"^urteil\s*[:\.]?\s*$",
            r"^beschluss\s*[:\.]?\s*$",
            r"^verfügung\s*[:\.]?\s*$",
            r"^entscheid\s*[:\.]?\s*$",
            r"^[A-Z]\.\s+dispositiv",
            r"^[IVX]+\.\s+dispositiv",
        ],
        "keywords": [
            "dispositiv", "demnach erkennt", "wird erkannt",
            "das gericht erkennt", "demnach beschliesst",
        ],
    },
    "rechtsmittel": {
        "weight": 0.15,
        "label_de": "Rechtsmittel",
        "label_zh": "上诉告知",
        "header_re": [
            r"^rechtsmittel\s*[:\.]?\s*$",
            r"^rechtsmittelbelehrung\s*[:\.]?\s*$",
            r"^hinweis\s*[:\.]?\s*$",
            r"^gegen\s+diesen\s+entscheid\s*",
            r"^kann\s+innert\s+",
            r"^[A-Z]\.\s+rechtsmittel",
        ],
        "keywords": [
            "rechtsmittel", "rechtsmittelbelehrung", "kann innert",
            "beschwerdefrist", "einzureichen beim", "rekurs",
        ],
    },
    "unterschrift": {
        "weight": 0.05,
        "label_de": "Unterschrift / Datum",
        "label_zh": "日期/签名",
        "header_re": [
            r"^unterschrift\s*[:\.]?\s*$",
            r"^der\s+präsident\s*",
            r"^die\s+präsidentin\s*",
            r"^der\s+richter\s*",
            r"^die\s+richterin\s*",
            r"^der\s+gerichtspräsident\s*",
            r"^im\s+namen\s+",
            r"^namens\s+",
            r"^\d{1,2}\.\s+\w+\s+\d{4}\s*$",   # e.g. "12. März 2024"
        ],
        "keywords": [
            "unterschrift", "der präsident", "die präsidentin",
            "im namen des", "namens des gerichts", "ausgestellt",
        ],
    },
}

# Compile all header regexes once
_COMPILED: dict[str, list[re.Pattern]] = {
    sec: [re.compile(pat, re.IGNORECASE) for pat in cfg["header_re"]]
    for sec, cfg in SECTION_PATTERNS.items()
}

# Ordered by descending weight → higher-confidence sections win ties
_SECTION_ORDER = sorted(
    SECTION_PATTERNS.keys(),
    key=lambda k: SECTION_PATTERNS[k]["weight"],
    reverse=True,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Section:
    section_type: str          # dict key, e.g. "dispositiv"
    label_de: str              # German label
    label_zh: str              # Chinese label
    weight: float              # confidence weight
    start_line: int            # 1-based
    end_line: int              # 1-based, inclusive
    text: str                  # raw text of this section
    confidence: str = ""       # "header" | "keyword" | "positional" | "fallback"
    notes: list[str] = field(default_factory=list)

    def char_count(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["char_count"] = self.char_count()
        return d


# ---------------------------------------------------------------------------
# Core detection helpers
# ---------------------------------------------------------------------------

def _match_header(line: str) -> Optional[str]:
    """Return the section key if 'line' matches a known section header, else None."""
    stripped = line.strip()
    if not stripped:
        return None
    for sec in _SECTION_ORDER:
        for pattern in _COMPILED[sec]:
            if pattern.search(stripped):
                return sec
    return None


def _keyword_score(line: str) -> dict[str, float]:
    """Return a score-per-section based on keyword hits in 'line'."""
    lower = line.lower()
    scores: dict[str, float] = {}
    for sec, cfg in SECTION_PATTERNS.items():
        hits = sum(1 for kw in cfg["keywords"] if kw in lower)
        if hits:
            scores[sec] = hits * cfg["weight"]
    return scores


def _best_keyword_section(block_text: str) -> Optional[str]:
    """
    Accumulate keyword scores over all lines in a block and return
    the section with the highest total score (must exceed threshold).
    """
    totals: dict[str, float] = {}
    for line in block_text.splitlines():
        for sec, score in _keyword_score(line).items():
            totals[sec] = totals.get(sec, 0.0) + score
    if not totals:
        return None
    best = max(totals, key=lambda k: totals[k])
    return best if totals[best] >= 1.0 else None


# ---------------------------------------------------------------------------
# Fallback strategies
# ---------------------------------------------------------------------------

def _positional_fallback(sections: list[Section]) -> list[Section]:
    """
    If there is only one un-typed block, assign it based on document position:
      - first block  → sachverhalt
      - last block   → unterschrift  (if short)
      - middle block → erwaegungen
    """
    untyped = [s for s in sections if s.confidence == "fallback"]
    total = len(sections)

    for s in untyped:
        idx = sections.index(s)
        if idx == 0:
            s.section_type = "sachverhalt"
            s.confidence = "positional"
            s.notes.append("Assigned as 'sachverhalt' (first block, positional fallback)")
        elif idx == total - 1 and s.char_count() < 400:
            s.section_type = "unterschrift"
            s.confidence = "positional"
            s.notes.append("Assigned as 'unterschrift' (last short block, positional fallback)")
        else:
            s.section_type = "erwaegungen"
            s.confidence = "positional"
            s.notes.append("Assigned as 'erwaegungen' (middle block, positional fallback)")

        # Update labels
        cfg = SECTION_PATTERNS[s.section_type]
        s.label_de = cfg["label_de"]
        s.label_zh = cfg["label_zh"]
        s.weight = cfg["weight"]

    return sections


def _merge_orphaned_blocks(sections: list[Section]) -> list[Section]:
    """
    Very short blocks (<= 3 lines) still marked 'fallback' are merged
    into the nearest preceding section.
    """
    result: list[Section] = []
    for s in sections:
        if s.confidence == "fallback" and len(s.text.splitlines()) <= 3 and result:
            prev = result[-1]
            prev.text = prev.text.rstrip() + "\n" + s.text
            prev.end_line = s.end_line
            prev.notes.append(f"Merged orphaned block (lines {s.start_line}-{s.end_line})")
        else:
            result.append(s)
    return result


# ---------------------------------------------------------------------------
# Main splitter
# ---------------------------------------------------------------------------

def split_court_document(text: str) -> list[Section]:
    """
    Split a Swiss German court document into its canonical sections.

    Strategy (in order):
      1. Header detection  — exact regex match on a trimmed line
      2. Keyword scoring   — accumulate keyword hits within a paragraph block
      3. Merge orphans     — tiny leftover blocks absorbed into predecessor
      4. Positional        — assign by position in document
      5. Ultimate fallback — label everything remaining as 'erwaegungen'

    Returns a list of Section objects in document order.
    """
    if not text or not text.strip():
        print("Empty input — returning empty section list.")
        return []

    lines = text.splitlines()
    raw_blocks: list[tuple[int, int, str]] = []  # (start_line, end_line, text)

    # ── Pass 1: split by blank-line-separated paragraphs ──────────────────
    buf: list[str] = []
    buf_start = 0

    for i, line in enumerate(lines):
        if line.strip() == "":
            if buf:
                raw_blocks.append((buf_start + 1, i, "\n".join(buf)))
                buf = []
                buf_start = i + 1
        else:
            if not buf:
                buf_start = i
            buf.append(line)

    if buf:
        raw_blocks.append((buf_start + 1, len(lines), "\n".join(buf)))

    if not raw_blocks:
        print("No non-empty paragraphs found.")
        return []

    # ── Pass 2: classify each block ───────────────────────────────────────
    sections: list[Section] = []
    pending_lines: list[str] = []   # accumulate lines before a header is found
    pending_start: int = 1

    def _flush_pending(until_line: int) -> None:
        nonlocal pending_lines, pending_start
        if pending_lines:
            blk_text = "\n".join(pending_lines).strip()
            if blk_text:
                sec = Section(
                    section_type="erwaegungen",  # temporary
                    label_de="(unclassified)",
                    label_zh="(未分类)",
                    weight=0.0,
                    start_line=pending_start,
                    end_line=until_line,
                    text=blk_text,
                    confidence="fallback",
                    notes=["No header or keywords matched — awaiting fallback resolution"],
                )
                sections.append(sec)
            pending_lines = []

    current_section_type: Optional[str] = None
    current_lines: list[str] = []
    current_start: int = 1

    def _commit_section(until_line: int) -> None:
        nonlocal current_section_type, current_lines, current_start
        if current_section_type and current_lines:
            blk_text = "\n".join(current_lines).strip()
            if blk_text:
                cfg = SECTION_PATTERNS[current_section_type]
                sections.append(Section(
                    section_type=current_section_type,
                    label_de=cfg["label_de"],
                    label_zh=cfg["label_zh"],
                    weight=cfg["weight"],
                    start_line=current_start,
                    end_line=until_line,
                    text=blk_text,
                    confidence="header",
                ))
        current_section_type = None
        current_lines = []

    for start, end, block_text in raw_blocks:
        first_line = block_text.splitlines()[0] if block_text else ""
        detected = _match_header(first_line)

        if detected:
            # Commit whatever was open
            _commit_section(start - 1)
            _flush_pending(start - 1)
            current_section_type = detected
            current_start = start
            current_lines = block_text.splitlines()
        else:
            if current_section_type:
                # Append to current section
                current_lines.extend(block_text.splitlines())
            else:
                # No section open yet — try keyword scoring
                kw_sec = _best_keyword_section(block_text)
                if kw_sec:
                    _flush_pending(start - 1)
                    cfg = SECTION_PATTERNS[kw_sec]
                    sections.append(Section(
                        section_type=kw_sec,
                        label_de=cfg["label_de"],
                        label_zh=cfg["label_zh"],
                        weight=cfg["weight"],
                        start_line=start,
                        end_line=end,
                        text=block_text.strip(),
                        confidence="keyword",
                        notes=["Classified via keyword score (no explicit header)"],
                    ))
                else:
                    # Accumulate as pending (might be merged later)
                    if not pending_lines:
                        pending_start = start
                    pending_lines.extend(block_text.splitlines())

    # Commit the final open section / pending buffer
    _commit_section(len(lines))
    _flush_pending(len(lines))

    # ── Pass 3: merge adjacent blocks of the same type ────────────────────
    merged: list[Section] = []
    for s in sections:
        if merged and merged[-1].section_type == s.section_type:
            prev = merged[-1]
            prev.text = prev.text.rstrip() + "\n\n" + s.text
            prev.end_line = s.end_line
            prev.notes.extend(s.notes)
        else:
            merged.append(s)
    sections = merged

    # ── Pass 4: fallback resolution ───────────────────────────────────────
    sections = _merge_orphaned_blocks(sections)
    sections = _positional_fallback(sections)

    # Ultimate fallback (Pass 5): anything still unresolved → erwaegungen
    for s in sections:
        if s.confidence == "fallback":
            cfg = SECTION_PATTERNS["erwaegungen"]
            s.section_type = "erwaegungen"
            s.label_de = cfg["label_de"]
            s.label_zh = cfg["label_zh"]
            s.weight = cfg["weight"]
            s.confidence = "fallback"
            s.notes.append("Ultimate fallback: assigned to 'erwaegungen'")

    return sections


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_summary(sections: list[Section]) -> None:
    """Print a human-readable summary to stdout."""
    SEP = "─" * 70
    print(f"\n{'═'*70}")
    print(f"  Swiss Court Document — Section Split  ({len(sections)} sections)")
    print(f"{'═'*70}")
    for i, s in enumerate(sections, 1):
        conf_tag = {
            "header":     "[HEADER]   ",
            "keyword":    "[KEYWORD]  ",
            "positional": "[POSITION] ",
            "fallback":   "[FALLBACK] ",
        }.get(s.confidence, "[?]        ")
        print(f"\n{SEP}")
        print(
            f"Section {i}/{len(sections)}  {conf_tag}"
            f"  weight={s.weight:.2f}  lines {s.start_line}–{s.end_line}"
        )
        print(f"  {s.section_type.upper():15s}  {s.label_de} / {s.label_zh}")
        if s.notes:
            for note in s.notes:
                print(f"  ⚠  {note}")
        print(SEP)
        # Print first 300 chars as preview
        preview = s.text[:300].replace("\n", "\n  ")
        print(f"  {preview}")
        if len(s.text) > 300:
            print(f"  … (+{len(s.text)-300} chars)")
    print(f"\n{'═'*70}\n")



# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
DEMO_TEXT = """
Sachverhalt
 
A. Die Klägerin X. AG (nachfolgend Klägerin) ist eine Aktiengesellschaft mit Sitz in Zürich.
Sie verlangt von der Beklagten Y. GmbH Schadenersatz im Betrag von CHF 45'000.
 
B. Die Beklagte bestreitet die Forderung vollumfänglich und beantragt Abweisung der Klage.
 
Erwägungen
 
1. Zuständigkeit
Das angerufene Gericht ist gemäss Art. 10 ZPO örtlich zuständig, da die Beklagte ihren Sitz
im Kanton Zürich hat. Die sachliche Zuständigkeit ergibt sich aus Art. 6 ZPO.
 
2. Materiell-rechtliche Beurteilung
Die Klägerin stützt ihre Forderung auf Art. 97 OR. Voraussetzung ist eine Pflichtverletzung,
ein Schaden sowie ein Kausalzusammenhang. Alle Tatbestandsmerkmale sind vorliegend erfüllt.
 
Dispositiv
 
Demnach erkennt das Gericht:
 
1. Die Beklagte wird verurteilt, der Klägerin CHF 45'000 nebst Zins zu 5% seit 1. März 2023
   zu bezahlen.
2. Die Gerichtskosten von CHF 2'500 werden der Beklagten auferlegt.
3. Die Beklagte hat der Klägerin eine Parteientschädigung von CHF 5'000 zu bezahlen.
 
Rechtsmittelbelehrung
 
Gegen diesen Entscheid kann innert 30 Tagen seit Zustellung beim Obergericht des Kantons
Zürich Berufung erhoben werden (Art. 308 ff. ZPO). Die Berufungsschrift ist schriftlich
einzureichen.
 
Der Gerichtspräsident
 
12. Januar 2025
 
Dr. Hans Müller
Gerichtspräsident
"""
if __name__ == "__main__":
    sections = split_court_document(DEMO_TEXT)
    print_summary(sections)

