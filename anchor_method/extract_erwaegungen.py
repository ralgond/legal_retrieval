"""
提取瑞士德语法律文本中的 Erwägungen 部分
输入：CSV（第一列 citation，第二列 text）
输出：JSONL（citation, erwaegungen, start_pattern, end_pattern, success）
策略：纯正则规则，单进程，速度优先
"""

import re
import json
import csv
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 正则规则（按优先级排列，越具体越靠前）
# ─────────────────────────────────────────────

START_PATTERNS = [
    # 最明确的形式
    r"(?im)^Erw[äa]gungen\s*:?\s*$",
    r"(?im)^E\.\s*$",
    r"(?im)^E\.\s+Erw[äa]gungen\s*:?\s*$",

    # 带罗马数字或字母编号
    r"(?im)^[IVX]+\.\s*Erw[äa]gungen\s*:?\s*$",
    r"(?im)^[A-Z]\.\s*Erw[äa]gungen\s*:?\s*$",

    # 带前缀
    r"(?im)^Aus\s+den\s+Erw[äa]gungen",
    r"(?im)^In\s+Erw[äa]gung\b",

    # 宽松匹配（兜底，放最后）
    r"(?i)Erw[äa]gungen\s*:",
    r"(?i)\bErw[äa]gungen\b",
]

END_PATTERNS = [
    r"(?im)^Dispositiv\s*:?\s*$",
    r"(?im)^[IVX]+\.\s*Dispositiv\s*$",
    r"(?im)^Urteilsformel\s*:?\s*$",
    r"(?im)^Demnach\s+erkennt",
    r"(?im)^Das\s+(Gericht|Bundesgericht|Obergericht)\s+erkennt",
    r"(?im)^Demgemäss\s+wird\s+erkannt",
    r"(?im)^Tenor\s*:?\s*$",
    r"(?im)^Schluss\s*:?\s*$",
]


# ─────────────────────────────────────────────
# 核心提取函数
# ─────────────────────────────────────────────

def extract_erwaegungen(text: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """返回 (提取文本, 命中的start_pattern, 命中的end_pattern)"""
    start_pos = None
    matched_start = None

    for pat in START_PATTERNS:
        m = re.search(pat, text)
        if m:
            start_pos = m.end()
            matched_start = pat
            break

    if start_pos is None:
        return None, None, None

    end_pos = len(text)
    matched_end = None

    for pat in END_PATTERNS:
        m = re.search(pat, text[start_pos:])
        if m:
            end_pos = start_pos + m.start()
            matched_end = pat
            break

    extracted = text[start_pos:end_pos].strip()

    if len(extracted) < 80:
        return None, matched_start, matched_end

    return extracted, matched_start, matched_end


# ─────────────────────────────────────────────
# 主处理函数
# ─────────────────────────────────────────────

def process_csv(
    input_file: str,
    output_file: str,
    limit: Optional[int] = None,
    log_every: int = 5000,
):
    success = 0
    failed = 0
    pattern_stats: dict[str, int] = {}

    citation_l = []
    text_l = []

    with open(input_file, encoding="utf-8", newline="") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:

        reader = csv.reader(f_in)
        header = next(reader, None)  # 跳过表头（如果有）

        for i, row in enumerate(reader):
            if limit and i >= limit:
                break

            if len(row) < 2:
                continue

            citation = row[0].strip()
            text     = row[1]

            erwaegungen, start_pat, end_pat = extract_erwaegungen(text)

            result = {
                "citation":      citation,
                "erwaegungen":   erwaegungen,
                "start_pattern": start_pat,
                "end_pattern":   end_pat,
                "success":       erwaegungen is not None,
            }
            if result["success"]:
                # f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                citation_l.append(citation)
                text_l.append(erwaegungen)

            if erwaegungen is not None:
                success += 1
                pattern_stats[start_pat] = pattern_stats.get(start_pat, 0) + 1
            else:
                failed += 1

            if (i + 1) % log_every == 0:
                logger.info(f"Progress: {i+1} | success={success} failed={failed}")

    df = pd.DataFrame({'citation':citation_l, 'text':text_l})
    df.to_csv("../data/anchor_method/erwaegungen.csv", index=False)

    total = success + failed
    logger.info(f"Done. Total={total} | success={success} | failed={failed}")
    logger.info(f"Pattern hits:\n{json.dumps(pattern_stats, indent=2, ensure_ascii=False)}")
    return {"total": total, "success": success, "failed": failed, "pattern_stats": pattern_stats}


# ─────────────────────────────────────────────
# 失败案例分析（帮助补充规则）
# ─────────────────────────────────────────────

def analyze_failures(output_file: str, sample_size: int = 20):
    """打印失败案例的 citation，便于回查原始CSV。"""
    failures = []
    with open(output_file, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not r["success"]:
                failures.append(r["citation"])

    logger.info(f"Total failed: {len(failures)}")
    print("\nFailed citations (sample):")
    for c in failures[:sample_size]:
        print(f"  {c}")


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    INPUT_FILE  = "../data/court_considerations.csv"  # ← 改成您的路径
    OUTPUT_FILE = "../data/anchor_method/erwaegungen.jsonl"

    # Step 1: 先跑100行试验
    stats = process_csv(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        limit=None,
    )
    print(stats)

    # Step 2: 看失败案例
    analyze_failures(OUTPUT_FILE, sample_size=20)

    # Step 3: 全量（去掉 limit）
    # stats = process_csv(INPUT_FILE, OUTPUT_FILE)