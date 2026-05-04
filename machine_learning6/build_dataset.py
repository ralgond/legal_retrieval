"""
build_dataset.py
----------------
输入：JSONL文件，每行格式见README
输出：../data/ml6/ 下的 train.jsonl / dev.jsonl / test.jsonl

输入格式（一对多）：
{
  "query_id": "q001",
  "query": "...",
  "gold_citations": ["Art. 335 OR", "BGE 110 II 99"],
  "cc_list": [
    {
      "cc_id": "BGE_123_I_456",
      "cc_sentences": [{"sent_id": 0, "text": "..."}, ...]
    },
    ...
  ]
}

gold_citations 为纯字符串列表，sent_id 由代码在每个 CC 的 sentences 中自动搜索。
若某个 gold citation 在当前 CC 中找不到，则跳过该 citation（它属于其他 CC）。

Evidence定义：citation所在句 ± window句（默认window=2）拼接而成。
Hard negative定义：gold citation的evidence窗口内出现的、不在gold集合中的citations。
Random negative定义：从同一CC中随机采样的非gold citations。
训练格式：Pairwise，每条样本 = (query, evidence_pos, citation_pos, evidence_neg, citation_neg)
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import os
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)
    
from citation_utils import extract_citations_from_text

# ── 配置 ──────────────────────────────────────────────────────────────────────
WINDOW = 2                  # evidence窗口大小（前后各N句）
HARD_NEG_PER_GOLD = 2       # 每个gold citation最多配几个hard negative
RAND_NEG_PER_GOLD = 1       # 每个gold citation额外配几个random negative
TRAIN_RATIO = 0.8
DEV_RATIO   = 0.1
# TEST_RATIO  = 1 - TRAIN_RATIO - DEV_RATIO
SEED = 42
OUTPUT_DIR = Path("../data/ml6")
# ─────────────────────────────────────────────────────────────────────────────

# ── Evidence构建 ──────────────────────────────────────────────────────────────

def build_evidence(sentences: list[str], anchor_idx: int, window: int = WINDOW) -> str:
    """
    以anchor_idx为中心，取前后window句（共最多2*window+1句）拼接为evidence。
    sentences: CC按句切好的纯文本列表。
    """
    n = len(sentences)
    start = max(0, anchor_idx - window)
    end   = min(n - 1, anchor_idx + window)
    return " ".join(sentences[start: end + 1])
# ── Hard Negative挖掘 ─────────────────────────────────────────────────────────

def mine_hard_negatives(
    sentences: list[str],
    gold_citations: list[dict],   # [{"citation_id": str, "sent_id": int}, ...]
    window: int = WINDOW,
    max_per_gold: int = HARD_NEG_PER_GOLD,
) -> list[dict]:
    """
    Hard negative：gold citation的evidence窗口内出现的、不在gold集合中的citations。
    返回 [{"citation_id": str, "evidence": str, "source": "hard"}, ...]
    """
    gold_ids = {g["citation_id"] for g in gold_citations}
    hard_negs = []
    seen_neg_ids = set()

    for gold in gold_citations:
        anchor = gold["sent_id"]
        n = len(sentences)
        start = max(0, anchor - window)
        end   = min(n - 1, anchor + window)
        window_text = " ".join(sentences[start: end + 1])

        candidates = extract_citations_from_text(window_text)
        count = 0
        for cand in candidates:
            if cand in gold_ids or cand in seen_neg_ids:
                continue
            # 找该citation在CC中最近的句子作为其evidence anchor
            neg_anchor = _find_citation_anchor(sentences, cand, preferred_near=anchor)
            if neg_anchor is None:
                neg_anchor = anchor  # fallback：复用同一窗口
            evidence = build_evidence(sentences, neg_anchor, window)
            hard_negs.append({
                "citation_id": cand,
                "evidence":    evidence,
                "source":      "hard",
            })
            seen_neg_ids.add(cand)
            count += 1
            if count >= max_per_gold:
                break

    return hard_negs
def _find_citation_anchor(
    sentences: list[str],
    citation_id: str,
    preferred_near: Optional[int] = None,
) -> Optional[int]:
    """在sentences中找citation_id首次出现的句子索引（优先靠近preferred_near）。"""
    candidates = []
    for i, sent in enumerate(sentences):
        if citation_id in sent:
            candidates.append(i)
    if not candidates:
        return None
    if preferred_near is None:
        return candidates[0]
    return min(candidates, key=lambda x: abs(x - preferred_near))
# ── Random Negative采样 ───────────────────────────────────────────────────────

def sample_random_negatives(
    sentences: list[str],
    gold_citations: list[dict],
    hard_neg_ids: set[str],
    n_per_gold: int = RAND_NEG_PER_GOLD,
    rng: random.Random = None,
) -> list[dict]:
    """
    从CC全文中提取所有citations，去掉gold和hard neg后随机采样。
    """
    if rng is None:
        rng = random.Random(SEED)

    gold_ids = {g["citation_id"] for g in gold_citations}
    all_citations = extract_citations_from_text(" ".join(sentences))
    pool = [c for c in all_citations if c not in gold_ids and c not in hard_neg_ids]
    if not pool:
        return []

    n_sample = min(len(pool), n_per_gold * len(gold_citations))
    sampled = rng.sample(pool, n_sample)

    result = []
    for cand in sampled:
        anchor = _find_citation_anchor(sentences, cand)
        if anchor is None:
            anchor = rng.randint(0, len(sentences) - 1)
        evidence = build_evidence(sentences, anchor)
        result.append({
            "citation_id": cand,
            "evidence":    evidence,
            "source":      "random",
        })
    return result
# ── Gold Citation定位 ─────────────────────────────────────────────────────────

def resolve_gold_citations(
    sentences: list[str],
    gold_citation_ids: list[str],
) -> list[dict]:
    """
    将 gold citation 字符串列表解析为带 sent_id 的 dict 列表。
    在 sentences 中搜索每个 citation_id，找到则记录其首次出现的句子索引。
    找不到则跳过（该 citation 属于其他 CC）。
    返回：[{"citation_id": str, "sent_id": int}, ...]
    """
    result = []
    for cit_id in gold_citation_ids:
        anchor = _find_citation_anchor(sentences, cit_id)
        if anchor is not None:
            result.append({"citation_id": cit_id, "sent_id": anchor})
    return result


# ── Pairwise样本构建 ───────────────────────────────────────────────────────────

def build_pairs(
    query_id: str,
    query: str,
    cc_id: str,
    sentences: list[str],
    gold_citations: list[dict],
    rng: random.Random,
) -> list[dict]:
    """
    为一条原始记录构建所有pairwise训练样本。
    每条样本：
    {
        query_id, cc_id, query,
        pos_citation_id, pos_evidence,
        neg_citation_id, neg_evidence, neg_source
    }
    """
    # 构建gold的evidence
    gold_items = []
    for g in gold_citations:
        ev = build_evidence(sentences, g["sent_id"])
        gold_items.append({"citation_id": g["citation_id"], "evidence": ev})

    # 挖掘negatives
    hard_negs  = mine_hard_negatives(sentences, gold_citations)
    hard_neg_ids = {h["citation_id"] for h in hard_negs}
    rand_negs  = sample_random_negatives(sentences, gold_citations, hard_neg_ids, rng=rng)

    all_negs = hard_negs + rand_negs
    if not all_negs:
        return []

    pairs = []
    for pos in gold_items:
        # 优先配hard neg，再配random neg
        for neg in all_negs:
            pairs.append({
                "query_id":       query_id,
                "cc_id":          cc_id,
                "query":          query,
                "pos_citation_id": pos["citation_id"],
                "pos_evidence":   pos["evidence"],
                "neg_citation_id": neg["citation_id"],
                "neg_evidence":   neg["evidence"],
                "neg_source":     neg["source"],
            })

    return pairs
# ── 主流程 ────────────────────────────────────────────────────────────────────

def load_raw(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
def split_data(
    items: list,
    train_ratio: float = TRAIN_RATIO,
    dev_ratio: float   = DEV_RATIO,
    rng: random.Random = None,
) -> tuple[list, list, list]:
    if rng:
        rng.shuffle(items)
    else:
        random.shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_dev   = int(n * dev_ratio)
    return items[:n_train], items[n_train: n_train + n_dev], items[n_train + n_dev:]
def write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  写入 {len(records):>6} 条 → {path}")
def build(input_path: str, output_dir: Path = OUTPUT_DIR, seed: int = SEED):
    rng = random.Random(seed)
    print(f"[1/4] 读取原始数据：{input_path}")
    raw_records = load_raw(input_path)
    print(f"      共 {len(raw_records)} 条原始记录")

    print("[2/4] 构建pairwise样本…")
    all_pairs = []
    skipped   = 0
    for rec in raw_records:
        gold_citation_ids = rec["gold_citations"]   # list[str]
        for cc in rec["cc_list"]:
            sentences = [s["text"] for s in cc["cc_sentences"]]
            # 在当前 CC 中定位 gold citations（找不到的自动跳过）
            gold_citations = resolve_gold_citations(sentences, gold_citation_ids)
            if not gold_citations:
                skipped += 1
                continue
            pairs = build_pairs(
                query_id      = rec["query_id"],
                query         = rec["query"],
                cc_id         = cc["cc_id"],
                sentences     = sentences,
                gold_citations= gold_citations,
                rng           = rng,
            )
            if not pairs:
                skipped += 1
            all_pairs.extend(pairs)

    print(f"      生成 {len(all_pairs)} 个pair，{skipped} 条记录因无negative被跳过")

    # 统计hard/random比例
    hard_cnt = sum(1 for p in all_pairs if p["neg_source"] == "hard")
    rand_cnt = len(all_pairs) - hard_cnt
    print(f"      hard negatives: {hard_cnt}  |  random negatives: {rand_cnt}")

    print("[3/4] 划分train/dev/test…")
    train, dev, test = split_data(all_pairs, rng=rng)

    print("[4/4] 写入文件…")
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "dev.jsonl",   dev)
    write_jsonl(output_dir / "test.jsonl",  test)
    print("完成。")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="构建Swiss法律检索pairwise数据集")
    parser.add_argument("--input",  help="原始JSONL路径，e.g. raw/swiss_legal.jsonl")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--window",     type=int, default=WINDOW,            help="evidence上下文窗口大小")
    parser.add_argument("--hard_neg",   type=int, default=HARD_NEG_PER_GOLD, help="每个gold最多配几个hard neg")
    parser.add_argument("--rand_neg",   type=int, default=RAND_NEG_PER_GOLD, help="每个gold额外配几个random neg")
    parser.add_argument("--seed",       type=int, default=SEED)
    args = parser.parse_args()

    # 允许命令行覆盖全局配置
    WINDOW            = args.window
    HARD_NEG_PER_GOLD = args.hard_neg
    RAND_NEG_PER_GOLD = args.rand_neg

    build(args.input, Path(args.output_dir), args.seed)