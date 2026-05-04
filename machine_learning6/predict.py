"""
predict.py
----------
对一个query下所有CC中的候选citations进行打分并排序（普通LoRA，无量化）。

输入：JSONL文件，每行一条预测记录（格式见文件底部 INPUT FORMAT）
输出：JSONL文件，每行包含该query下各CC排序后的citations列表（含分数）

用法：
    python predict.py \
        --base_model_path /path/to/base_model \
        --adapter_path    ../data/ml6/checkpoints/best \
        --input_file      data/predict_input.jsonl \
        --output_file     data/predict_output.jsonl \
        --batch_size 16 \
        --max_seq_len 1024
"""

import json
import argparse
from pathlib import Path

import os
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)
from citation_utils import split_sentences, extract_citations_from_text

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm


# ── Prompt（与训练保持完全一致）────────────────────────────────────────────────
PROMPT_TEMPLATE = (
    "<s>[INST] Frage: {query}\n"
    "Beleg: {evidence}\n"
    "Zitat: {citation_id}\n"
    "Ist dieses Zitat relevant? [/INST] "
)
POS_LABEL = "Ja"

WINDOW = 2   # 与build_dataset.py保持一致


# ── Evidence构建（与build_dataset.py完全一致）──────────────────────────────────

def build_evidence(sentences: list[str], anchor_idx: int, window: int = WINDOW) -> str:
    n = len(sentences)
    start = max(0, anchor_idx - window)
    end   = min(n - 1, anchor_idx + window)
    return " ".join(sentences[start: end + 1])


def build_prompt(query: str, evidence: str, citation_id: str) -> str:
    return PROMPT_TEMPLATE.format(
        query=query,
        evidence=evidence,
        citation_id=citation_id,
    )


# ── 模型加载 ───────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    base_model_path: str,
    adapter_path: str | None = None,
    torch_dtype: str = "bfloat16",
):
    """
    base_model_path : HuggingFace模型名或本地路径（原始base model）
    adapter_path    : LoRA adapter目录（含adapter_config.json）；
                      为None时直接用base model推理（已merge或全量微调）
    torch_dtype     : 模型精度，bfloat16 / float16 / float32
    """
    # tokenizer：优先从adapter目录加载（保留训练时的special tokens配置）
    tokenizer_path = adapter_path if adapter_path else base_model_path
    print(f"加载tokenizer：{tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        print("  fast tokenizer加载成功")
    except Exception as e:
        print(f"  fast tokenizer加载失败（{e}），回退slow tokenizer…")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        print("  slow tokenizer加载成功")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    t_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    # 加载base model
    print(f"加载base model：{base_model_path}  dtype={torch_dtype}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=t_dtype,
        device_map={"": device},
        trust_remote_code=True,
    )

    # 加载LoRA adapter
    if adapter_path:
        adapter_cfg_file = Path(adapter_path) / "adapter_config.json"
        if not adapter_cfg_file.exists():
            raise FileNotFoundError(
                f"adapter_config.json 不存在于 {adapter_path}，"
                "请确认路径是否为训练保存的PEFT checkpoint目录。"
            )
        print(f"加载LoRA adapter：{adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        print("  LoRA adapter加载完成")

    model.eval()
    return model, tokenizer, device


# ── 批量打分 ───────────────────────────────────────────────────────────────────

NEG_LABEL = "Nein"

@torch.no_grad()
def score_batch(
    model,
    tokenizer,
    prompts: list[str],
    pos_label_id: int,
    neg_label_id: int,
    max_seq_len: int,
    device: str,
) -> list[float]:
    """
    对一批prompt打分，返回 log_p(Ja) - log_p(Nein) 列表（float）。
    与训练时 evaluate_f1 的评分方式保持完全一致。
    """
    enc = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    ).to(device)

    outputs   = model(**enc)
    last_logits = outputs.logits[:, -1, :]               # (B, V)
    log_probs   = F.log_softmax(last_logits, dim=-1)     # (B, V)
    scores = (log_probs[:, pos_label_id] - log_probs[:, neg_label_id]).cpu().tolist()
    return scores


# ── 跨CC全局排序 ──────────────────────────────────────────────────────────────

def build_global_ranking(cc_results: list[dict]) -> list[dict]:
    """
    对同一query下所有CC的citations做跨CC全局排序：
      1. 每个CC内先做softmax归一化，消除不同CC的置信度基线差异
      2. 合并所有CC的归一化分数，全局降序排列
      3. citation_id相同时取最高归一化分数（同一法条可能在多个CC出现）
    返回去重后的全局排序列表。
    """
    # 收集每个CC内的 (citation_id, cc_id, sent_id, raw_score)
    all_entries = []
    for cc in cc_results:
        items = cc["ranked_citations"]
        if not items:
            continue
        raw_scores = torch.tensor([c["score"] for c in items], dtype=torch.float32)
        normed     = torch.softmax(raw_scores, dim=0).tolist()
        for c, ns in zip(items, normed):
            all_entries.append({
                "citation_id":   c["citation_id"],
                "cc_id":         cc["cc_id"],
                "sent_id":       c["sent_id"],
                "raw_score":     c["score"],
                "normed_score":  round(ns, 6),
            })

    if not all_entries:
        return []

    # 去重：同一citation_id保留normed_score最高的那条
    best: dict[str, dict] = {}
    for e in all_entries:
        cid = e["citation_id"]
        if cid not in best or e["normed_score"] > best[cid]["normed_score"]:
            best[cid] = e

    # 全局降序排列
    global_ranked = sorted(best.values(), key=lambda x: x["normed_score"], reverse=True)
    for rank, entry in enumerate(global_ranked, 1):
        entry["global_rank"] = rank

    return global_ranked


# ── 单条记录预测 ───────────────────────────────────────────────────────────────

def predict_one(
    record: dict,
    model,
    tokenizer,
    pos_label_id: int,
    neg_label_id: int,
    max_seq_len: int,
    batch_size: int,
    device: str,
) -> dict:
    """
    输入一条预测记录，对每个CC下的候选citations打分排序。
    cc_text由split_sentences分句、extract_citations_from_text提取候选。
    输入格式见文件底部 INPUT FORMAT。
    """
    query      = record["query"]
    cc_results = []

    for cc in record["cc_list"]:
        # 分句
        sentences = split_sentences(cc["cc_text"])   # list[str]

        # 提取候选citations，并定位每个citation在sentences中的首个anchor句
        citation_ids = extract_citations_from_text(cc["cc_text"])  # list[str]，去重保序
        candidates = []
        for cit_id in citation_ids:
            for sent_id, sent in enumerate(sentences):
                if cit_id in sent:
                    candidates.append({"citation_id": cit_id, "sent_id": sent_id})
                    break  # 取首次出现句即可

        if not candidates:
            cc_results.append({"cc_id": cc["cc_id"], "ranked_citations": []})
            continue

        # 构建所有prompt
        prompts = []
        for cand in candidates:
            evidence = build_evidence(sentences, cand["sent_id"])
            prompt   = build_prompt(query, evidence, cand["citation_id"]) + POS_LABEL
            prompts.append(prompt)

        # 分批打分
        all_scores = []
        for i in range(0, len(prompts), batch_size):
            scores = score_batch(
                model, tokenizer,
                prompts[i: i + batch_size],
                pos_label_id, neg_label_id,
                max_seq_len, device,
            )
            all_scores.extend(scores)

        # 排序
        ranked = sorted(zip(candidates, all_scores), key=lambda x: x[1], reverse=True)
        cc_results.append({
            "cc_id": cc["cc_id"],
            "ranked_citations": [
                {
                    "rank":        rank + 1,
                    "citation_id": cand["citation_id"],
                    "sent_id":     cand["sent_id"],
                    "score":       round(score, 6),
                }
                for rank, (cand, score) in enumerate(ranked)
            ],
        })

    return {
        "query_id":               record["query_id"],
        "query":                  query,
        "cc_results":             cc_results,             # 各CC内部排序（原始分数）
        "global_ranked_citations": build_global_ranking(cc_results),  # 跨CC全局排序
    }


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Swiss法律引用排序推理（普通LoRA，无量化）")
    parser.add_argument("--base_model_path", required=True,
                        help="base model路径或HuggingFace名称")
    parser.add_argument("--adapter_path",    default=None,
                        help="LoRA adapter目录（含adapter_config.json）；不传则直接用base model")
    parser.add_argument("--input_file",   required=True,  help="预测输入JSONL路径")
    parser.add_argument("--output_file",  required=True,  help="预测输出JSONL路径")
    parser.add_argument("--batch_size",   type=int, default=16,       help="批量打分的batch size")
    parser.add_argument("--max_seq_len",  type=int, default=1024,     help="最大token长度，需与训练一致")
    parser.add_argument("--window",       type=int, default=2,        help="evidence窗口，需与训练一致")
    parser.add_argument("--torch_dtype",  default="bfloat16",
                        help="模型精度：bfloat16 / float16 / float32")
    args = parser.parse_args()

    global WINDOW
    WINDOW = args.window

    # 加载模型（device在函数内部确定并返回）
    model, tokenizer, device = load_model_and_tokenizer(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        torch_dtype=args.torch_dtype,
    )
    print(f"使用设备：{device}")

    pos_label_id = tokenizer.encode(POS_LABEL,  add_special_tokens=False)[0]
    neg_label_id = tokenizer.encode(NEG_LABEL,  add_special_tokens=False)[0]
    print(f"POS token id={pos_label_id} ({POS_LABEL!r}), NEG token id={neg_label_id} ({NEG_LABEL!r})")

    # 读取输入
    records = []
    with open(args.input_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"共 {len(records)} 条预测记录")

    # 预测并写出
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fout:
        for rec in tqdm(records, desc="预测中"):
            result = predict_one(
                rec, model, tokenizer,
                pos_label_id=pos_label_id,
                neg_label_id=neg_label_id,
                max_seq_len=args.max_seq_len,
                batch_size=args.batch_size,
                device=device,
            )
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"完成。结果写入：{output_path}")


# ────────────────────────────────────────────────────────────────────────────
# INPUT FORMAT（你需要对齐的输入格式）
# ────────────────────────────────────────────────────────────────────────────
#
# 文件格式：JSONL，每行一条，与 build_dataset.py 的输入格式一致（去掉gold_citations）：
#
# {
#   "query_id": "q_test_001",
#   "query":    "Welche Voraussetzungen gelten für die Kündigung?",
#   "cc_list": [
#     {
#       "cc_id":   "BGE_129_III_335",
#       "cc_text": "Das Bundesgericht hat... Art. 335 OR... BGE 110 II 99..."
#     },
#     {
#       "cc_id":   "BGE_110_II_99",
#       "cc_text": "..."
#     }
#   ]
# }
#
# 说明：
#   cc_text 为CC原始文本，分句和candidate citations提取均由代码自动完成：
#     - split_sentences(cc_text)            → sentences列表
#     - extract_citations_from_text(cc_text)→ citation_id列表（去重保序）
#   不需要手动提供 cc_sentences 或 candidate_citations。
#
# ────────────────────────────────────────────────────────────────────────────
# OUTPUT FORMAT
# ────────────────────────────────────────────────────────────────────────────
#
# {
#   "query_id": "q_test_001",
#   "query":    "Welche Voraussetzungen...",
#   "cc_results": [                          // 各CC内部排序（原始分数，仅CC内可比）
#     {
#       "cc_id": "BGE_129_III_335",
#       "ranked_citations": [
#         {"rank": 1, "citation_id": "Art. 335 OR",  "sent_id": 1, "score":  0.83},
#         {"rank": 2, "citation_id": "BGE 110 II 99","sent_id": 8, "score":  0.31},
#         {"rank": 3, "citation_id": "Art. 336 OR",  "sent_id": 3, "score": -0.22}
#       ]
#     }
#   ],
#   "global_ranked_citations": [             // 跨CC全局排序（softmax归一化后可比）
#     {"global_rank": 1, "citation_id": "Art. 335 OR",  "cc_id": "BGE_129_III_335",
#      "sent_id": 1, "raw_score": 0.83, "normed_score": 0.512},
#     {"global_rank": 2, "citation_id": "BGE 110 II 99","cc_id": "BGE_129_III_335",
#      "sent_id": 8, "raw_score": 0.31, "normed_score": 0.384},
#     ...
#   ]
# }
#
# cc_results       : 各CC内部排序，raw score仅在同一CC内可比
# global_ranked_citations : 跨CC全局排序，每个CC内softmax归一化后合并排序；
#                           同一citation_id在多个CC出现时保留normed_score最高的那条
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()