"""
predict_encoder.py
------------------
用训练好的 EncoderRanker 对query下所有CC中的citations打分排序。
评估逻辑与 train_encoder.py 的 evaluate_f1 完全一致：
  - 全局候选池：跨CC合并去重citations
  - 输入格式：cc_text字段，分句和citation提取由代码完成

用法：
    python predict_encoder.py \
        --model_dir  ../data/ml6/checkpoints_encoder/best \
        --base_model /path/to/xlm-roberta-large \
        --input_file  data/predict_input.jsonl \
        --output_file data/predict_output_encoder.jsonl
"""

import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import sys

src_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)
from citation_utils import split_sentences, extract_citations_from_text


WINDOW = 2


# ── 模型定义（与train_encoder.py完全一致）─────────────────────────────────────

class EncoderRanker(nn.Module):
    def __init__(self, encoder, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.scorer  = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs    = self.encoder(**kwargs)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        cls_hidden = self.dropout(cls_hidden)
        return self.scorer(cls_hidden).squeeze(-1)


# ── 工具函数（与train_encoder.py完全一致）────────────────────────────────────

def _build_evidence_for_eval(sentences: list[str], cit_id: str, window: int = WINDOW) -> str:
    for i, s in enumerate(sentences):
        if cit_id in s:
            start = max(0, i - window)
            end   = min(len(sentences) - 1, i + window)
            return " ".join(sentences[start: end + 1])
    return " ".join(sentences[:min(window * 2 + 1, len(sentences))])


# def build_input(query: str, evidence: str, citation_id: str,
#                 sep_token: str = "[SEP]") -> tuple[str, str]:
#     """[CLS] query [SEP] evidence [SEP] citation_id [SEP]"""
#     return query, f"{evidence} {sep_token} {citation_id}"

def build_input(query: str, evidence: str, citation_id: str,
                sep_token: str = "[SEP]") -> tuple[str, str]:
    text_a = f"{query} {sep_token} {evidence}"   # query和evidence合并为segment A
    text_b = citation_id                          # citation_id独立为segment B
    return text_a, text_b

# ── 模型加载 ──────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_dir: str, base_model: str, torch_dtype: str = "bfloat16"):
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    t_dtype   = dtype_map.get(torch_dtype, torch.bfloat16)
    device    = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"加载tokenizer：{model_dir}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)

    print(f"加载encoder：{base_model}")
    encoder = AutoModel.from_pretrained(
        base_model, torch_dtype=t_dtype,
        attn_implementation="sdpa", trust_remote_code=True,
    )
    hidden_size = encoder.config.hidden_size

    ckpt_path = Path(model_dir) / "ranker.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ranker.pt 不存在于 {model_dir}")
    print(f"加载ranker权重：{ckpt_path}")
    ckpt    = torch.load(ckpt_path, map_location="cpu")
    dropout = ckpt.get("dropout", 0.1)
    model   = EncoderRanker(encoder, hidden_size, dropout)
    model.load_state_dict(ckpt["model_state_dict"])
    model   = model.to(device, dtype=t_dtype)
    model.eval()

    print(f"  加载完成  device={device}  dtype={torch_dtype}")
    return model, tokenizer, device


# ── 主预测函数（Step逻辑与evaluate_f1完全对齐）────────────────────────────────

@torch.no_grad()
def predict_one(
    record:         dict,
    model,
    tokenizer,
    max_seq_len:    int,
    batch_size:     int,
    device:         str,
    top_k:          int = 25,
    topn_cc:        int = 50,
    max_cand:       int = 1000,
    max_cit_per_cc: int = 50,
    sep_token:      str = "[SEP]",
) -> dict:
    query   = record["query"]
    cc_list = record["cc_list"]

    # Step 1：按rerank_score排序，取前topn_cc
    if cc_list and "rerank_score" in cc_list[0]:
        cc_list = sorted(cc_list, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    cc_list = cc_list[:topn_cc]

    # Step 2：跨CC收集全局候选池，同一citation只保留首次出现的evidence
    cand_dict: dict[str, dict] = {}

    for cc in cc_list:
        if "cc_text" in cc:
            sentences = split_sentences(cc["cc_text"])
        else:
            sentences = [s["text"] for s in cc["cc_sentences"]]

        cit_ids = extract_citations_from_text(" ".join(sentences))
        if not cit_ids:
            continue
        if len(cit_ids) > max_cit_per_cc:
            cit_ids = cit_ids[:max_cit_per_cc]

        for cit_id in cit_ids:
            if cit_id not in cand_dict:
                sent_id = next(
                    (idx for idx, s in enumerate(sentences) if cit_id in s), 0
                )
                cand_dict[cit_id] = {
                    "evidence": _build_evidence_for_eval(sentences, cit_id),
                    "cc_id":    cc["cc_id"],
                    "sent_id":  sent_id,
                }

    if not cand_dict:
        return {"query_id": record["query_id"], "query": query,
                "global_ranked_citations": [], "top_k_citations": []}

    # Step 3：控制候选总数
    cand_items = list(cand_dict.items())   # [(cit_id, info_dict)]
    if len(cand_items) > max_cand:
        cand_items = cand_items[:max_cand]
    cit_ids = [cid for cid, _ in cand_items]

    # Step 4：构建sentence pairs，按长度排序减少padding
    pairs_with_idx = [
        (i, build_input(query, info["evidence"], cid, sep_token))
        for i, (cid, info) in enumerate(cand_items)
    ]
    pairs_with_idx.sort(key=lambda x: len(x[1][0]) + len(x[1][1]))
    sorted_indices = [x[0] for x in pairs_with_idx]
    sorted_pairs   = [x[1] for x in pairs_with_idx]

    # Step 5：批量推理
    sorted_scores = []
    for i in range(0, len(sorted_pairs), batch_size):
        batch   = sorted_pairs[i: i + batch_size]
        batch_a = [p[0] for p in batch]
        batch_b = [p[1] for p in batch]
        enc = tokenizer(
            batch_a, batch_b,
            padding=True, truncation=True,
            max_length=max_seq_len, return_tensors="pt",
        ).to(device)
        scores = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            token_type_ids=enc.get("token_type_ids"),
        ).view(-1).cpu().tolist()
        sorted_scores.extend(scores)
        del enc, scores
        torch.cuda.empty_cache()

    # Step 6：还原顺序，全局排序
    all_scores = [0.0] * len(cand_items)
    for rank_pos, orig_idx in enumerate(sorted_indices):
        all_scores[orig_idx] = sorted_scores[rank_pos]

    k     = min(top_k, len(cit_ids))
    order = sorted(range(len(all_scores)), key=lambda idx: all_scores[idx], reverse=True)

    global_ranked = [
        {
            "rank":        rank + 1,
            "citation_id": cit_ids[orig_idx],
            "cc_id":       cand_dict[cit_ids[orig_idx]]["cc_id"],
            "sent_id":     cand_dict[cit_ids[orig_idx]]["sent_id"],
            "score":       round(all_scores[orig_idx], 6),
        }
        for rank, orig_idx in enumerate(order)
    ]

    return {
        "query_id":                record["query_id"],
        "query":                   query,
        "global_ranked_citations": global_ranked,      # 全部候选的完整排序
        "top_k_citations":         global_ranked[:k],  # 精简版top-K
    }


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EncoderRanker推理（全局候选池）")
    parser.add_argument("--model_dir",      required=True)
    parser.add_argument("--base_model",     required=True)
    parser.add_argument("--input_file",     required=True)
    parser.add_argument("--output_file",    required=True)
    parser.add_argument("--batch_size",     type=int, default=32)
    parser.add_argument("--max_seq_len",    type=int, default=512)
    parser.add_argument("--top_k",          type=int, default=40,
                        help="输出top-K citations")
    parser.add_argument("--topn_cc",        type=int, default=50,
                        help="只取rerank_score最高的前N个CC")
    parser.add_argument("--max_cand",       type=int, default=1000,
                        help="全局候选池最大数量")
    parser.add_argument("--max_cit_per_cc", type=int, default=50,
                        help="每个CC最多提取的citations数")
    parser.add_argument("--torch_dtype",    default="bfloat16")
    args = parser.parse_args()

    model, tokenizer, device = load_model_and_tokenizer(
        args.model_dir, args.base_model, args.torch_dtype
    )
    sep_token = tokenizer.sep_token or "[SEP]"
    print(f"sep_token={sep_token!r}")

    records = []
    with open(args.input_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"共 {len(records)} 条预测记录")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fout:
        for rec in tqdm(records, desc="预测中"):
            result = predict_one(
                rec, model, tokenizer,
                max_seq_len=args.max_seq_len,
                batch_size=args.batch_size,
                device=device,
                top_k=args.top_k,
                topn_cc=args.topn_cc,
                max_cand=args.max_cand,
                max_cit_per_cc=args.max_cit_per_cc,
                sep_token=sep_token,
            )
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"完成。结果写入：{output_path}")


# ── INPUT / OUTPUT FORMAT ─────────────────────────────────────────────────────
#
# INPUT（每行一条）：
# {
#   "query_id": "q_test_001",
#   "query":    "Welche Voraussetzungen gelten für die Kündigung?",
#   "cc_list": [
#     {
#       "cc_id":        "BGE_129_III_335",
#       "rerank_score": 0.92,          // 可选，有则按score排序取topn_cc
#       "cc_text":      "..."          // 原始文本（优先）
#       // 或者："cc_sentences": [{"sent_id": 0, "text": "..."}, ...]
#     }
#   ]
# }
#
# OUTPUT（每行一条）：
# {
#   "query_id": "q_test_001",
#   "query":    "...",
#   "global_ranked_citations": [       // 所有候选的完整排序
#     {"rank": 1, "citation_id": "Art. 335 OR", "cc_id": "BGE_129_III_335",
#      "sent_id": 1, "score": 2.3456},
#     ...
#   ],
#   "top_k_citations": [               // top-K精简版（默认K=25）
#     {"rank": 1, "citation_id": "Art. 335 OR", ...},
#     ...
#   ]
# }
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()