"""
train_lora.py
-------------
用LoRA对decode-only模型做pairwise ranking微调（Swiss法律引用检索）。
纯bf16/fp16训练，不使用bitsandbytes量化。

训练目标：给定query，对CC中的citations排序。
损失函数：Pairwise margin ranking loss（基于LM的sequence log-prob）。

Prompt格式（单条正/负样本的评分输入）：
  <s>[INST] Frage: {query}
  Beleg: {evidence}
  Zitat: {citation_id}
  Ist dieses Zitat relevant? [/INST] Ja </s>

正样本 target token = "Ja"，负样本 target token = "Nein"。
Pairwise loss = max(0, margin - log_p(Ja|pos) + log_p(Ja|neg))
"""

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    get_cosine_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
import wandb
from tqdm import tqdm

import os
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

    
from citation_utils import extract_citations_from_text


# ── 参数定义 ──────────────────────────────────────────────────────────────────

@dataclass
class DataArguments:
    train_file:    str = field(default="../data/ml6/train.jsonl")
    dev_file:      str = field(default="../data/ml6/dev.jsonl")
    dev_raw_file:  str = field(default="../data/ml6/dev_raw.jsonl",
                               metadata={"help": "原始格式dev集（与build_dataset输入格式相同），用于F1评估"})
    max_seq_len:   int = field(default=1024,  metadata={"help": "单条prompt最大token数"})
    hard_neg_only: bool= field(default=False,metadata={"help": "只用hard negative训练"})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="mistralai/Mistral-7B-Instruct-v0.2")
    torch_dtype: str = field(default="bfloat16", metadata={"help": "模型精度：bfloat16 / float16 / float32"})
    # LoRA
    lora_r:      int   = field(default=16)
    lora_alpha:  int   = field(default=32)
    lora_dropout:float = field(default=0.05)
    lora_target_modules: str = field(
        default="q_proj,v_proj,k_proj,o_proj",
        metadata={"help": "逗号分隔的target modules"}
    )


@dataclass
class TrainArguments:
    output_dir:          str   = field(default="../data/ml6/checkpoints")
    num_train_epochs:    int   = field(default=3)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size:  int = field(default=8)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate:       float = field(default=2e-4)
    warmup_ratio:        float = field(default=0.05)
    weight_decay:        float = field(default=0.01)
    margin:              float = field(default=1.0,  metadata={"help": "Pairwise margin"})
    hard_neg_weight:     float = field(default=2.0,  metadata={"help": "hard neg loss权重"})
    logging_steps:       int   = field(default=50)
    eval_steps:          int   = field(default=200)
    save_steps:          int   = field(default=200)
    save_total_limit:    int   = field(default=3)
    fp16:                bool  = field(default=False)
    bf16:                bool  = field(default=True)
    dataloader_num_workers: int = field(default=2)
    report_to:           str   = field(default="none")  # "wandb" 或 "none"
    seed:                int   = field(default=42)


# ── Prompt构建 ────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = (
    "<s>[INST] Frage: {query}\n"
    "Beleg: {evidence}\n"
    "Zitat: {citation_id}\n"
    "Ist dieses Zitat relevant? [/INST] "
)
POS_LABEL = "Ja"
NEG_LABEL = "Nein"


def build_prompt(query: str, evidence: str, citation_id: str) -> str:
    return PROMPT_TEMPLATE.format(
        query=query,
        evidence=evidence,
        citation_id=citation_id,
    )


# ── Dataset ───────────────────────────────────────────────────────────────────

class PairwiseDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        max_seq_len: int = 512,
        hard_neg_only: bool = False,
    ):
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        self.pairs       = []

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if hard_neg_only and item.get("neg_source") != "hard":
                    continue
                self.pairs.append(item)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        pos_prompt = build_prompt(p["query"], p["pos_evidence"], p["pos_citation_id"])
        neg_prompt = build_prompt(p["query"], p["neg_evidence"], p["neg_citation_id"])
        return {
            "pos_prompt":  pos_prompt,
            "neg_prompt":  neg_prompt,
            "neg_source":  p.get("neg_source", "random"),
        }


def collate_fn(batch, tokenizer, max_seq_len, pos_label_id, neg_label_id):
    """
    返回:
      pos_input_ids, pos_attention_mask   → (B, L)
      neg_input_ids, neg_attention_mask   → (B, L)
      is_hard                             → (B,) bool
    """
    pos_texts = [b["pos_prompt"] for b in batch]   # 不拼label，取[/INST]后第一个token的分布
    neg_texts = [b["neg_prompt"] for b in batch]   # 同上
    is_hard   = torch.tensor([b["neg_source"] == "hard" for b in batch], dtype=torch.bool)

    def tokenize(texts):
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"]

    pos_ids, pos_mask = tokenize(pos_texts)
    neg_ids, neg_mask = tokenize(neg_texts)

    return {
        "pos_input_ids":      pos_ids,
        "pos_attention_mask": pos_mask,
        "neg_input_ids":      neg_ids,
        "neg_attention_mask": neg_mask,
        "is_hard":            is_hard,
    }


# ── 评分函数：取目标token的log-prob ───────────────────────────────────────────

def get_last_token_logprob(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_token_id: int,
) -> torch.Tensor:
    """
    对每条样本，取最后一个位置的logit，返回target_token_id对应的log-prob。
    shape: (B,)
    训练时正常走梯度图；eval时由调用方的 @torch.no_grad() 负责关闭梯度。
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # logits: (B, L, V)
    last_logits = outputs.logits[:, -1, :]          # (B, V)
    log_probs   = F.log_softmax(last_logits, dim=-1) # (B, V)
    return log_probs[:, target_token_id]             # (B,)


# ── Pairwise Loss ─────────────────────────────────────────────────────────────

def pairwise_loss(
    pos_logprob: torch.Tensor,
    neg_logprob: torch.Tensor,
    is_hard: torch.Tensor,
    margin: float = 1.0,
    hard_weight: float = 2.0,
) -> torch.Tensor:
    """
    loss_i = max(0, margin - pos_logprob_i + neg_logprob_i)
    hard negatives乘以hard_weight加权。
    """
    raw_loss = F.relu(margin - pos_logprob + neg_logprob)   # (B,)
    weights  = torch.where(is_hard,
                           torch.full_like(raw_loss, hard_weight),
                           torch.ones_like(raw_loss))
    return (raw_loss * weights).mean()


# ── Evaluation：Macro F1@K ────────────────────────────────────────────────────

def _build_evidence_for_eval(sentences: list[str], cit_id: str, window: int = 2) -> str:
    """在sentences中找cit_id首次出现句，构建evidence窗口。"""
    for i, s in enumerate(sentences):
        if cit_id in s:
            start = max(0, i - window)
            end   = min(len(sentences) - 1, i + window)
            return " ".join(sentences[start: end + 1])
    return " ".join(sentences[:min(window * 2 + 1, len(sentences))])


@torch.no_grad()
def evaluate_f1(
    model,
    tokenizer,
    dev_raw_path: str,
    max_seq_len: int,
    pos_label_id: int,
    neg_label_id: int,
    device: str,
    k: int = 25,
    max_samples: int = 200,
    eval_batch_size: int = 24,
    max_cit_per_cc: int = 100,
) -> float:
    """
    Macro F1@K，从原始格式 dev 集（build_dataset 的输入格式）读取：
      - 候选池：用 extract_citations_from_text 从 CC 全文提取所有 citations
      - gold集：record["gold_citations"]（字符串列表）
    对每个 (query, CC) pair 算 F1@K，最后宏平均。

    显存控制参数：
      eval_batch_size : 打分时的batch大小，显存不足时调小
      max_cit_per_cc  : 单个CC最多保留的候选citation数（gold优先保留）

    F1 = 2*P*R/(P+R)
      P = |pred_top_k ∩ gold| / k
      R = |pred_top_k ∩ gold| / |gold|
    """
    model.eval()

    f1_scores = []
    count = 0

    with open(dev_raw_path, encoding="utf-8") as f:
        for line in f:
            if count >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            query    = rec["query"]
            gold_ids = set(rec["gold_citations"])

            for cc in rec["cc_list"]:
                sentences = [s["text"] for s in cc["cc_sentences"]]
                full_text = " ".join(sentences)

                cit_ids = extract_citations_from_text(full_text)
                if not cit_ids or not gold_ids:
                    continue

                # 只保留在本CC中实际出现的gold citations
                local_gold = {g for g in gold_ids if any(g in s for s in sentences)}
                if not local_gold:
                    continue

                # 限制候选数：gold全部保留，其余随机截断到 max_cit_per_cc
                if len(cit_ids) > max_cit_per_cc:
                    gold_in_pool  = [c for c in cit_ids if c in local_gold]
                    other_in_pool = [c for c in cit_ids if c not in local_gold]
                    max_other = max_cit_per_cc - len(gold_in_pool)
                    cit_ids = gold_in_pool + other_in_pool[:max_other]

                # 构建prompt，按字符长度升序排列以减少batch内padding浪费
                prompts_with_idx = [
                    (i, build_prompt(query, _build_evidence_for_eval(sentences, cid), cid) + POS_LABEL)
                    for i, cid in enumerate(cit_ids)
                ]
                prompts_with_idx.sort(key=lambda x: len(x[1]))
                sorted_indices = [x[0] for x in prompts_with_idx]
                sorted_prompts = [x[1] for x in prompts_with_idx]

                sorted_scores = []
                for i in range(0, len(sorted_prompts), eval_batch_size):
                    enc = tokenizer(
                        sorted_prompts[i: i + eval_batch_size],
                        padding=True,
                        truncation=True,
                        max_length=max_seq_len,
                        return_tensors="pt",
                    ).to(device)
                    outputs   = model(**enc)
                    logits    = outputs.logits[:, -1, :].float()
                    log_probs = F.log_softmax(logits, dim=-1)
                    scores    = (log_probs[:, pos_label_id] - log_probs[:, neg_label_id]).cpu()
                    sorted_scores.extend(scores.tolist())
                    del enc, outputs, logits, log_probs, scores
                    torch.cuda.empty_cache()

                # 还原原始顺序（保证all_scores[i]对应cit_ids[i]）
                all_scores = [0.0] * len(cit_ids)
                for rank_pos, orig_idx in enumerate(sorted_indices):
                    all_scores[orig_idx] = sorted_scores[rank_pos]

                # top-K预测集合
                top_k    = min(k, len(cit_ids))
                order    = sorted(range(len(all_scores)), key=lambda i: all_scores[i], reverse=True)
                pred_ids = {cit_ids[i] for i in order[:top_k]}

                tp        = len(pred_ids & local_gold)
                precision = tp / top_k
                recall    = tp / len(local_gold)
                f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
                f1_scores.append(f1)

            count += 1

    torch.cuda.empty_cache()
    model.train()
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


# ── 主训练循环 ────────────────────────────────────────────────────────────────

def main():
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainArguments))
    data_args, model_args, train_args = parser.parse_args_into_dataclasses()

    torch.manual_seed(train_args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ── Tokenizer ──────────────────────────────────────────────────────────
    print("加载tokenizer…")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, use_fast=True
        )
        print("  fast tokenizer加载成功")
    except Exception as e:
        print(f"  fast tokenizer加载失败（{e}），回退到slow tokenizer…")
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, use_fast=False
        )
        print("  slow tokenizer加载成功")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # decode-only用left padding

    pos_label_id = tokenizer.encode(POS_LABEL, add_special_tokens=False)[0]
    neg_label_id = tokenizer.encode(NEG_LABEL, add_special_tokens=False)[0]
    print(f"  POS token id={pos_label_id} ({POS_LABEL}), NEG token id={neg_label_id} ({NEG_LABEL})")

    # ── 模型加载 ───────────────────────────────────────────────────────────
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(model_args.torch_dtype, torch.bfloat16)
    print(f"加载模型：{model_args.model_name_or_path}  dtype={model_args.torch_dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map={"": device},
        trust_remote_code=True,
    )

    # ── LoRA ───────────────────────────────────────────────────────────────
    target_modules = [m.strip() for m in model_args.lora_target_modules.split(",")]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset & DataLoader ───────────────────────────────────────────────
    print("构建Dataset…")
    train_ds = PairwiseDataset(
        data_args.train_file, tokenizer,
        data_args.max_seq_len, data_args.hard_neg_only
    )
    dev_ds = PairwiseDataset(
        data_args.dev_file, tokenizer,
        data_args.max_seq_len, hard_neg_only=False
    )

    _collate = lambda batch: collate_fn(
        batch, tokenizer, data_args.max_seq_len, pos_label_id, neg_label_id
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_args.per_device_train_batch_size,
        shuffle=True,
        num_workers=train_args.dataloader_num_workers,
        collate_fn=_collate,
    )

    # ── Optimizer & Scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_args.learning_rate,
        weight_decay=train_args.weight_decay,
    )
    total_steps = (
        math.ceil(len(train_loader) / train_args.gradient_accumulation_steps)
        * train_args.num_train_epochs
    )
    warmup_steps = int(total_steps * train_args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Logging ────────────────────────────────────────────────────────────
    if train_args.report_to == "wandb":
        wandb.init(project="swiss-legal-rerank", config={
            **vars(data_args), **vars(model_args), **vars(train_args)
        })

    output_dir = Path(train_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 训练 ───────────────────────────────────────────────────────────────
    global_step  = 0
    best_f1      = 0.0
    accum_loss   = 0.0
    optimizer.zero_grad()

    for epoch in range(train_args.num_train_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_args.num_train_epochs}")

        for step, batch in enumerate(pbar):
            pos_ids   = batch["pos_input_ids"].to(device)
            pos_mask  = batch["pos_attention_mask"].to(device)
            neg_ids   = batch["neg_input_ids"].to(device)
            neg_mask  = batch["neg_attention_mask"].to(device)
            is_hard   = batch["is_hard"].to(device)

            pos_lp = get_last_token_logprob(model, pos_ids, pos_mask, pos_label_id)
            neg_lp = get_last_token_logprob(model, neg_ids, neg_mask, pos_label_id)  # 同样取Ja的概率

            loss = pairwise_loss(
                pos_lp, neg_lp, is_hard,
                margin=train_args.margin,
                hard_weight=train_args.hard_neg_weight,
            )
            loss = loss / train_args.gradient_accumulation_steps
            loss.backward()
            accum_loss += loss.item()

            if (step + 1) % train_args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                pbar.set_postfix({"loss": f"{accum_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

                if global_step % train_args.logging_steps == 0:
                    log_data = {"train/loss": accum_loss, "train/lr": scheduler.get_last_lr()[0]}
                    if train_args.report_to == "wandb":
                        wandb.log(log_data, step=global_step)

                accum_loss = 0.0

                # ── Eval ───────────────────────────────────────────────
                if global_step % train_args.eval_steps == 0:
                    f1 = evaluate_f1(
                        model, tokenizer, data_args.dev_raw_file,
                        data_args.max_seq_len, pos_label_id, neg_label_id, device
                    )
                    print(f"\nStep {global_step} | Macro F1@25 = {f1:.4f}")
                    if train_args.report_to == "wandb":
                        wandb.log({"eval/MacroF1@25": f1}, step=global_step)
                    if f1 > best_f1:
                        best_f1 = f1
                        model.save_pretrained(output_dir / "best")
                        tokenizer.save_pretrained(output_dir / "best")
                        print(f"  ★ 保存best model（Macro F1@25={best_f1:.4f}）")

                # ── Checkpoint ────────────────────────────────────────
                if global_step % train_args.save_steps == 0:
                    ckpt_path = output_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(ckpt_path)
                    tokenizer.save_pretrained(ckpt_path)

    print(f"\n训练完成。Best Macro F1@25 = {best_f1:.4f}")
    # 保存最终
    model.save_pretrained(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")


if __name__ == "__main__":
    main()