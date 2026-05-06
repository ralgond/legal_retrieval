"""
train_encoder.py
----------------
用Encode-only模型（BERT/RoBERTa/XLM-R等）做pairwise ranking微调（Swiss法律引用检索）。
数据格式与train_qlora.py完全相同，无需重新构建数据集。

打分方式：linear([CLS]) → scalar score
  - 在[CLS]表示上接一个线性层输出单个相关性分数
  - 不依赖next token prediction，天然适合分类/排序任务

Prompt格式（单条正/负样本的输入）：
  [CLS] {query} [SEP] {evidence} [SEP] {citation_id} [SEP]

损失函数：Pairwise margin ranking loss
  loss = max(0, margin - score(pos) + score(neg))
  hard negatives乘以hard_neg_weight加权
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
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
    train_file:    str  = field(default="../data/ml6/train.jsonl")
    dev_file:      str  = field(default="../data/ml6/dev.jsonl")
    dev_raw_file:  str  = field(default="../data/ml6/dev_raw.jsonl",
                                metadata={"help": "原始格式dev集，用于F1评估"})
    max_seq_len:   int  = field(default=1024,  metadata={"help": "单条prompt最大token数"})
    hard_neg_only: bool = field(default=False, metadata={"help": "只用hard negative训练"})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="xlm-roberta-base",
                                    metadata={"help": "encode-only模型，推荐xlm-roberta-base/large"})
    torch_dtype: str = field(default="bfloat16",
                             metadata={"help": "模型精度：bfloat16 / float16 / float32"})
    dropout:     float = field(default=0.1, metadata={"help": "分类头dropout"})
    # LoRA（可选，encode-only模型参数少，也可以全量微调）
    use_lora:    bool  = field(default=False, metadata={"help": "是否使用LoRA，encode-only模型通常不需要"})
    lora_r:      int   = field(default=8)
    lora_alpha:  int   = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default="query,value",
                                     metadata={"help": "逗号分隔，BERT用query,value；RoBERTa用query,value"})


@dataclass
class TrainArguments:
    output_dir:          str   = field(default="../data/ml6/checkpoints_encoder")
    num_train_epochs:    int   = field(default=5)
    per_device_train_batch_size: int = field(default=16)
    gradient_accumulation_steps: int = field(default=2)
    learning_rate:       float = field(default=2e-5,
                                       metadata={"help": "encode-only微调lr通常比decode-only小一个数量级"})
    warmup_ratio:        float = field(default=0.05)
    weight_decay:        float = field(default=0.01)
    margin:              float = field(default=1.0,  metadata={"help": "Pairwise margin"})
    hard_neg_weight:     float = field(default=2.0,  metadata={"help": "hard/inter_cc_hard neg的loss权重"})
    logging_steps:       int   = field(default=50)
    eval_steps:          int   = field(default=1000)
    save_steps:          int   = field(default=10000000)
    save_total_limit:    int   = field(default=3)
    dataloader_num_workers: int = field(default=2)
    report_to:           str   = field(default="none")
    seed:                int   = field(default=42)
    early_stopping_patience: int = field(default=10,
                                         metadata={"help": "连续N次eval未提升则停止，0表示不启用"})
    eval_k: int = field(default=35, metadata={"help": "F1@K的K，应小于候选池平均大小"})


# ── Prompt构建 ────────────────────────────────────────────────────────────────
# encode-only不需要instruction格式，直接拼接三段文本
# tokenizer会自动在开头加[CLS]、段间加[SEP]

# def build_input(query: str, evidence: str, citation_id: str,
#                sep_token: str = "[SEP]") -> tuple[str, str]:
#     """
#     返回 (text_a, text_b) 供tokenizer的sentence-pair输入。
#     最终编码为：[CLS] query [SEP] evidence [SEP] citation_id [SEP]
#     其中第一个[SEP]由tokenizer的sentence-pair机制插入（text_a和text_b之间），
#     第二个[SEP]手动插入text_b内部以区分evidence和citation_id。
#     """
#     text_a = query
#     text_b = f"{evidence} {sep_token} {citation_id}"
#     return text_a, text_b

def build_input(query: str, evidence: str, citation_id: str,
                sep_token: str = "[SEP]") -> tuple[str, str]:
    text_a = f"{query} {sep_token} {evidence}"   # query和evidence合并为segment A
    text_b = citation_id                          # citation_id独立为segment B
    return text_a, text_b

# ── 模型定义 ──────────────────────────────────────────────────────────────────

class EncoderRanker(nn.Module):
    """
    Encode-only模型 + 线性打分头。
    取[CLS] token的表示，经过dropout后接线性层输出scalar分数。
    """
    def __init__(self, encoder, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.scorer  = nn.Linear(hidden_size, 1)
        # 初始化打分头，避免初始分数过大
        nn.init.xavier_uniform_(self.scorer.weight)
        nn.init.zeros_(self.scorer.bias)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs    = self.encoder(**kwargs)
        cls_hidden = outputs.last_hidden_state[:, 0, :]  # (B, H)
        cls_hidden = self.dropout(cls_hidden)
        score      = self.scorer(cls_hidden).squeeze(-1)  # (B,)
        return score


# ── Dataset ───────────────────────────────────────────────────────────────────

class PairwiseDataset(Dataset):
    def __init__(self, path: str, hard_neg_only: bool = False):
        self.pairs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if hard_neg_only and item.get("neg_source") not in ("hard", "inter_cc_hard"):
                    continue
                self.pairs.append(item)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        pos_a, pos_b = build_input(p["query"], p["pos_evidence"], p["pos_citation_id"])
        neg_a, neg_b = build_input(p["query"], p["neg_evidence"], p["neg_citation_id"])
        return {
            "pos_a":      pos_a,
            "pos_b":      pos_b,
            "neg_a":      neg_a,
            "neg_b":      neg_b,
            "neg_source": p.get("neg_source", "random"),
        }


def collate_fn(batch, tokenizer, max_seq_len):
    pos_a = [b["pos_a"] for b in batch]
    pos_b = [b["pos_b"] for b in batch]
    neg_a = [b["neg_a"] for b in batch]
    neg_b = [b["neg_b"] for b in batch]
    is_hard = torch.tensor(
        [b["neg_source"] in ("hard", "inter_cc_hard") for b in batch],
        dtype=torch.bool,
    )

    def tokenize(text_a, text_b):
        enc = tokenizer(
            text_a, text_b,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        return enc

    pos_enc = tokenize(pos_a, pos_b)
    neg_enc = tokenize(neg_a, neg_b)

    return {
        "pos_input_ids":      pos_enc["input_ids"],
        "pos_attention_mask": pos_enc["attention_mask"],
        "pos_token_type_ids": pos_enc.get("token_type_ids"),
        "neg_input_ids":      neg_enc["input_ids"],
        "neg_attention_mask": neg_enc["attention_mask"],
        "neg_token_type_ids": neg_enc.get("token_type_ids"),
        "is_hard":            is_hard,
    }


# ── Pairwise Loss（BPR）─────────────────────────────────────────────────────────
# 用 BPR loss 替代 margin ranking loss：
#   loss = -log(sigmoid(pos_score - neg_score))
# 优点：只要 pos_score != neg_score 就有梯度，不存在梯度消失问题；
#       不依赖 margin 超参，训练更稳定。
# hard negative 仍然通过 hard_weight 加权。

def pairwise_loss(
    pos_score:   torch.Tensor,
    neg_score:   torch.Tensor,
    is_hard:     torch.Tensor,
    margin:      float = 1.0,   # 保留参数兼容命令行，BPR模式下不使用
    hard_weight: float = 2.0,
) -> torch.Tensor:
    raw_loss = -F.logsigmoid(pos_score - neg_score)     # (B,)
    weights  = torch.where(is_hard,
                           torch.full_like(raw_loss, hard_weight),
                           torch.ones_like(raw_loss))
    return (raw_loss * weights).mean()


# ── Evaluation：Macro F1@K ────────────────────────────────────────────────────

def _build_evidence_for_eval(sentences: list[str], cit_id: str, window: int = 2) -> str:
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
    max_seq_len:  int,
    device:       str,
    k:            int = 25,
    max_samples:  int = 200,
    eval_batch_size: int = 32,
    topn_cc:         int = 50,    # 只取rerank_score最高的前N个CC
    max_cand:        int = 1000,  # 全局候选池最大数量
    max_cit_per_cc:  int = 50,    # 每个CC最多保留多少citations
) -> float:
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
            rec      = json.loads(line)
            query    = rec["query"]
            gold_ids = set(rec["gold_citations"])

            # ── Step 1：按rerank_score排序，取前topn_cc ──────────────────
            cc_list = rec["cc_list"]
            if cc_list and "rerank_score" in cc_list[0]:
                cc_list = sorted(cc_list, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            cc_list = cc_list[:topn_cc]

            # ── Step 2：跨CC收集全局候选池，同一citation只保留首次evidence ──
            cand_dict: dict[str, str] = {}   # cit_id -> evidence
            for cc in cc_list:
                sentences = [s["text"] for s in cc["cc_sentences"]]
                cit_ids_cc = extract_citations_from_text(" ".join(sentences))
                if len(cit_ids_cc) > max_cit_per_cc:
                    cit_ids_cc = cit_ids_cc[:max_cit_per_cc]
                for cid in cit_ids_cc:
                    if cid not in cand_dict:
                        cand_dict[cid] = _build_evidence_for_eval(sentences, cid)

            if not cand_dict:
                count += 1
                continue

            # ── Step 3：控制候选总数，gold优先保留 ─────────────────────────
            cand_items = list(cand_dict.items())   # [(cit_id, evidence)]
            if len(cand_items) > max_cand:
                gold_part  = [(cid, ev) for cid, ev in cand_items if cid in gold_ids]
                other_part = [(cid, ev) for cid, ev in cand_items if cid not in gold_ids]
                cand_items = gold_part + other_part[:max(0, max_cand - len(gold_part))]

            cit_ids = [cid for cid, _ in cand_items]

            # ── Step 4：构建pairs，按长度排序减少padding ────────────────────
            pairs_with_idx = [
                (i, build_input(query, ev, cid))
                for i, (cid, ev) in enumerate(cand_items)
            ]
            pairs_with_idx.sort(key=lambda x: len(x[1][0]) + len(x[1][1]))
            sorted_indices = [x[0] for x in pairs_with_idx]
            sorted_pairs   = [x[1] for x in pairs_with_idx]

            # ── Step 5：批量推理 ─────────────────────────────────────────────
            sorted_scores = []
            for i in range(0, len(sorted_pairs), eval_batch_size):
                batch_a = [p[0] for p in sorted_pairs[i: i + eval_batch_size]]
                batch_b = [p[1] for p in sorted_pairs[i: i + eval_batch_size]]
                enc = tokenizer(
                    batch_a, batch_b,
                    padding=True, truncation=True,
                    max_length=max_seq_len, return_tensors="pt",
                ).to(device)
                batch_scores = model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    token_type_ids=enc.get("token_type_ids"),
                ).view(-1).cpu().tolist()
                sorted_scores.extend(batch_scores)
                del enc, batch_scores
                torch.cuda.empty_cache()

            # ── Step 6：还原顺序，全局排序 ───────────────────────────────────
            all_scores = [0.0] * len(cand_items)
            for rank_pos, orig_idx in enumerate(sorted_indices):
                all_scores[orig_idx] = sorted_scores[rank_pos]

            top_k    = min(k, len(cit_ids))
            order    = sorted(range(len(all_scores)), key=lambda idx: all_scores[idx], reverse=True)
            pred_ids = {cit_ids[j] for j in order[:top_k]}

            # ── Step 7：F1（gold与全局候选对比）────────────────────────────
            tp        = len(pred_ids & gold_ids)
            precision = tp / top_k if top_k > 0 else 0.0
            recall    = tp / len(gold_ids)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)

            # ── DEBUG ────────────────────────────────────────────────────────
            if len(f1_scores) <= 2:
                score_std  = torch.tensor(all_scores).std().item()
                gold_ranks = [order.index(cit_ids.index(g)) + 1
                              for g in gold_ids if g in cit_ids]
                print(f"  [DEBUG] n_cand={len(cit_ids)} | top_k={top_k} | "
                      f"score_range=[{min(all_scores):.4f}, {max(all_scores):.4f}] | "
                      f"std={score_std:.4f} | gold_ranks={gold_ranks}")

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
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
        print("  fast tokenizer加载成功")
    except Exception as e:
        print(f"  fast tokenizer失败（{e}），回退slow tokenizer…")
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False)

    # ── 模型加载 ───────────────────────────────────────────────────────────
    dtype_map  = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(model_args.torch_dtype, torch.bfloat16)
    print(f"加载模型：{model_args.model_name_or_path}  dtype={model_args.torch_dtype}")

    encoder = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    # ── 可选LoRA ───────────────────────────────────────────────────────────
    if model_args.use_lora:
        target_modules = [m.strip() for m in model_args.lora_target_modules.split(",")]
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        encoder = get_peft_model(encoder, lora_config)
        encoder.print_trainable_parameters()
    else:
        n_params = sum(p.numel() for p in encoder.parameters())
        print(f"  全量微调，参数量：{n_params/1e6:.1f}M")

    # ── 组装 EncoderRanker ─────────────────────────────────────────────────
    hidden_size = encoder.config.hidden_size
    model = EncoderRanker(encoder, hidden_size, model_args.dropout)
    model = model.to(device, dtype=torch.bfloat16)

    # ── Dataset & DataLoader ───────────────────────────────────────────────
    print("构建Dataset…")
    train_ds = PairwiseDataset(data_args.train_file, data_args.hard_neg_only)

    _collate = lambda batch: collate_fn(batch, tokenizer, data_args.max_seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_args.per_device_train_batch_size,
        shuffle=True,
        num_workers=train_args.dataloader_num_workers,
        collate_fn=_collate,
    )

    # ── Optimizer & Scheduler ──────────────────────────────────────────────
    # 打分头用更大的lr，encoder用较小的lr
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": train_args.learning_rate},
        {"params": model.scorer.parameters(),  "lr": train_args.learning_rate * 10},
    ], weight_decay=train_args.weight_decay)

    total_steps = (
        math.ceil(len(train_loader) / train_args.gradient_accumulation_steps)
        * train_args.num_train_epochs
    )
    warmup_steps = int(total_steps * train_args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Logging ────────────────────────────────────────────────────────────
    if train_args.report_to == "wandb":
        wandb.init(project="swiss-legal-rerank-encoder", config={
            **vars(data_args), **vars(model_args), **vars(train_args)
        })

    output_dir = Path(train_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 训练 ───────────────────────────────────────────────────────────────
    global_step        = 0
    best_f1            = 0.0
    accum_loss         = 0.0
    no_improvement_cnt = 0
    stop_training      = False
    optimizer.zero_grad()

    for epoch in range(train_args.num_train_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_args.num_train_epochs}")

        for step, batch in enumerate(pbar):
            pos_ids   = batch["pos_input_ids"].to(device)
            pos_mask  = batch["pos_attention_mask"].to(device)
            pos_ttype = batch["pos_token_type_ids"].to(device) if batch["pos_token_type_ids"] is not None else None
            neg_ids   = batch["neg_input_ids"].to(device)
            neg_mask  = batch["neg_attention_mask"].to(device)
            neg_ttype = batch["neg_token_type_ids"].to(device) if batch["neg_token_type_ids"] is not None else None
            is_hard   = batch["is_hard"].to(device)

            pos_score = model(pos_ids, pos_mask, pos_ttype)
            neg_score = model(neg_ids, neg_mask, neg_ttype)

            loss = pairwise_loss(
                pos_score, neg_score, is_hard,
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

                pbar.set_postfix({
                    "loss":      f"{accum_loss:.4f}",
                    "lr_enc":    f"{scheduler.get_last_lr()[0]:.3e}",
                    "lr_scorer": f"{scheduler.get_last_lr()[1]:.3e}",
                })

                if global_step % train_args.logging_steps == 0:
                    if train_args.report_to == "wandb":
                        wandb.log({"train/loss": accum_loss,
                                   "train/lr":   scheduler.get_last_lr()[0]},
                                  step=global_step)

                accum_loss = 0.0

                # ── Eval ───────────────────────────────────────────────
                if global_step % train_args.eval_steps == 0:
                    f1 = evaluate_f1(
                        model, tokenizer, data_args.dev_raw_file,
                        data_args.max_seq_len, device,
                        k=train_args.eval_k,
                    )
                    print(f"\nStep {global_step} | Macro F1@{train_args.eval_k} = {f1:.4f}")
                    if train_args.report_to == "wandb":
                        wandb.log({"eval/MacroF1@25": f1}, step=global_step)

                    if f1 > best_f1:
                        best_f1 = f1
                        no_improvement_cnt = 0
                        # 保存整个 EncoderRanker（encoder + scorer）
                        torch.save({
                            "model_state_dict": model.state_dict(),
                            "hidden_size":      hidden_size,
                            "dropout":          model_args.dropout,
                        }, output_dir / "best" / "ranker.pt")
                        tokenizer.save_pretrained(output_dir / "best")
                        (output_dir / "best").mkdir(parents=True, exist_ok=True)
                        # 同时保存encoder权重（方便后续加载）
                        if model_args.use_lora:
                            model.encoder.save_pretrained(output_dir / "best")
                        else:
                            model.encoder.save_pretrained(output_dir / "best" / "encoder")
                        print(f"  ★ 保存best model（Macro F1@{train_args.eval_k}={best_f1:.4f}）")
                    else:
                        no_improvement_cnt += 1
                        print(f"  无提升（{no_improvement_cnt}/{train_args.early_stopping_patience}）")
                        if train_args.early_stopping_patience > 0 and \
                                no_improvement_cnt >= train_args.early_stopping_patience:
                            print(f"\n★ Early stopping：连续{no_improvement_cnt}次eval未提升")
                            stop_training = True

                # ── Checkpoint ────────────────────────────────────────
                if global_step % train_args.save_steps == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    torch.save({"model_state_dict": model.state_dict(),
                                "hidden_size":      hidden_size,
                                "dropout":          model_args.dropout},
                               ckpt_dir / "ranker.pt")
                    tokenizer.save_pretrained(ckpt_dir)

                if stop_training:
                    break

        if stop_training:
            break

    print(f"\n训练完成。Best Macro F1@{train_args.eval_k} = {best_f1:.4f}")
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "hidden_size":      hidden_size,
                "dropout":          model_args.dropout},
               final_dir / "ranker.pt")
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()