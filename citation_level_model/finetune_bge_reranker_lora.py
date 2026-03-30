"""
LoRA 微调 bge-reranker-v2-m3
样本格式: {'query': str, 'passage': str, 'label': int (0 or 1)}

修复说明:
  原代码在 LoraConfig 中设置 modules_to_save=["classifier"]，
  PEFT 会把分类头包进 ModulesToSaveWrapper，导致后续再查找 target_modules
  (如 "dense") 时因为层级结构改变而抛出 AttributeError。
  解法: 不使用 modules_to_save，改为在 get_peft_model 之后手动解冻分类头。
"""

import json
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)
from sklearn.metrics import accuracy_score, roc_auc_score


# ──────────────────────────────────────────────
# 1. 参数配置
# ──────────────────────────────────────────────

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="BAAI/bge-reranker-v2-m3",
        metadata={"help": "预训练模型路径或 HuggingFace Hub ID"}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "最大序列长度"}
    )


@dataclass
class DataArguments:
    train_file: str = field(
        default="train.jsonl",
        metadata={"help": "训练集路径，每行一个 JSON 对象"}
    )
    eval_file: Optional[str] = field(
        default="eval.jsonl",
        metadata={"help": "验证集路径，可选"}
    )


@dataclass
class LoraArguments:
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    # bge-reranker-v2-m3 底层是 XLM-RoBERTa，attention 投影命名如下
    lora_target_modules: str = field(
        default="query,key,value,dense",
        metadata={"help": "逗号分隔的目标模块名，不含空格"}
    )


# ──────────────────────────────────────────────
# 2. 数据集
# ──────────────────────────────────────────────

class RerankerDataset(Dataset):
    def __init__(self, data_file: str, tokenizer, max_length: int):
        self.samples = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.samples.append(obj)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        query = item["query"]
        passage = item["passage"]
        label = int(item["label"])  # 0 or 1

        # bge-reranker 推荐格式: query + sep + passage
        encoding = self.tokenizer(
            query,
            passage,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            # num_labels=1 时 Transformers 内部用 BCEWithLogitsLoss，label 必须是 float
            "labels": torch.tensor(label, dtype=torch.float),
        }


# ──────────────────────────────────────────────
# 3. 评估指标
# ──────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # num_labels=1：logits shape 为 (batch, 1) 或 (batch,)
    logits = np.array(logits).squeeze(-1)          # → (batch,)
    pos_probs = 1 / (1 + np.exp(-logits))          # sigmoid
    preds = (pos_probs >= 0.5).astype(int)
    labels_int = labels.astype(int)

    acc = accuracy_score(labels_int, preds)
    try:
        auc = roc_auc_score(labels_int, pos_probs)
    except ValueError:
        auc = 0.0

    return {"accuracy": acc, "auc": auc}


# ──────────────────────────────────────────────
# 4. 主函数
# ──────────────────────────────────────────────

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, LoraArguments, TrainingArguments))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    # ---------- tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True,
    )

    # ---------- base model ----------
    # bge-reranker-v2-m3 原始分类头输出维度为 1（相关性打分），
    # 保持 num_labels=1，用 BCE loss 训练二分类，避免权重尺寸不匹配。
    #
    # ⚠️  必须用 torch.float32 加载模型，不能直接以 fp16 存储权重。
    # PyTorch AMP（fp16=True）的设计是"主权重 fp32、前向计算 fp16"：
    # GradScaler.unscale_() 要求优化器持有的参数是 fp32；
    # 若以 torch_dtype=float16 加载，参数本身是 fp16，
    # unscale_ 会抛出 "Attempting to unscale FP16 gradients"。
    # 正确做法：模型始终 fp32 加载，Trainer 设置 fp16=True 后
    # Accelerate 会在前向时自动 cast 并管理 GradScaler。
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=1,
        ignore_mismatched_sizes=False,   # 尺寸完全匹配，无需 ignore
        torch_dtype=torch.float32,       # 固定 fp32；混合精度交给 Trainer/Accelerate 管理
    )

    # ---------- LoRA 配置 ----------
    target_modules = [m.strip() for m in lora_args.lora_target_modules.split(",")]

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        # ⚠️  不要在这里写 modules_to_save=["classifier"]
        # PEFT 会将其包进 ModulesToSaveWrapper，导致后续找 target_modules
        # 中的子层（如 dense）时因层级变化抛出 AttributeError。
        # 分类头通过下面手动 requires_grad=True 的方式参与训练并由
        # Trainer 自动保存到 output_dir/pytorch_model.bin。
    )

    model = get_peft_model(model, lora_config)

    # 手动解冻分类头，使其与 LoRA 层一同更新
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True

    model.print_trainable_parameters()

    # ---------- 数据集 ----------
    train_dataset = RerankerDataset(data_args.train_file, tokenizer, model_args.max_length)
    eval_dataset = None
    if data_args.eval_file:
        eval_dataset = RerankerDataset(data_args.eval_file, tokenizer, model_args.max_length)

    # ---------- Trainer ----------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # ---------- 保存 LoRA 权重 ----------
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"\n✅ LoRA 权重已保存到: {training_args.output_dir}")


# ──────────────────────────────────────────────
# 5. 推理示例（训练完成后使用）
# ──────────────────────────────────────────────

def inference_example(lora_dir: str, base_model: str = "BAAI/bge-reranker-v2-m3"):
    tokenizer = AutoTokenizer.from_pretrained(lora_dir)
    # num_labels=1，与训练时保持一致
    base = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1)
    model = PeftModel.from_pretrained(base, lora_dir)
    model.eval()

    query = "什么是机器学习？"
    passages = [
        "机器学习是人工智能的一个分支，研究如何让计算机从数据中学习。",
        "今天天气很好，适合出门散步。",
    ]

    scores = []
    for p in passages:
        enc = tokenizer(query, p, return_tensors="pt", max_length=512,
                        truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**enc).logits          # shape: (1, 1)
        score = torch.sigmoid(logits).item()      # 相关性概率 [0, 1]
        scores.append(score)

    for p, s in zip(passages, scores):
        print(f"score={s:.4f}  passage: {p[:50]}")


if __name__ == "__main__":
    main()