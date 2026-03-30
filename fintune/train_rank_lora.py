"""
LoRA Pointwise Reranker 微调
任务：recall doc -> rerank doc -> extract law citation -> rank citation by doc score

软标签逻辑：doc 的标签由它包含的 law citation 质量决定
  - 正样本 doc：包含与 query 相关的目标 law citation
  - 软标签：citation 覆盖率 × citation 质量分的加权组合

依赖：
    pip install transformers peft accelerate bitsandbytes
"""

import json
import math
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class RerankerConfig:
    base_model: str = "/root/.cache/modelscope/hub/models/BAAI/bge-reranker-v2-m3"
    output_dir: str = "../ft_data/reranker-lora-output"
    max_length: int = 512

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list = field(
        default_factory=lambda: ["query", "value"]
    )

    # 数据
    train_file: str = "../ft_data/train.jsonl"
    val_file: str = "../ft_data/val.jsonl"

    # 软标签参数
    label_smoothing: float = 0.05
    # citation 质量归一化分母（p95 法条引用频次，用于压缩高频法条权重）
    citation_freq_p95: int = 200

    # 正样本权重（缓解正负不平衡，正负比约 1:6 时建议 5.0）
    pos_weight: float = 5.0

    # 训练
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    eval_steps: int = 200
    save_steps: int = 200
    bf16: bool = True


CFG = RerankerConfig()


# ---------------------------------------------------------------------------
# 软标签计算：doc 维度
# ---------------------------------------------------------------------------

class DocSoftLabelCalculator:
    """
    根据 doc 中包含的 law citation 信息，计算该 doc 的软标签。

    软标签 = coverage_score × quality_score，映射到 [0.5+eps, 1-eps]

    coverage_score：doc 中命中的目标 citation 数 / 查询期望的总 citation 数
                    反映"这个 doc 有多完整地覆盖了我们想要的法条"

    quality_score：命中 citation 的加权质量均值
                   用 log(1 + freq) 压缩高频法条，避免 § 242 BGB 这类
                   超高频条款主导整个训练信号
    """

    def __init__(self, cfg: RerankerConfig):
        self.cfg = cfg

    def compute(self, sample: dict) -> float:
        """
        sample 字段说明：
          label              : 1.0 正样本 / 0.0 负样本（必须）
          matched_citations  : list，doc 中命中的目标 citation（可选）
              每个元素: {"citation_id": "§242BGB", "global_freq": 847}
          total_expected     : int，当前 query 期望的目标 citation 总数（可选）
        """
        eps = self.cfg.label_smoothing
        raw_label = float(sample.get("label", 0.0))

        # 负样本：固定低值，给一点 label smoothing 容忍边界模糊案例
        if raw_label < 0.5:
            return eps

        matched = sample.get("matched_citations", [])
        total_expected = sample.get("total_expected", 1)

        # 没有细粒度 citation 信息时退化为硬正样本
        if not matched or total_expected == 0:
            return 1.0 - eps

        # coverage：命中比例，上限 1.0
        coverage = min(1.0, len(matched) / max(1, total_expected))

        # quality：对每个命中 citation，用 log 压缩其全局引用频次
        quality_scores = []
        for c in matched:
            freq = c.get("global_freq", 1)
            q = math.log(1 + freq) / math.log(1 + self.cfg.citation_freq_p95)
            quality_scores.append(min(1.0, q))

        quality = sum(quality_scores) / len(quality_scores)

        # 综合分：覆盖率 × 质量均值，映射到 [0.5+eps, 1-eps]
        combined = coverage * quality
        soft = 0.5 + (0.5 - eps) * combined
        return max(0.5 + eps, min(1.0 - eps, soft))


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------

class DocRerankDataset(Dataset):
    """
    每条 JSONL 样本格式（完整版）：
    {
        "query": "法院考量段文本",
        "doc":   "候选判决段落文本",
        "label": 1.0,
        "matched_citations": [
            {"citation_id": "§ 280 Abs. 1 BGB", "global_freq": 312},
            {"citation_id": "§ 242 BGB",         "global_freq": 847}
        ],
        "total_expected": 3
    }

    最简格式（无 citation 细节时退化为硬标签）：
    {
        "query": "...",
        "doc":   "...",
        "label": 1.0
    }
    """

    def __init__(self, file_path: str, tokenizer, cfg: RerankerConfig):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.label_calc = DocSoftLabelCalculator(cfg)
        self.samples = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        pos = sum(1 for s in self.samples if float(s.get("label", 0)) > 0.5)
        logger.info(
            f"加载 {len(self.samples)} 条样本（正:{pos} 负:{len(self.samples)-pos}）"
            f" 来自 {file_path}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        encoding = self.tokenizer(
            sample["query"],
            sample["doc"],
            max_length=self.cfg.max_length,
            truncation="longest_first",   # 超长优先截断 doc，保留完整 query
            padding="max_length",
            return_tensors="pt",
        )

        soft_label = self.label_calc.compute(sample)

        item = {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(soft_label, dtype=torch.float),
        }
        # BERT 系模型有 token_type_ids，XLM-R 没有
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        return item


# ---------------------------------------------------------------------------
# 自定义 Trainer：BCE + 软标签 + 正样本权重
# ---------------------------------------------------------------------------

class SoftLabelTrainer(Trainer):
    def __init__(self, *args, pos_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos_weight_value = pos_weight
        self._loss_fn: Optional[nn.BCEWithLogitsLoss] = None

    def _get_loss_fn(self, device):
        if self._loss_fn is None:
            pw = torch.tensor([self._pos_weight_value], device=device)
            self._loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        return self._loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")          # (B,) float 软标签

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs.get("token_type_ids"),
        )

        logits = outputs.logits.squeeze(-1)    # (B,)
        loss = self._get_loss_fn(logits.device)(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# 评估指标
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score

    logits, labels = eval_pred
    logits = logits.squeeze(-1) if logits.ndim == 2 else logits
    probs = 1 / (1 + np.exp(-logits))             # sigmoid
    binary_labels = (labels > 0.5).astype(int)

    try:
        auc = roc_auc_score(binary_labels, probs)
        ap  = average_precision_score(binary_labels, probs)
    except ValueError:
        auc, ap = 0.0, 0.0

    acc = ((probs > 0.5).astype(int) == binary_labels).mean()
    return {"auc": float(auc), "ap": float(ap), "acc": float(acc)}


# ---------------------------------------------------------------------------
# 模型初始化
# ---------------------------------------------------------------------------

def build_model_and_tokenizer(cfg: RerankerConfig):
    logger.info(f"加载基座模型：{cfg.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model,
        num_labels=1,                   # sigmoid 输出，配合 BCEWithLogitsLoss
        ignore_mismatched_sizes=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        inference_mode=False,
    )

    model = get_peft_model(base_model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(f"可训练参数：{trainable:,}/{total:,} ({100*trainable/total:.2f}%)")
    return model, tokenizer


# ---------------------------------------------------------------------------
# 主训练流程
# ---------------------------------------------------------------------------

def train(cfg: RerankerConfig = CFG):
    model, tokenizer = build_model_and_tokenizer(cfg)

    train_ds = DocRerankDataset(cfg.train_file, tokenizer, cfg)
    val_ds   = DocRerankDataset(cfg.val_file,   tokenizer, cfg)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=True,
        bf16=cfg.bf16,
        optim="paged_adamw_8bit",
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = SoftLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        pos_weight=cfg.pos_weight,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()

    adapter_path = Path(cfg.output_dir) / "best_adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info(f"LoRA adapter 已保存：{adapter_path}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# 推理：对候选 doc 列表打分，citation 排名由调用方基于分数完成
# ---------------------------------------------------------------------------

class DocReranker:
    """
    加载训练好的 LoRA adapter，对 doc 列表打分。
    citation 的最终排名 = 包含该 citation 的所有 doc 中最高的 doc_score。
    （这一步在你的现有 pipeline 里完成，本类只负责 doc 打分）
    """

    def __init__(self, adapter_path: str, base_model: str = CFG.base_model,
                 device: str = "cuda"):
        from peft import PeftModel
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        base = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=1
        )
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval().to(device)

    @torch.no_grad()
    def score_docs(self, query: str, docs: list[str],
                   batch_size: int = 16, max_length: int = 512) -> list[float]:
        """返回每个 doc 的相关性概率（0~1，越高越好）。"""
        scores = []
        for i in range(0, len(docs), batch_size):
            batch = docs[i: i + batch_size]
            enc = self.tokenizer(
                [query] * len(batch), batch,
                max_length=max_length,
                truncation="longest_first",
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits.squeeze(-1)
            probs  = torch.sigmoid(logits).cpu().tolist()
            scores.extend(probs if isinstance(probs, list) else [probs])
        return scores

    def rerank_and_extract(
        self,
        query: str,
        docs: list[dict],
        top_k_docs: int = 10,
    ) -> list[dict]:
        """
        完整的 rerank + citation 聚合流程示例：
          docs 格式: [{"text": "...", "citations": ["§242BGB", ...]}, ...]

        1. 对所有 doc 打分
        2. 保留 top_k_docs
        3. 从保留的 doc 中提取 citation，以 doc_score 作为 citation_score
        4. 对同一 citation 出现在多个 doc 中的情况取最高分
        返回按分数降序的 citation 列表。
        """
        texts  = [d["text"] for d in docs]
        scores = self.score_docs(query, texts)

        ranked_docs = sorted(
            zip(docs, scores), key=lambda x: x[1], reverse=True
        )[:top_k_docs]

        # 聚合：同一 citation 取最高 doc_score
        citation_scores: dict[str, float] = {}
        for doc, doc_score in ranked_docs:
            for citation in doc.get("citations", []):
                if citation not in citation_scores or doc_score > citation_scores[citation]:
                    citation_scores[citation] = doc_score

        return sorted(
            [{"citation": c, "score": s} for c, s in citation_scores.items()],
            key=lambda x: x["score"],
            reverse=True,
        )


# ---------------------------------------------------------------------------
# 调试工具：检查不同 citation 配置下的软标签值
# ---------------------------------------------------------------------------

def inspect_soft_labels():
    cfg  = RerankerConfig()
    calc = DocSoftLabelCalculator(cfg)

    cases = [
        {"label": 0.0},
        {"label": 1.0},
        {"label": 1.0, "matched_citations": [{"citation_id": "§ 999 BGB", "global_freq": 2}],   "total_expected": 3},
        {"label": 1.0, "matched_citations": [{"citation_id": "§ 280 BGB", "global_freq": 80},
                                              {"citation_id": "§ 249 BGB", "global_freq": 50}],  "total_expected": 3},
        {"label": 1.0, "matched_citations": [{"citation_id": "§ 242 BGB", "global_freq": 847},
                                              {"citation_id": "§ 280 BGB", "global_freq": 312},
                                              {"citation_id": "§ 823 BGB", "global_freq": 150}], "total_expected": 3},
    ]
    descs = ["负样本", "正样本（无细节）", "正：低覆盖低质量",
             "正：中覆盖中质量", "正：全覆盖高质量"]

    print(f"\n{'样本描述':<24} {'软标签':>8}")
    print("-" * 36)
    for desc, case in zip(descs, cases):
        print(f"{desc:<24} {calc.compute(case):>8.4f}")


# ---------------------------------------------------------------------------
# 生成示例数据（快速验证代码可运行）
# ---------------------------------------------------------------------------

def create_sample_data(output_dir: str = "data"):
    import os
    os.makedirs(output_dir, exist_ok=True)

    templates = [
        {
            "query": "Der Schuldner hat die Kosten des Rechtsstreits zu tragen.",
            "doc": "Nach § 91 ZPO hat die unterliegende Partei die Kosten zu tragen. Vgl. § 92 ZPO.",
            "label": 1.0,
            "matched_citations": [
                {"citation_id": "§ 91 ZPO", "global_freq": 520},
                {"citation_id": "§ 92 ZPO", "global_freq": 180},
            ],
            "total_expected": 2,
        },
        {
            "query": "Schadensersatzanspruch wegen Verletzung vertraglicher Pflichten.",
            "doc": "Gemäß § 280 Abs. 1 BGB kann der Gläubiger Ersatz verlangen. Der Schuldner handelte schuldhaft im Sinne des § 276 BGB.",
            "label": 1.0,
            "matched_citations": [
                {"citation_id": "§ 280 Abs. 1 BGB", "global_freq": 312},
                {"citation_id": "§ 276 BGB",         "global_freq": 95},
            ],
            "total_expected": 3,
        },
        {
            "query": "Der Schuldner hat die Kosten des Rechtsstreits zu tragen.",
            "doc": "Das Gericht hat festgestellt, dass der Beklagte seiner Mitwirkungspflicht nicht nachgekommen ist.",
            "label": 0.0,
            "matched_citations": [],
            "total_expected": 2,
        },
        {
            "query": "Schadensersatzanspruch wegen Verletzung vertraglicher Pflichten.",
            "doc": "Die Berufung wurde als unbegründet zurückgewiesen.",
            "label": 0.0,
            "matched_citations": [],
            "total_expected": 3,
        },
    ]

    for split, n in [("train", 200), ("val", 40)]:
        path = f"{output_dir}/{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for i in range(n):
                s = templates[i % len(templates)]
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.info(f"示例数据 → {path} ({n} 条)")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "inspect", "sample"], default="train")
    parser.add_argument("--base_model",  default=CFG.base_model)
    parser.add_argument("--train_file",  default=CFG.train_file)
    parser.add_argument("--val_file",    default=CFG.val_file)
    parser.add_argument("--output_dir",  default=CFG.output_dir)
    parser.add_argument("--lora_r",      type=int,   default=CFG.lora_r)
    parser.add_argument("--epochs",      type=int,   default=CFG.num_train_epochs)
    parser.add_argument("--pos_weight",  type=float, default=CFG.pos_weight)
    args = parser.parse_args()

    if args.mode == "sample":
        create_sample_data()
    elif args.mode == "inspect":
        inspect_soft_labels()
    elif args.mode == "train":
        cfg = RerankerConfig(
            base_model=args.base_model,
            train_file=args.train_file,
            val_file=args.val_file,
            output_dir=args.output_dir,
            lora_r=args.lora_r,
            num_train_epochs=args.epochs,
            pos_weight=args.pos_weight,
        )
        train(cfg)