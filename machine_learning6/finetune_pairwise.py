"""
mT5-base 微调：瑞士法律 citation 二分类
任务：给定 (query, cc, citation)，判断 citation 是否为 gold citation

数据说明
--------
- cc 文本里通过规则/NER抽取出若干 citation
- 人工标注其中哪些是 gold citation
- 负例来源：
    1. 同一条 cc 里抽取到的非 gold citation（in-cc negatives，最自然）
    2. 其他高分 cc 里的非 gold citation（hard negatives，跨 cc）
    3. 全局非 gold citation 随机采样（random negatives）

原始数据格式（JSONL，每行一个 query）
--------------------------------------
{
  "query_id": "q_001",
  "query": "Welche Voraussetzungen gelten für die Kündigung?",
  "court_considerations": [
    {
      "cc_id": "cc_abc",
      "text": "Das Gericht erwägt... gemäss BGE 147 III 215, BGE 143 II 506, BGE 100 I 1...",
      "rerank_score": 0.92,
      "extracted_citations": [        # 从 cc 文本中抽取的所有 citation
          "BGE_147_III_215",
          "BGE_143_II_506",
          "BGE_100_I_1"
      ],
      "gold_citations": [             # 人工标注的 gold
          "BGE_147_III_215",
          "BGE_143_II_506"
      ]
    },
    ...
  ]
}

训练样本格式（JSONL，每行一条）
--------------------------------
{
  "query_id": "q_001",
  "query": "...",
  "cc_id": "cc_abc",
  "cc_text": "...",     # citation 在 cc 全文中的前后各两句上下文（非全文）
  "rerank_score": 0.92,
  "citation": "BGE_147_III_215",
  "label": 1,           # 1 = gold, 0 = non-gold
  "neg_type": null      # null(正例) / "in_context"(同context内) / "random"(全局随机)
}
"""

import json
import random
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. 负采样
# ─────────────────────────────────────────────

@dataclass
class NegSamplingConfig:
    min_cc_score: float = 0.5
    n_in_cc: int = 2      # 同一 context 窗口内的非 gold citation（最难，共享相同上下文）
    n_hard: int = 2       # 同 context 内负例不够时的补充配额（合并计入 context 内负例）
    n_random: int = 1     # context 内负例仍不够时，从全局非 gold 池随机补充
    seed: int = 42


class NegativeSampler:
    """
    正例：(query, cc, gold_citation)，label=1
    负例来源（均从同一 query 的 cc 上下文出发）：

    In-CC negative（最重要）：
        同一条 cc 里被抽取到、但未被标注为 gold 的 citation。
        这些 citation 和 gold 出现在完全相同的文本上下文里，
        模型必须学会从语义上区分 gold 与非 gold。

    Hard negative：
        同一 query 下其他 cc（rerank_score 高的优先）里的非 gold citation。
        这些 citation 与 query 同领域，但来自不同的 cc，更难区分。

    Random negative：
        跨 query 的全局非 gold citation 随机采样，保证分布多样性。
    """

    def __init__(self, cfg: NegSamplingConfig):
        self.cfg = cfg
        random.seed(cfg.seed)

        # 全局非 gold citation 池，build_dataset 调用前先 scan 一遍填充
        self._global_non_gold: list[str] = []

    def _scan_global_pool(self, raw_path: str):
        """扫描全量数据，收集所有非 gold citation 作为 random negative 来源"""
        non_gold_set: set[str] = set()
        with open(raw_path) as f:
            for line in f:
                ex = json.loads(line.strip())
                for cc in ex["court_considerations"]:
                    gold = set(cc.get("gold_citations", []))
                    for cit in cc.get("extracted_citations", []):
                        if cit not in gold:
                            non_gold_set.add(cit)
        self._global_non_gold = list(non_gold_set)
        logger.info(f"Global non-gold citation pool size: {len(self._global_non_gold)}")

    def _sample(self, pool: list[str], exclude: set[str], n: int) -> list[str]:
        cands = [c for c in pool if c not in exclude]
        return random.sample(cands, min(n, len(cands))) if cands else []

    @staticmethod
    def extract_context(cc_text: str, citation: str, context_sentences: int = 2) -> str:
        """
        从 cc_text 中找到 citation 出现的位置，返回其前后各 N 句话。

        citation 在文本中可能以多种形式出现，例如：
          "BGE_147_III_215" → 匹配 "BGE 147 III 215" 或 "BGE_147_III_215"
        先尝试原始字符串匹配，再尝试将下划线替换为空格后匹配。

        若 citation 在文本中找不到，退而返回整段 cc_text（保证不丢数据）。
        """
        import re

        # 将 citation 规范化为两种查找形式
        variants = [citation, citation.replace("_", " ")]

        # 按句子切分（德语文本以 "." "!" "?" 结尾，保留分隔符）
        sentences = re.split(r'(?<=[.!?])\s+', cc_text.strip())
        if not sentences:
            return cc_text

        # 找到包含 citation 的句子索引
        hit_idx = None
        for variant in variants:
            for i, sent in enumerate(sentences):
                if variant in sent:
                    hit_idx = i
                    break
            if hit_idx is not None:
                break

        if hit_idx is None:
            # citation 未在文本中找到，返回全文作为 fallback
            return cc_text

        start = max(0, hit_idx - context_sentences)
        end   = min(len(sentences), hit_idx + context_sentences + 1)
        return " ".join(sentences[start:end])

    def build_samples(self, example: dict) -> list[dict]:
        """
        以每个 gold citation 的 context 为锚点构建训练样本：

        正例：(query, context_of_gold, gold_citation)
        负例：在同一个 context 窗口里出现的其他 citation（非 gold）

        正负例的 cc_text 完全相同，模型只能靠 citation 本身区分。

        若 context 里没有其他 citation 可用作负例，
        则降级到全局非 gold 池随机采样（此时 cc_text 仍用正例的 context）。
        """
        query    = example["query"]
        query_id = example.get("query_id", "")
        ccs = [
            cc for cc in example["court_considerations"]
            if cc["rerank_score"] >= self.cfg.min_cc_score
        ]
        if not ccs:
            return []

        samples = []

        for cc in ccs:
            cc_id     = cc.get("cc_id", "")
            cc_text   = cc["text"]
            score     = cc["rerank_score"]
            gold_set  = set(cc.get("gold_citations", []))
            extracted = set(cc.get("extracted_citations", []))

            if not gold_set:
                continue

            non_gold_in_cc = extracted - gold_set   # 同一 cc 里的非 gold citation

            for gold_cit in gold_set:
                # ── 1. 提取正例的 context ──
                context = self.extract_context(cc_text, gold_cit, context_sentences=2)

                def make(citation, label, neg_type):
                    return {
                        "query_id":     query_id,
                        "query":        query,
                        "cc_id":        cc_id,
                        "cc_text":      context,   # 所有样本共享同一 context
                        "rerank_score": score,
                        "citation":     citation,
                        "label":        label,
                        "neg_type":     neg_type,
                    }

                samples.append(make(gold_cit, 1, None))

                exclude = {gold_cit}

                # ── 2. 在同一 context 里找负例 ──
                # 找出所有在这个 context 窗口中出现的非 gold citation
                context_neg_pool = [
                    cit for cit in non_gold_in_cc
                    if cit not in exclude
                    and any(
                        v in context
                        for v in [cit, cit.replace("_", " ")]
                    )
                ]

                # 优先取 context 内的负例（和正例出现在完全相同的上下文里，最难）
                context_negs = self._sample(
                    context_neg_pool, exclude,
                    self.cfg.n_in_cc + self.cfg.n_hard
                )
                for cit in context_negs:
                    samples.append(make(cit, 0, "in_context"))
                exclude.update(context_negs)

                # ── 3. context 内负例不够，补充全局随机负例 ──
                # cc_text 仍然用正例的 context，只是 citation 来自全局池
                n_needed = (self.cfg.n_in_cc + self.cfg.n_hard + self.cfg.n_random
                            - len(context_negs))
                if n_needed > 0:
                    rand_negs = self._sample(
                        self._global_non_gold, exclude, n_needed
                    )
                    for cit in rand_negs:
                        samples.append(make(cit, 0, "random"))

        return samples

    def build_dataset(self, raw_path: str, output_path: str) -> int:
        self._scan_global_pool(raw_path)

        all_samples: list[dict] = []
        skipped = 0
        with open(raw_path) as f:
            for line in f:
                ex = json.loads(line.strip())
                s = self.build_samples(ex)
                if s:
                    all_samples.extend(s)
                else:
                    skipped += 1

        # 统计
        pos = sum(1 for s in all_samples if s["label"] == 1)
        neg = len(all_samples) - pos
        type_dist: dict[str, int] = {}
        for s in all_samples:
            k = s["neg_type"] or "positive"
            type_dist[k] = type_dist.get(k, 0) + 1

        logger.info(
            f"Built {len(all_samples)} samples "
            f"(pos={pos}, neg={neg}), skipped {skipped} queries.\n"
            f"Distribution: {type_dist}"
        )

        with open(output_path, "w") as f:
            for s in all_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        return len(all_samples)


# ─────────────────────────────────────────────
# 2. Dataset
# ─────────────────────────────────────────────

class SwissLegalDataset(Dataset):
    """
    Pairwise 格式：对同一个 (query_id, cc_id) 下的正负例配对。
    每条样本 = (query, cc_text, pos_citation, neg_citation)，
    正负例的 cc_text 完全相同（同一个 context 窗口）。
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer  = tokenizer
        self.max_length = max_length

        with open(data_path) as f:
            raw = [json.loads(l) for l in f]

        self.pairs = self._make_pairs(raw)
        logger.info(f"Loaded {len(raw)} samples → {len(self.pairs)} pairs from {data_path}")

    def _make_pairs(self, samples: list[dict]) -> list[dict]:
        """
        同一 (query_id, cc_id, cc_text) 下正负例两两配对。
        cc_text 也作为分组 key，因为同一 gold citation 对应唯一 context。
        负例多于正例时循环复用正例，反之循环复用负例。
        """
        from collections import defaultdict
        groups = defaultdict(lambda: {"pos": [], "neg": []})
        for s in samples:
            key = (s["query_id"], s["cc_id"], s["cc_text"])
            if s["label"] == 1:
                groups[key]["pos"].append(s)
            else:
                groups[key]["neg"].append(s)

        pairs = []
        for key, g in groups.items():
            pos_list, neg_list = g["pos"], g["neg"]
            if not pos_list or not neg_list:
                continue
            n = max(len(pos_list), len(neg_list))
            for i in range(n):
                p = pos_list[i % len(pos_list)]
                n_ = neg_list[i % len(neg_list)]
                pairs.append({
                    "query":        p["query"],
                    "cc_text":      p["cc_text"],   # 正负例共享同一 context
                    "pos_citation": p["citation"],
                    "neg_citation": n_["citation"],
                })
        return pairs

    def _encode(self, query: str, cc_text: str, citation: str) -> dict:
        text = f"Query: {query} Context: {cc_text} Citation: {citation}"
        return self.tokenizer(
            text, max_length=self.max_length,
            truncation=True, padding="max_length", return_tensors="pt",
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        pos_enc = self._encode(p["query"], p["cc_text"], p["pos_citation"])
        neg_enc = self._encode(p["query"], p["cc_text"], p["neg_citation"])
        return {
            "pos_input_ids":      pos_enc["input_ids"].squeeze(0),
            "pos_attention_mask": pos_enc["attention_mask"].squeeze(0),
            "neg_input_ids":      neg_enc["input_ids"].squeeze(0),
            "neg_attention_mask": neg_enc["attention_mask"].squeeze(0),
        }


def collate_fn(batch):
    return {
        "pos_input_ids":      torch.stack([b["pos_input_ids"] for b in batch]),
        "pos_attention_mask": torch.stack([b["pos_attention_mask"] for b in batch]),
        "neg_input_ids":      torch.stack([b["neg_input_ids"] for b in batch]),
        "neg_attention_mask": torch.stack([b["neg_attention_mask"] for b in batch]),
    }


# ─────────────────────────────────────────────
# 3. Cross-Encoder（pairwise，支持任意 AutoModel encoder）
# ─────────────────────────────────────────────

class CrossEncoderClassifier(nn.Module):
    """
    Pairwise ranking：给定 (query, cc_text, pos_citation) 和
    (query, cc_text, neg_citation)，用 margin loss 让 pos_score > neg_score。

    结构：AutoModel（RoBERTa / mT5EncoderModel 等）+ 线性分类头
    取 [CLS] token（index 0）的表示接线性头。
    Loss：-logsigmoid(pos_score - neg_score)

    兼容 legal-swiss-roberta-large 和 google/mt5-base。
    """

    def __init__(self, model_name: str, freeze_layers: int = 6):
        super().__init__()
        from transformers import AutoModel, AutoConfig

        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,   # 显存减半，动态范围与 fp32 相同
        )
        hidden_size  = config.hidden_size

        # 冻结底部 freeze_layers 层
        # RoBERTa: encoder.layer[i]
        # mT5 encoder: encoder.encoder.block[i]
        layers = None
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            layers = self.encoder.encoder.layer          # RoBERTa
        elif hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "block"):
            layers = self.encoder.encoder.block          # mT5
        elif hasattr(self.encoder, "roberta"):
            layers = self.encoder.roberta.encoder.layer  # RobertaForMaskedLM wrapper

        if layers is not None:
            for i, layer in enumerate(layers):
                if i < freeze_layers:
                    for p in layer.parameters():
                        p.requires_grad = False
            logger.info(f"Frozen {freeze_layers}/{len(layers)} layers")
        else:
            logger.warning("Could not detect layer structure, no layers frozen")

        self.classifier = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        logger.info(f"CrossEncoderClassifier: hidden_size={hidden_size}, model={model_name}")

    def get_scores(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # autocast 上下文外做 float32 运算，防止 bf16 精度截断（步长 0.125）
        with torch.autocast(device_type="cuda", enabled=False):
            cls = out.last_hidden_state[:, 0, :].to(torch.float32)  # (B, hidden)
            return self.classifier(cls).squeeze(-1)                   # (B,) float32

    def forward(self, batch: dict, step: int = -1) -> tuple[torch.Tensor, dict]:
        device = next(self.parameters()).device

        pos_s = self.get_scores(
            batch["pos_input_ids"].to(device),
            batch["pos_attention_mask"].to(device),
        )
        neg_s = self.get_scores(
            batch["neg_input_ids"].to(device),
            batch["neg_attention_mask"].to(device),
        )

        if step == 0:
            logger.info(f"[diag step=0] pos_s={pos_s.tolist()} neg_s={neg_s.tolist()}")

        loss = -nn.functional.logsigmoid(pos_s - neg_s).mean()

        with torch.no_grad():
            acc     = (pos_s > neg_s).float().mean().item()
            avg_pos = pos_s.mean().item()
            avg_neg = neg_s.mean().item()

        return loss, {
            "avg_pos_score": avg_pos,
            "avg_neg_score": avg_neg,
            "acc":           acc,
        }

# 保持向后兼容
MT5Classifier = CrossEncoderClassifier




# ─────────────────────────────────────────────
# 4. 训练循环
# ─────────────────────────────────────────────

class Trainer:
    def __init__(
        self,
        model: MT5Classifier,
        train_ds: SwissLegalDataset,
        eval_ds: Optional[SwissLegalDataset],
        output_dir: str,
        num_epochs: int = 3,
        batch_size: int = 8,
        grad_accum_steps: int = 4,   # 有效 batch_size = batch_size * grad_accum_steps
        lr: float = 2e-5,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        bf16: bool = True,    # 用 bf16 替代 fp16，mT5 在 fp16 下 RMSNorm 会溢出
        eval_steps: int = 500,
        save_steps: int = 1000,
    ):
        self.model             = model
        self.train_ds          = train_ds
        self.eval_ds           = eval_ds
        self.output_dir        = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs        = num_epochs
        self.batch_size        = batch_size
        self.grad_accum_steps  = grad_accum_steps
        self.lr                = lr
        self.warmup_ratio      = warmup_ratio
        self.max_grad_norm     = max_grad_norm
        self.bf16              = bf16
        self.eval_steps        = eval_steps
        self.save_steps        = save_steps
        self.device            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # bf16 不需要 GradScaler（动态范围与 fp32 相同，无需 loss scaling）

    @torch.no_grad()
    def evaluate(self) -> dict:
        if self.eval_ds is None:
            return {}
        self.model.eval()
        loader = DataLoader(
            self.eval_ds, batch_size=self.batch_size * 2,
            collate_fn=collate_fn, num_workers=4,
        )
        total_loss, total_acc, n = 0.0, 0.0, 0
        for batch in loader:
            if self.bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss, m = self.model(batch)
            else:
                loss, m = self.model(batch)
            total_loss += loss.item()
            total_acc  += m["acc"]
            n += 1
        self.model.train()
        return {"eval_loss": total_loss / n, "eval_acc": total_acc / n}

    def _save(self, name: str):
        path = self.output_dir / name
        path.mkdir(parents=True, exist_ok=True)
        self.model.encoder.save_pretrained(path)
        torch.save(self.model.classifier.state_dict(), path / "classifier.pt")
        logger.info(f"Saved → {path}")

    def train(self):
        loader = DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=4, pin_memory=True,
        )
        total_update_steps = (len(loader) * self.num_epochs) // self.grad_accum_steps

        # 8-bit AdamW：优化器状态从 fp32（每参数 8 bytes）压缩到 int8（2 bytes），
        # 显存再减 75%。需要 pip install bitsandbytes。
        # 若未安装则退回标准 AdamW。
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                self.model.parameters(), lr=self.lr, weight_decay=0.01
            )
            logger.info("Using 8-bit AdamW (bitsandbytes)")
        except ImportError:
            optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
            logger.info("bitsandbytes not found, using standard AdamW")
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_update_steps * self.warmup_ratio),
            num_training_steps=total_update_steps,
        )
        global_step, best_acc = 0, 0.0
        running_acc = 0.0
        optimizer.zero_grad()

        import time
        total_micro_steps = len(loader) * self.num_epochs
        train_start = time.time()

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for step, batch in enumerate(loader):
                micro_step = epoch * len(loader) + step + 1
                is_accum_step = (step + 1) % self.grad_accum_steps != 0

                if self.bf16:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss, m = self.model(batch, step=step)
                else:
                    loss, m = self.model(batch, step=step)

                (loss / self.grad_accum_steps).backward()

                if not is_accum_step:
                    # 检查梯度是否真的在流动
                    if global_step == 0:
                        for name, p in self.model.named_parameters():
                            if p.requires_grad and p.grad is not None:
                                logger.info(f"[grad check] {name}: grad_norm={p.grad.norm().item():.4e}")
                            elif p.requires_grad:
                                logger.info(f"[grad check] {name}: grad=None ← 梯度未传到")

                    total_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    if global_step % 100 == 0:
                        logger.info(f"lr={scheduler.get_last_lr()[0]:.2e} grad_norm={total_norm:.3f}")

                running_acc = 0.98 * running_acc + 0.02 * m["acc"]
                epoch_loss += loss.item()

                if step % 100 == 0:
                    elapsed = time.time() - train_start
                    speed = micro_step / elapsed
                    eta_sec = (total_micro_steps - micro_step) / speed if speed > 0 else 0
                    eta_h, rem = divmod(int(eta_sec), 3600)
                    eta_m, eta_s = divmod(rem, 60)
                    eta_str = f"{eta_h}h {eta_m:02d}m {eta_s:02d}s"
                    elapsed_h, rem = divmod(int(elapsed), 3600)
                    elapsed_m, elapsed_s = divmod(rem, 60)
                    elapsed_str = f"{elapsed_h}h {elapsed_m:02d}m {elapsed_s:02d}s"

                    logger.info(
                        f"Epoch {epoch+1}/{self.num_epochs} "
                        f"step {step}/{len(loader)} "
                        f"[elapsed {elapsed_str} | ETA {eta_str}] "
                        f"loss={loss.item():.4f} "
                        f"pos={m['avg_pos_score']:.3f} neg={m['avg_neg_score']:.3f} "
                        f"acc={m['acc']:.3f} running_acc={running_acc:.3f}"
                    )
                if global_step % self.eval_steps == 0 and not is_accum_step:
                    ev = self.evaluate()
                    logger.info(f"[update_step {global_step}] {ev}")
                    if ev.get("eval_acc", 0) > best_acc:
                        best_acc = ev["eval_acc"]
                        self._save("best_model")
                if global_step % self.save_steps == 0 and not is_accum_step:
                    self._save(f"checkpoint-{global_step}")

            logger.info(
                f"Epoch {epoch+1} done. avg_loss={epoch_loss/len(loader):.4f}"
            )

        self._save("final_model")


# ─────────────────────────────────────────────
# 5. 推理
# ─────────────────────────────────────────────

class CitationClassifier:
    """推理：给定 (query, cc_text, candidate_citations)，返回每个 citation 的相关性分数。"""

    def __init__(self, model_path: str):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model     = CrossEncoderClassifier(model_path)
        clf_path = Path(model_path) / "classifier.pt"
        if clf_path.exists():
            self.model.classifier.load_state_dict(
                torch.load(clf_path, map_location="cpu")
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        query: str,
        cc_text: str,
        citations: list[str],
        batch_size: int = 32,
        max_length: int = 512,
    ) -> list[tuple[str, float]]:
        """返回 [(citation, score), ...] 按分数降序。"""
        scores = []
        for i in range(0, len(citations), batch_size):
            batch_cits = citations[i: i + batch_size]
            texts = [
                f"Query: {query} Context: {cc_text} Citation: {cit}"
                for cit in batch_cits
            ]
            enc = self.tokenizer(
                texts, max_length=max_length, truncation=True,
                padding=True, return_tensors="pt",
            ).to(self.device)
            batch_scores = self.model.get_scores(enc["input_ids"], enc["attention_mask"])
            scores.extend(batch_scores.cpu().tolist())
        return sorted(zip(citations, scores), key=lambda x: x[1], reverse=True)


# ─────────────────────────────────────────────
# 6. 主流程
# ─────────────────────────────────────────────

def main():
    MODEL_NAME = "/root/.cache/modelscope/hub/models/ralgond/legal-swiss-roberta-large"
    from transformers import AutoTokenizer
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── 步骤 1：负采样，构建训练样本 ──
    cfg     = NegSamplingConfig(
        min_cc_score=0.5,
        n_in_cc=2,
        n_hard=2,
        n_random=1,
        seed=42,
    )
    # sampler = NegativeSampler(cfg)
    # sampler.build_dataset("data/train_raw.jsonl", "data/train_samples.jsonl")
    # sampler.build_dataset("data/eval_raw.jsonl",  "data/eval_samples.jsonl")

    # ── 步骤 2：构建 Dataset ──
    # legal-swiss-roberta-large 最大序列长度 512
    train_ds = SwissLegalDataset("../data/ml6/train_samples.jsonl", tokenizer, max_length=512)
    eval_ds  = SwissLegalDataset("../data/ml6/eval_samples.jsonl",  tokenizer, max_length=512)
    logger.info(f"Train pairs: {len(train_ds)}, Eval pairs: {len(eval_ds)}")

    # ── 步骤 3：训练 ──
    model   = CrossEncoderClassifier(
        MODEL_NAME,
        freeze_layers=12,   # roberta-large 共 24 层，冻结底部 12 层
    )
    trainer = Trainer(
        model=model,
        train_ds=train_ds,
        eval_ds=eval_ds,
        output_dir="../data/ml6/checkpoints/legal_swiss_roberta",
        num_epochs=1,
        batch_size=8,
        grad_accum_steps=1,
        lr=1e-4,
        warmup_ratio=0.01,
        bf16=True,
        eval_steps=50000000,
        save_steps=100000,
    )
    trainer.train()

    # ── 步骤 4：推理示例 ──
    # clf     = CitationClassifier("checkpoints/legal_swiss_roberta/best_model")
    # query   = "Welche Fristen gelten bei der Anfechtung eines Testaments?"
    # cc_text = "Das Gericht erwägt, dass gemäss Art. 521 ZGB die Anfechtungsklage..."
    # candidates = ["BGE_148_III_1", "BGE_141_III_210", "BGE_138_I_49", "BGE_147_III_215"]
    # results = clf.predict(query, cc_text, candidates)
    # for cit, score in results:
    #     print(f"{cit:30s}  score={score:.4f}")


if __name__ == "__main__":
    main()