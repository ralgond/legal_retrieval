import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from collections import defaultdict
from sklearn.metrics import roc_auc_score
TRAIN_GROUP_SIZE = 8
MAX_LEN = 512
MODEL_NAME="/root/.cache/modelscope/hub/models/BAAI/bge-reranker-v2-m3"

class RerankerEvalDataset(Dataset):
    """
    eval 不需要 group，每条样本是一个 (query, passage, label) 的 pair
    label 直接传给 compute_metrics 计算 AUC
    """
    def __init__(self, raw_data, tokenizer, max_len=MAX_LEN):
        self.data = raw_data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoded = self.tokenizer(
            [[item["query"], item["passage"]]],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),      # (seq_len,)
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.float),
        }
        
def evaluate_baseline(eval_data, model_name=MODEL_NAME, batch_size=32, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1
    ).to(device)
    model.eval()

    dataset = RerankerEvalDataset(eval_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits.squeeze(-1)

            scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(labels)

    auc = roc_auc_score(all_labels, all_scores)
    print(f"baseline AUC: {auc:.4f}")
    return auc


if __name__ == "__main__":
    eval_data = []
    with open("../ft_data/valid.jsonl", encoding='utf-8') as inf:
        for line in inf:
            d = json.loads(line.strip())
            if d['label'] == "1":
                d['label'] = 1
            elif d['label'] == "0":
                d['label'] = 0
            eval_data.append(d)

    # evaluate_baseline(eval_data)
    evaluate_baseline(eval_data, "../ft_data/merged_reranker")
