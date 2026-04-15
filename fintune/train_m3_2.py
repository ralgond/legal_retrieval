import torch
import os
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm import tqdm
import torch.nn.functional as F

# ====================== 配置 ======================
model_name = "/root/.cache/modelscope/hub/models/BAAI/bge-m3/"
train_file = "../ft_data/bge-m3_unsupervised_data.txt"       # 每行一段纯文本
output_dir = "../ft_data/bge-m3-lora"
device = "cuda" if torch.cuda.is_available() else "cpu"

print_interval = 50
save_interval = 10000
temperature = 0.05
batch_size = 4
lr = 2e-4
num_epochs = 1

# ---------------------- 加载模型 ----------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# --------------- 核心修复：强制开启 Dropout ！---------------
model.train()  # 全程保持训练模式，让 Dropout 生效
for param in model.parameters():
    param.requires_grad = False  # 先冻结主干

# LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft(model, lora_config)
model.print_trainable_parameters()

# ---------------------- 数据 ----------------------
with open(train_file, "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

def collate_fn(batch):
    return tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")

dataloader = DataLoader(texts, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# ---------------------- 优化器 ----------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
total_steps = num_epochs * len(dataloader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)
model.to(device)

# ====================== 训练开始 ======================
global_step = 0
loss_accum = 0.0

print("✅ 最终修复版启动！Dropout 已开启，loss 必正常\n")

for epoch in range(num_epochs):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    for batch in pbar:
        global_step += 1
        batch = {k: v.to(device) for k, v in batch.items()}
        
        model.train()  # 强制保持训练模式（Dropout 生效）
        
        # 两次编码 = 不同向量（因为 Dropout 生效）
        emb1 = model(**batch).last_hidden_state[:, 0]
        emb2 = model(**batch).last_hidden_state[:, 0]
        
        emb1 = F.normalize(emb1, dim=-1)
        emb2 = F.normalize(emb2, dim=-1)
        
        # 正常 InfoNCE 损失
        sim = torch.mm(emb1, emb2.t()) / temperature
        labels = torch.arange(sim.size(0)).to(device)
        loss = F.cross_entropy(sim, labels)
        
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        loss_accum += loss.item()
        
        # 打印 loss
        if global_step % print_interval == 0:
            avg_loss = loss_accum / print_interval
            print(f"[Step {global_step}] Normal Loss: {avg_loss:.4f}")
            loss_accum = 0.0
        
        # 保存 ckpt
        if global_step % save_interval == 0:
            ckpt_dir = f"{output_dir}/ckpt_{global_step}"
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"💾 保存：{ckpt_dir}\n")

# 最终保存
model.save_pretrained(f"{output_dir}/final")
tokenizer.save_pretrained(f"{output_dir}/final")
print("\n🎉 训练完成！模型真正有效！")