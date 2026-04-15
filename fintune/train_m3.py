from FlagEmbedding import BGEM3Trainer, BGEM3Model
from peft import LoraConfig

# ---------------------- 配置 ----------------------
model_name_or_path = "/root/.cache/modelscope/hub/models/BAAI/bge-m3/"  # 自监督基座
train_data = "../ft_data/bge-m3_unsupervised_data.txt"                        # 纯文本文件
output_dir = "../ft_data/bge-m3-lora"                 # 输出路径

# LoRA 配置（自监督微调 + LoRA）
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["key", "query", "value"],
    lora_dropout=0.05,
    bias="none"
)

# ---------------------- 训练器 ----------------------
trainer = BGEM3Trainer(
    model_name_or_path=model_name_or_path,
    lora_config=lora_config,          # 启用 LoRA
    use_self_distill=True,            # 自监督蒸馏（核心）
    unified_finetuning=True,          # 同时训练 dense/sparse/multi-vector
    fp16=True,
    device="cuda"
)

# ---------------------- 开始训练 ----------------------
trainer.train(
    train_data=train_data,
    output_dir=output_dir,
    learning_rate=2e-4,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    train_group_size=2
)