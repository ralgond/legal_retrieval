# 第一步：合并并保存
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("../ft_data/bge-reranker-lora")
base = AutoModelForSequenceClassification.from_pretrained(
    "/root/.cache/modelscope/hub/models/BAAI/bge-reranker-v2-m3", num_labels=1
)
# model = PeftModel.from_pretrained(base, "../ft_data/lora_reranker_output")
model = PeftModel.from_pretrained(base, "../ft_data/bge-reranker-lora")
merged = model.merge_and_unload()

merged.save_pretrained("../ft_data/merged_reranker")
tokenizer.save_pretrained("../ft_data/merged_reranker")