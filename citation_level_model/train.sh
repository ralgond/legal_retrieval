#!/usr/bin/env bash
# run_train.sh — 启动 LoRA 微调
 
python finetune_bge_reranker_lora.py \
  --model_name_or_path /root/.cache/modelscope/hub/models/BAAI/bge-reranker-v2-m3 \
  --train_file ../ft_data/train.jsonl \
  --eval_file ../ft_data/valid.jsonl \
  --max_length 512 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules "query,key,value" \
  --output_dir ../ft_data/bge-reranker-lora \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type cosine \
  --bf16 True \
  --eval_strategy epoch \
  --save_strategy epoch \
  --load_best_model_at_end True \
  --metric_for_best_model auc \
  --logging_steps 50 \
  --report_to none