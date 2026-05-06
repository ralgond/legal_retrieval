python encoder_only_train.py \
    --model_name_or_path /root/.cache/modelscope/hub/models/AI-ModelScope/ModernBERT-base \
    --train_file ../data/ml6/train.jsonl \
    --dev_raw_file ../data/ml6/dev_raw.jsonl \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 4 \
    --num_train_epochs 3