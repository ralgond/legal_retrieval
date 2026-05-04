python predict.py \
    --base_model_path /root/.cache/modelscope/hub/models/Qwen/Qwen2___5-0___5B-Instruct \
    --adapter_path ../data/ml6/checkpoints/best \
    --input_file ../data/ml6/predict.jsonl \
    --output_file ../data/ml6/output.jsonl \
    --batch_size 8 \
    --max_seq_len 1024