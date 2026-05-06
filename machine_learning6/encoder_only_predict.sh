python encoder_only_predict.py \
    --model_dir  ../data/ml6/checkpoints_encoder/best \
    --base_model /root/.cache/modelscope/hub/models/jhu-clsp/mmBERT-base \
    --input_file  ../data/ml6/predict.jsonl \
    --output_file ../data/ml6/output.jsonl \
    --max_seq_len 1024 \
    --batch_size 16