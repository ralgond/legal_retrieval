from modelscope import snapshot_download

# 指定模型ID，可以从 ModelScope 官网获取
# model_id = 'Qwen/Qwen3-0.6B'
model_id = 'Qwen/Qwen2.5-0.5B-Instruct'
# 开始下载
# cache_dir 参数用于指定保存路径
model_dir = snapshot_download(model_id)

print(f"模型已成功下载到: {model_dir}")