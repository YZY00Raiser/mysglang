# 1. 启动服务
python -m sglang.launch_server \
    --model-path /path/to/Qwen3-VL-235B-A22B-Instruct \
    --trust-remote-code \
    --cuda-graph-max-bs 32 \
    --enable-multimodal \
    --mem-fraction-static 0.8 \
    --attention-backend ascend \
    --disable-cuda-graph \
    --tp-size 16 &

# 2. 运行 MMMU 测试（指定本地数据集）
python benchmark/mmmu/bench_sglang.py \
    --port 30000 \
    --concurrency 64 \
    --dataset-path /home/y30082119/dataset/MMMU \
    --split validation \
    --max-new-tokens 30 \
    --response-answer-regex "(.*)"


# 自动从 HuggingFace 加载 MMMU/MMMU 数据集
python -m sglang.test.run_eval \
    --eval-name mmmu \
    --num-examples 100 \
    --num-threads 64 \
    --max-tokens 30


python -m sglang.test.run_eval \
    --eval-name mmmu \
    --num-examples 100 \
    --num-threads 64 \
    --max-tokens 30 \
    --port 30000 \
    --host 127.0.0.1 \
    --dataset-path /home/y30082119/dataset/MMMU \
    --response-answer-regex "(.*)"

数据集缓存位置 ：

- 默认缓存路径： ~/.cache/huggingface/datasets/MMMU/
- 或通过环境变量指定： export HF_DATASETS_CACHE="/path/to/cache"
