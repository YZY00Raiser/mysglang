vllm bench serve \
  --backend openai-chat \
  \
  --model /data/ascend-ci-share-pkking-sglang/modelscope/hub/models/Qwen/Qwen2.5-VL-72B-Instruct \
  --tokenizer /data/ascend-ci-share-pkking-sglang/modelscope/hub/models/Qwen/Qwen2.5-VL-72B-Instruct \
  --served-model-name Qwen2.5-VL-72B-Instruct \
  \
  --host 127.0.0.1 \
  --port 22001 \
  --endpoint /v1/chat/completions \
  \
  --dataset-name random-mm \
  --random-input-len 0 \
  --random-output-len 1024 \
  --random-mm-base-items-per-request 1 \
  --random-mm-limit-mm-per-prompt '{"image": 1}' \
  --random-mm-bucket-config '{(1024,1024,1): 1}' \
  \
  --request-rate 1024 \
  --max-concurrency 148 \
  --num-prompts 384 \
  \
  --percentile-metrics ttft,tpot,itl,e2el \
  --temperature 0 \
  --seed 1000 \
  --ignore-eos \
  --trust-remote-code
