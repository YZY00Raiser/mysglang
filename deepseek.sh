export MODEL_PATH=/data/models/Qwen2.5-72B-Instruct  # 设置模型路径
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

#Deepep communication settings
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export HCCL_BUFFSIZE=1600

#spec overlap
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

#npu acceleration operator
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_USE_FIA_NZ=1

python3 -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --tp 16 \
    --trust-remote-code \
    --attention-backend ascend \
    --device npu \
    --quantization modelslim \
    --watchdog-timeout 9000 \
    --cuda-graph-bs 8 16 24 28 32 \
    --mem-fraction-static 0.68 \
    --max-running-requests 128 \
    --context-length 8188 \
    --disable-radix-cache \
    --chunked-prefill-size -1 \
    --max-prefill-tokens 16384 \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --enable-dp-attention \
    --dp-size 4 \
    --enable-dp-lm-head \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --dtype bfloat16
