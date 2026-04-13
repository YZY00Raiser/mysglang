export MODEL_PATH=/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/DeepSeek-R1-0528-w4a8-per-channel  # 设置模型路径
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export PREFILL_HOST_IP=172.22.3.154
export PORT=8000
export DECODE_HOST_IP=172.22.3.71
#memfabric config store
export ASCEND_MF_STORE_URL="tcp://<PREFILL_HOST_IP>:<PORT>"

#Deepep communication settings
export HCCL_BUFFSIZE=720
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=88

#spec overlap
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

#npu acceleration operator
unset TASK_QUEUE_ENABLE
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_USE_FIA_NZ=1

# suggest max-running-requests <= max-cuda-graph-bs * dp_size, Because when this value is exceeded, performance will significantly degrade.
python -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --disaggregation-mode decode \
    --host $DECODE_HOST_IP \
    --port 8001 \
    --trust-remote-code \
    --nnodes 1 \
    --node-rank 0 \
    --tp-size 16 \
    --dp-size 16 \
    --mem-fraction-static 0.8 \
    --max-running-requests 352 \
    --attention-backend ascend \
    --device npu \
    --quantization modelslim \
    --moe-a2a-backend deepep \
    --enable-dp-attention \
    --deepep-mode low_latency \
    --enable-dp-lm-head \
    --cuda-graph-bs 8 10 12 14 16 18 20 22 \
    --disaggregation-transfer-backend ascend \
    --watchdog-timeout 9000 \
    --context-length 8192 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --disable-shared-experts-fusion \
    --dtype bfloat16 \
    --tokenizer-worker-num 4
