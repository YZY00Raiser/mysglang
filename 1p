export MODEL_PATH=/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/DeepSeek-R1-0528-w4a8-per-channel  # 设置模型路径
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

#memfabric config store
export ASCEND_MF_STORE_URL="tcp://<PREFILL_HOST_IP>:<PORT>"

#Deepep communication settings
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export HCCL_BUFFSIZE=1536

#npu acceleration operator
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_USE_FIA_NZ=1
export TASK_QUEUE_ENABLE=2

python -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --host $PREFILL_HOST_IP \
    --port 8000 \
    --disaggregation-mode prefill \
    --disaggregation-bootstrap-port 8996 \
    --disaggregation-transfer-backend ascend \
    --trust-remote-code \
    --nnodes 1 \
    --node-rank 0 \
    --tp-size 16 \
    --mem-fraction-static 0.6 \
    --attention-backend ascend \
    --device npu \
    --quantization modelslim \
    --load-balance-method round_robin \
    --max-running-requests 8 \
    --context-length 8192 \
    --disable-radix-cache \
    --chunked-prefill-size -1 \
    --max-prefill-tokens 28680 \
    --moe-a2a-backend deepep \
    --deepep-mode normal \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --dp-size 2 \
    --enable-dp-attention \
    --disable-shared-experts-fusion \
    --dtype bfloat16
