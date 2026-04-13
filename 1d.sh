# ===================== 基础环境配置 =====================
# 模型路径（与Prefill节点保持完全一致）
export MODEL_PATH=/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/DeepSeek-R1-0528-w4a8-per-channel
# NPU内存分配优化
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# 单设备并发流
export STREAMS_PER_DEVICE=32

# 网络配置（Prefill + Decode 节点IP）
export PREFILL_HOST_IP=172.22.3.154
export PORT=8000
export DECODE_HOST_IP=172.22.3.71

# ===================== MemFabric 通信配置（已修复） =====================
export ASCEND_MF_STORE_URL="tcp://${PREFILL_HOST_IP}:${PORT}"

# ===================== DeepEp 通信优化 =====================
export HCCL_BUFFSIZE=720
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=88

# ===================== 推测解码 + 流重叠优化 =====================
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

# ===================== NPU 算子加速 =====================
unset TASK_QUEUE_ENABLE
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_USE_FIA_NZ=1

# ===================== 启动 SGLang Decode 解耦服务 =====================
python -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    # 核心：解耦推理 - 仅运行Decode阶段
    --disaggregation-mode decode \
    --host $DECODE_HOST_IP \
    --port 8001 \
    --trust-remote-code \
    --nnodes 1 \
    --node-rank 0 \
    # 张量并行 + 数据并行（与Prefill匹配）
    --tp-size 16 \
    --dp-size 16 \
    # NPU内存占用80%
    --mem-fraction-static 0.8 \
    # 最大并发请求（符合CUDA Graph约束）
    --max-running-requests 352 \
    # 昇腾NPU配置
    --attention-backend ascend \
    --device npu \
    --quantization modelslim \
    # MOE专家模型通信后端
    --moe-a2a-backend deepep \
    --enable-dp-attention \
    --deepep-mode low_latency \
    --enable-dp-lm-head \
    # CUDA Graph 动态Batch配置
    --cuda-graph-bs 8 10 12 14 16 18 20 22 \
    # 昇腾解耦传输后端
    --disaggregation-transfer-backend ascend \
    # 看门狗超时时间
    --watchdog-timeout 9000 \
    # 上下文长度（与Prefill一致）
    --context-length 8192 \
    # 推测解码配置
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    # MOE优化开关
    --disable-shared-experts-fusion \
    --dtype bfloat16 \
    --tokenizer-worker-num 4
