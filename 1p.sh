# ===================== 基础环境配置 =====================
# 模型路径（DeepSeek-R1 4bit量化模型）
export MODEL_PATH=/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/DeepSeek-R1-0528-w4a8-per-channel
# NPU内存分配优化
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# 单设备并发流数
export STREAMS_PER_DEVICE=32
# Prefill节点IP + 服务端口
export PREFILL_HOST_IP=172.22.3.154
export PORT=8000

# ===================== MemFabric 存储配置 =====================
# 修正：正确引用环境变量
export ASCEND_MF_STORE_URL="tcp://${PREFILL_HOST_IP}:${PORT}"

# ===================== DeepEp 通信/量化配置 =====================
# 开启INT8量化
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
# HCCL通信缓冲区大小（昇腾集合通信）
export HCCL_BUFFSIZE=1536

# ===================== NPU算子加速配置 =====================
# 启用昇腾MLAPO算子加速
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_USE_FIA_NZ=1
# 任务队列优化
export TASK_QUEUE_ENABLE=2

# ===================== 启动SGLang 解耦Prefill服务 =====================
python -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --host $PREFILL_HOST_IP \
    --port 8000 \
    # 核心：解耦推理 - 仅运行Prefill阶段
    --disaggregation-mode prefill \
    # 解耦推理引导端口
    --disaggregation-bootstrap-port 8996 \
    # 昇腾专属传输后端
    --disaggregation-transfer-backend ascend \
    --trust-remote-code \
    # 集群节点数
    --nnodes 1 \
    --node-rank 0 \
    # Tensor并行维度 16
    --tp-size 16 \
    # 静态内存占比 60%
    --mem-fraction-static 0.6 \
    # 昇腾注意力算子
    --attention-backend ascend \
    # 设备类型：NPU
    --device npu \
    # 模型量化方式：modelslim
    --quantization modelslim \
    # 负载均衡策略
    --load-balance-method round_robin \
    # 最大并发请求数
    --max-running-requests 8 \
    # 模型上下文长度
    --context-length 8192 \
    # 禁用radix缓存（解耦推理场景要求）
    --disable-radix-cache \
    # 分块预填充大小
    --chunked-prefill-size -1 \
    # 最大预填充token数
    --max-prefill-tokens 28680 \
    # MOE模型通信后端
    --moe-a2a-backend deepep \
    --deepep-mode normal \
    # 推测解码算法
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    # 数据并行维度 2
    --dp-size 2 \
    # 开启数据并行注意力
    --enable-dp-attention \
    # 禁用共享专家融合
    --disable-shared-experts-fusion \
    # 计算精度
    --dtype bfloat16
