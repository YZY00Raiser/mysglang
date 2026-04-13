# CPU 性能模式
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 内核参数优化
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

# 绑核开关
export SGLANG_SET_CPU_AFFINITY=1

# 工作目录 & 环境变量
cd /home/wzy/0409/sgl-sglang
export PYTHONPATH=${PWD}/python:$PYTHONPATH

# 清空代理
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

# 昇腾环境加载
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /home/rjw/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/bin/set_env.bash

# 内存碎片优化
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

# 网卡绑定
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

# DeepEP 调度配置
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=330
export DEEPEP_NORMAL_LONG_SEQ_ROUND=5
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=3000
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1

# HCCL 通信配置
export HCCL_OP_EXPANSION_MODE=AIV
export TASK_QUEUE_ENABLE=1
export ASCEND_USE_FIA=1

# NPU 流控制
export SGLANG_NPU_USE_MULTI_STREAM=0

# 调试/性能
export ENABLE_PROFILING=0

# 模型路径
MODEL_PATH=/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/Qwen3-Next-80B-A3B-Instruct

# 服务预热
export SGLANG_WARMUP_TIMEOUT=3600
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export FORCE_DRAFT_MODEL_NON_QUANT=1

# ZBCCL 通信配置
export HCCL_BUFFSIZE=2000
unset PYTORCH_NPU_ALLOC_CONF
export ZBCCL_LOCAL_MEM_SIZE=60416
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
export ZBCCL_BOOTSTRAP_URL="tcp://127.0.0.1:24669"

# ZBCCL 内存分配
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ZBCCL_NPU_ALLOC_CONF=use_vmm_for_static_memory:True

# ZBCCL 图优化
export ZBCCL_ENABLE_GRAPH=1

# 启动 SGLang 服务
python3 -m sglang.launch_server \
  --model-path ${MODEL_PATH} \
  --page-size 128 \
  --tp-size 4 \
  --trust-remote-code \
  --attention-backend ascend \
  --device npu \
  --watchdog-timeout 9000 \
  --host 127.0.0.1 \
  --port 6699 \
  --mem-fraction-static 0.75 \
  --disable-radix-cache \
  --max-prefill-tokens 14080 \
  --context-length 26384 \
  --dp-size 2 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --speculative-draft-model-quantization unquant \
  --chunked-prefill-size -1 \
  --max-running-requests 312 \
  --cuda-graph-bs 2 4 16 32 48 64 80 96 128 140 156 \
  --mamba-ssm-dtype bfloat16 \
  --base-gpu-id 0 \
  --speculative-draft-model-path /data/ascend-ci-share-pkking-sglang/modelscope/hub/models/Qwen3-Next-80B-A3B-Instruct \
  --quantization modelslim \
  --moe-a2a-backend deepep \
  --deepep-mode auto
