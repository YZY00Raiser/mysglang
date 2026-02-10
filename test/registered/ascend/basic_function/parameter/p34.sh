#!/bin/bash
# P节点：192.168.0.188（Prefill节点）- 1P1D架构
# 系统性能优化配置
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
export SGLANG_SET_CPU_AFFINITY=1
unset ASCEND_LAUNCH_BLOCKING

# 昇腾环境变量加载（补充op_api库路径，修正原脚本exportLD语法错误）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest

# 核心环境变量配置
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16
export HCCL_BUFFSIZE=2800  # 原脚本保留的最优值，注释3200备用
export HAS_INDEX_K=1
export SGLANG_DEEPEP_BF16_DISPATCH=0
export SGLANG_NPU_USE_MLAPO=0
export SGLANG_USE_AG_AFTER_QLORA=0
export USE_MULTI_STREAM=1
export ENABLE_MOE_NZ=1
export PROFILING_MODE=dynamic
export HCCL_OP_EXPANSION_MODE=AIV

# 1P1D网络通信配置（关键：MF存储地址指向D节点192.168.0.244，网卡与D节点保持一致）
export ASCEND_MF_STORE_URL="tcp://192.168.0.188:24667"
export HCCL_SOCKET_IFNAME="enp23s0f3"
export GLOO_SOCKET_IFNAME="enp23s0f3"

# 启动SGLang服务（1P1D核心配置：nnodes=1、node-rank=0，删除多余dist-init-addr，修正量化参数笔误）
python -m sglang.launch_server \
--model-path /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V3.2-W8A8 \
--tp 16 \
--trust-remote-code \
--attention-backend ascend \
--device npu \
--watchdog-timeout 9000 \
--host 192.168.0.244 \
--port 30000 \
--mem-fraction-static 0.8 \
--max-total-tokens 68000 \
--context-length 68000 \
--chunked-prefill-size 327680 \
--max-prefill-tokens 68000 \
--max-running-requests 16 \
--moe-a2a-backend deepep \
--deepep-mode auto \
--quantization modelslim \
--disaggregation-transfer-backend ascend \
--disaggregation-mode prefill \
--disable-cuda-graph \
--nnodes 1 \
--node-rank 0 \
--disable-overlap-schedule \
--enable-hierarchical-cache \
--hicache-ratio 1.2 \
--hicache-size 0 \
--hicache-write-policy write_through \
--hicache-storage-backend file \
--hicache-storage-prefetch-policy wait_complete \
--mem-fraction-static 0.8
