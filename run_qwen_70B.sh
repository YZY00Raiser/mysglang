# cpu高性能
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
# 绑核
export SGLANG_SET_CPU_AFFINITY=1

#export PYTHONPATH=/home/c30058706/code/sglang/python:$PYTHONPATH

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
# 内存碎片
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
# 网卡
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

export HCCL_OP_EXPANSION_MODE=AIV
export TASK_QUEUE_ENABLE=1
export HCCL_BUFFSIZE=200
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15

# profiling
export SGLANG_NPU_PROFILING=0
export SGLANG_NPU_PROFILING_BS=8
export SGLANG_NPU_PROFILING_STAGE='decode'
export SGLANG_NPU_PROFILING_STEP=10

export SGLANG_MM_SKIP_COMPUTE_HASH=1
export SGLANG_ENABLE_SPEC_V2=1

python -m sglang.launch_server \
    --model-path /data/ascend-ci-share-pkking-sglang/modelscope/hub/models/Qwen/Qwen2.5-VL-72B-Instruct \
    --host 127.0.0.1 \
    --port 22001 \
    --tp 8 \
    --max-prefill-tokens 102400 \
    --chunked-prefill-size 102400 \
    --attention-backend ascend \
    --mem-fraction-static 0.6 \
    --enable-multimodal \
    --mm-attention-backend ascend_attn \
    --sampling-backend ascend \
    --max-running-requests 96 \
    --cuda-graph-bs 96 \
    --mm-enable-dp-encoder \
    --tokenizer-worker-num 4 \
    --skip-server-warmup 

#    --base-gpu-id 8
#2>&1 | tee fy_log.log &
#--enable-broadcast-mm-inputs-process \
#    --mm-enable-dp-encoder --enable-dp-attention --dp-size 8 
#    --speculative-algorithm EAGLE3 --speculative-draft-model-path /home/c30058706/Qwen2.5-VL-72B-Instruct-eagle3-sgl --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \

