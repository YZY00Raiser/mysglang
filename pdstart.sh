# 必须和 Prefill/Decode 节点的 IP 保持一致
export PREFILL_HOST_IP=172.22.3.154
export DECODE_HOST_IP=172.22.3.71

# 启动 SGLang 路由网关（自动转发 Prefill + Decode 请求）
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://${PREFILL_HOST_IP}:8000 8996 \
    --decode http://${DECODE_HOST_IP}:8001 \
    --host 0.0.0.0 \
    --port 6688
