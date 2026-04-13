python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://P_IP:8000 8998 \
    --prefill http://P_IP:8000 8999 \
    --decode http://D_IP:8001 \
    --host 127.0.0.1 \
    --port 6688 \
    --mini-lb
