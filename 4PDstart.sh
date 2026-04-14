python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://172.22.3.71:8000 8998 \
    --prefill http://172.22.3.154:8000 8999 \
    --decode http://172.22.3.181:8001 \
    --host 127.0.0.1 \
    --port 6688 \
    --mini-lb
