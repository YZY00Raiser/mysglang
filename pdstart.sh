python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://<PREFILL_HOST_IP>:8000 8996 \
    --decode http://<DECODE_HOST_IP>:8001 \
    --host 127.0.0.1 \
    --port 6688
