#export SGLANG_DP_ROUND_ROBIN=1
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://172.22.3.19:8000 8996 \
    --decode http://172.22.3.244:8000 \
    --host 172.22.3.19 \
    --port 6688
