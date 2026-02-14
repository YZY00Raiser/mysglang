#export SGLANG_DP_ROUND_ROBIN=1
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://172.22.3.34:8000 8996 \
    --decode http://172.22.3.188:8000 \
    --host 172.22.3.34 \
    --port 6688
