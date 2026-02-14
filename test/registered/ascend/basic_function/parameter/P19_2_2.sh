export ASCEND_MF_STORE_URL="tcp://172.22.3.19:24667"
python3 -m sglang.launch_server \
--model-path /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V3.2-W8A8 \
--mem-fraction-static 0.8 \
--host 172.22.3.19 \
--port 8000 \
--disaggregation-mode prefill \
--disaggregation-bootstrap-port 8996 \
--disaggregation-transfer-backend ascend \
--trust-remote-code \
--tp-size 16 \
--attention-backend ascend \
--device npu \
--chunked-prefill-size -1 \
--watchdog-timeout 9000 \
--max-prefill-tokens 68000 \
--enable-hierarchical-cache \
--hicache-storage-backend file

