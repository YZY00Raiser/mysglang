export ASCEND_MF_STORE_URL="tcp://172.22.3.34:24667"
python3 -m sglang.launch_server \
--model-path /root/.cache/modelscope/hub/models/Qwen/Qwen3-32B \
--mem-fraction-static 0.75 \
--host 172.22.3.34 \
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
--page-size 128 \
--enable-hierarchical-cache \
--hicache-storage-backend file

