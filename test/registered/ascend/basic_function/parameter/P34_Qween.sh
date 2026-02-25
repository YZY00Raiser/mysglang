export ASCEND_MF_STORE_URL="tcp://172.22.3.34:24667"
python3 -m sglang.launch_server \
--model-path /root/.cache/modelscope/hub/models/Qwen/Qwen3-32B \
--mem-fraction-static 0.75 \
--host 172.22.3.34 \
--port 8000 \
--disaggregation-mode prefill \
--disaggregation-bootstrap-port 8996 \
--disaggregation-transfer-backend ascend \
--disable-cuda-graph \
--trust-remote-code \
--tp-size 4 \
--attention-backend ascend \
--enable-hierarchical-cache \
--hicache-storage-backend file \
--device npu


