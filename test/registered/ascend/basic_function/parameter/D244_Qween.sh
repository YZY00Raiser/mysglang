export ASCEND_MF_STORE_URL="tcp://172.22.3.34:24667"
python3 -m sglang.launch_server \
--model-path /root/.cache/modelscope/hub/models/Qwen/Qwen3-32B \
--mem-fraction-static 0.75 \
--host 172.22.3.244 --port 8000 \
--disaggregation-mode decode \
--disaggregation-transfer-backend ascend \
--trust-remote-code \
--tp-size 4 \
--attention-backend ascend \
--disaggregation-decode-enable-offload-kvcache \
--hicache-storage-backend file \
--device npu

