export ASCEND_MF_STORE_URL="tcp://172.22.3.19:24667"
python3 -m sglang.launch_server \
--model-path /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V2-Lite-W8A8 \
--mem-fraction-static 0.8 \
--host 172.22.3.19 \
--port 8000 \
--disaggregation-mode prefill \
--disaggregation-bootstrap-port 8996 \
--disaggregation-transfer-backend ascend \
--trust-remote-code \
--tp-size 2 \
--attention-backend ascend \
--disable-radix-cache \
--disable-cuda-graph \
--device npu \
--quantization modelslim \
--chunked-prefill-size -1
