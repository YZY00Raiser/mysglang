export ASCEND_MF_STORE_URL="tcp://172.22.3.34:24667"
python3 -m sglang.launch_server \
--model-path /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V3.2-W8A8 \
--mem-fraction-static 0.75 \
--host 172.22.3.244 --port 8000 \
--disaggregation-mode decode \
--disaggregation-transfer-backend ascend \
--trust-remote-code \
--tp-size 16 \
--attention-backend ascend \
--device npu \
--chunked-prefill-size -1 \
--quantization modelslim \
--page-size 128 \
--disaggregation-decode-enable-offload-kvcache \
--hicache-storage-backend file

