# encoder 0
python -m sglang.launch_server \
  --model-path /home/weights/Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30000 \
  --enable-prefix-mm-cache \
  --base-gpu-id 4 
  
  
  
# language-only server
python -m sglang.launch_server \
  --model-path /home/weights/Qwen/Qwen3-VL-8B-Instruct \
  --language-only \
  --encoder-urls http://127.0.0.1:30000 \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30002 \
  --base-gpu-id 12 
