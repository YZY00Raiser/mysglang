curl --location 'http://127.0.0.1:30000/generate' \
--header 'Content-Type:application/json' \
--data '{"text":"The capital of France is", "sampling_params":{"temperature":0, "max_new_tokens":100}}'

