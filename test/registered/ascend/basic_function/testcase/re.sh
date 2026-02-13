curl "http://127.0.0.1:3000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The capital of France is",
    "sampling_params": {
      "temperature": 0,
      "max_new_tokens": 32
    }
  }'
