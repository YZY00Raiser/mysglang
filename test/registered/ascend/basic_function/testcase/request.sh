curl -X POST \
  "http://localhost:33890/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The capital of France is",
    "sampling_params": {
      "temperature": 0,
      "max_new_tokens": 32
    }
  }'
