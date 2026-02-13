import requests

response = requests.post(
    f"http://127.0.0.1:33890/generate",
    json={
        "text": "The capital of France is",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 32,
        },
    },
)

print(response.json())



