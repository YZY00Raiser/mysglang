from types import SimpleNamespace
from sglang.test.few_shot_gsm8k import run_eval

def gsm8k():
    args = SimpleNamespace(
         num_shots=8,
         data_path="./test.jsonl",
         num_questions=200,
         max_new_tokens=512,
         parallel=32,
         host=f"http://172.22.3.19",
         port=8000,
    )
    metrics = run_eval(args)
    print(f"{metrics=}")
    print(f"{metrics['accuracy']=}")
if __name__ == "__main__":
    gsm8k()
