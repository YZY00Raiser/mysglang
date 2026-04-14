from types import SimpleNamespace
from sglang.test.few_shot_gsm8k import run_eval

def gsm8k():
    args = SimpleNamespace(
        num_shots=5,
        data_path=None,
        num_questions=200,
        max_new_tokens=512,
        parallel=32,
        host=f"http://127.0.0.1",
        port=30000,
    )
    metrics = run_eval(args)
    print(f"{metrics=}")
    print(f"{metrics['accuracy']=}")
if __name__ == "__main__":
    gsm8k()
