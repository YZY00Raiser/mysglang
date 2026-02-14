from types import SimpleNamespace
from sglang.test.few_shot_gsm8k import run_eval


def gsm8k():
    args = SimpleNamespace(
        num_shots=5,
        data_path="./test.jsonl",
        num_questions=200,
        max_new_tokens=512,
        parallel=128,
        host=f"http://172.22.3.19",
        port=6688,
    )
    accuracy_list = []
    total_runs = 5
    for i in range(total_runs):
        print(f"\n===== 开始执行第 {i + 1} 次 GSM8K 测试 =====")
        try:
            metrics = run_eval(args)
            accuracy = metrics['accuracy']
            accuracy_list.append(accuracy)
            print(f"第 {i + 1} 次测试完成，精度: {accuracy:.4f}")
        except Exception as e:
            # 捕获单次测试异常，避免整体中断
            print(f"第 {i + 1} 次测试执行失败: {str(e)}")

    avg_accuracy = sum(accuracy_list) / len(accuracy_list)
    print(f"\n===== 测试结果汇总 =====")
    print(f"各次精度结果: {[f'{acc:.4f}' for acc in accuracy_list]}")
    print(f"五次测试精度平均值: {avg_accuracy:.4f}")


if __name__ == "__main__":
    gsm8k()
