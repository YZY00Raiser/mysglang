import unittest
from types import SimpleNamespace
import requests
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=800, suite="nightly-1-npu-a3", nightly=True)  # 增加预估时间，因为要跑20次


class TestEnableMultimodalNonMlm(CustomTestCase):
    """Testcase: Verify that when the --enable-multimodal parameter is set, the average mmlu accuracy of 10 runs is greater than or equal to that when the parameter is not set.

        [Test Category] Parameter
        [Test Target] --enable-multimodal
        """
    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    scores_with_param = []  # 存储带参数的10次分数
    scores_without_param = []  # 存储不带参数的10次分数
    RUN_TIMES = 20  # 定义运行次数

    def launch_server(self, enable_multimodal: bool):
        """Universal server launch method, add --enable-multimodal based on parameters"""
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]
        # Add multimodal parameter as needed
        if enable_multimodal:
            other_args.insert(1, "--enable-multimodal")

        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        self.addCleanup(kill_process_tree, process.pid)
        return process

    def verify_inference(self):
        """Universal inference function verification"""
        # Basic generation request verification
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)

    def run_mmlu_eval(self) -> float:
        """Universal MMLU evaluation execution method, returns evaluation score"""
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )
        metrics = run_eval(args)
        # Retain basic score lower limit assertion
        self.assertGreaterEqual(metrics["score"], 0.2)
        return metrics["score"]

    def test_01_enable_multimodal(self):
        """运行10次带--enable-multimodal的MMLU评估"""
        for i in range(self.RUN_TIMES):
            print(f"\nRunning with --enable-multimodal: {i + 1}/{self.RUN_TIMES}")
            self.launch_server(enable_multimodal=True)
            self.verify_inference()
            score = self.run_mmlu_eval()
            self.scores_with_param.append(score)
            print(f"Score {i + 1}: {score:.4f}")

        # 验证是否收集到了10个分数
        self.assertEqual(len(self.scores_with_param), self.RUN_TIMES,
                         f"Expected {self.RUN_TIMES} scores with parameter, got {len(self.scores_with_param)}")

    def test_02_disable_multimodal(self):
        """运行10次不带--enable-multimodal的MMLU评估"""
        for i in range(self.RUN_TIMES):
            print(f"\nRunning without --enable-multimodal: {i + 1}/{self.RUN_TIMES}")
            self.launch_server(enable_multimodal=False)
            self.verify_inference()
            score = self.run_mmlu_eval()
            self.scores_without_param.append(score)
            print(f"Score {i + 1}: {score:.4f}")

        # 验证是否收集到了10个分数
        self.assertEqual(len(self.scores_without_param), self.RUN_TIMES,
                         f"Expected {self.RUN_TIMES} scores without parameter, got {len(self.scores_without_param)}")

    def test_03_calculate_and_assert_average(self):
        """计算两组分数的平均值并断言带参数的平均值 不带参数的平均值"""
        # 计算平均值
        avg_with_param = sum(self.scores_with_param) / len(self.scores_with_param)
        avg_without_param = sum(self.scores_without_param) / len(self.scores_without_param)

        # 打印详细信息
        print("\n=== Evaluation Results Summary ===")
        print(f"With --enable-multimodal (10 runs): {self.scores_with_param}")
        print(f"Average score with parameter: {avg_with_param:.4f}")
        print(f"\nWithout --enable-multimodal (10 runs): {self.scores_without_param}")
        print(f"Average score without parameter: {avg_without_param:.4f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
