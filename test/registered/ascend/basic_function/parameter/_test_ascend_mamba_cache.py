import unittest
from types import SimpleNamespace
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
# from sglang.test.ascend.test_ascend_utils import QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_FOR_TEST
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)
QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_FOR_TEST="/home/weights/Qwen/Qwen3-Next-80B-A3B-Instruct-W8A8"

class TestMambaCache(GSM8KAscendMixin, CustomTestCase):
    """Testcase：Verify the MambaCache

    [Test Category] Parameter
    [Test Target] --lora-target-modules
    """

    model = QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_FOR_TEST
    accuracy = 0.92
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        # "--max-mamba-cache-size",
        # "None",
        "--mamba-ssm-dtype",
        "float32",
        "--mamba-full-memory-ratio",
        "0.9",
        "--mamba-scheduler-strategy",
        "auto",
        "--mamba-track-interval",
        "256",
        "--tp-size",
        4,
        "--disable-radix-cache"
    ]
    # args = SimpleNamespace(
    #     num_shots=5,
    #     data_path="./test.jsonl",
    #     num_questions=200,
    #     max_new_tokens=512,
    #     parallel=128,
    #     host=f"http://172.22.3.19",
    #     port=6688,
    # )

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=self.gsm8k_num_shots,
            data_path="./test.jsonl",
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        self.assertGreaterEqual(
            metrics["accuracy"],
            self.accuracy,
            f'Accuracy of {self.model} is {str(metrics["accuracy"])}, is lower than {self.accuracy}',
        )



if __name__ == "__main__":
    unittest.main()
