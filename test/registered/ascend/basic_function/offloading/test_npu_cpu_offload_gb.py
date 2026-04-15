import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
# from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)

QWEN3_32B_WEIGHTS_PATH="/home/weights/Qwen/Qwen3-32B"
class TestNpuCpuOffloadGb(CustomTestCase):
    """Testcase: Tests core functionality with --cpu-offload-gb configuration, inference requests successful.
    and the inference accuracy using the GSM8K dataset is no less than 0.86.

    [Test Category] Parameter
    [Test Target] --cpu-offload-gb
    """

    @classmethod
    def setUpClass(cls):
        cls.other_args = [
            "--cpu-offload-gb",
            10,
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--tp-size",
            2,
            "--mem-fraction-static",
            0.8,
            "--base-gpu-id",
            "12",
        ]
        cls.process = popen_launch_server(
            QWEN3_32B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            kill_process_tree(cls.process.pid)

    def test_cpu_offload_gb_basic_inference(self):
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

        args = SimpleNamespace(
            max_new_tokens=512,
            base_url=DEFAULT_URL_FOR_TEST,
            model=QWEN3_32B_WEIGHTS_PATH,
            eval_name="gsm8k",
            api="completion",
            num_examples=200,
            num_threads=128,
            num_shots=5,
        )
        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.86)


if __name__ == "__main__":
    unittest.main()
