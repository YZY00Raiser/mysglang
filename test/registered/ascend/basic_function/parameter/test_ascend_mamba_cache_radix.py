import unittest

import requests

from sglang.srt.utils import kill_process_tree
# from sglang.test.ascend.test_ascend_utils import QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_FOR_TEST
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase, DEFAULT_URL_FOR_TEST, popen_launch_server, \
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)
QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_FOR_TEST="/home/weights/Qwen3-Next-80B-A3B-Instruct-W8A8"

class TestMambaCache(CustomTestCase):
    """Testcase：Verify the MambaCache

    [Test Category] Parameter
    [Test Target] --lora-target-modules
    """

    model = QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_FOR_TEST
    @classmethod
    def setUpClass(cls):
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            # "--max-mamba-cache-size",
            # "127",
            # "--mamba-ssm-dtype",
            # "float32",
            # "--mamba-full-memory-ratio",
            # "0.9",
            # "--mamba-scheduler-strategy",
            # "auto",
            # "--mamba-track-interval",
            # "256",
            "--tp-size",
            "4",
            "--disable-radix-cache"
        ]
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_ascend_mamba_cache(self):
        # Verify that the inference API functions properly.
        response1 = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        print("---------------------------------response1.json()---------------------------------------------------")
        print(response1.json())
        # self.assertEqual(response.status_code, 200)
        # self.assertIn("Paris", response.text)
        response2 = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is Paris",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        print("---------------------------------response2.json()---------------------------------------------------")
        print(response2.json())

        response3 = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is Paris",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        print("---------------------------------response3.json()---------------------------------------------------")
        print(response3.json())


if __name__ == "__main__":
    unittest.main()
