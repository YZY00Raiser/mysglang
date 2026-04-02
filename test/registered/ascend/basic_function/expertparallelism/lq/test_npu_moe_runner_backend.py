import unittest

import requests

from sglang.srt.utils import kill_process_tree
# from sglang.test.ascend.test_ascend_utils import DEEPSEEK_CODER_V2_LITE_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)
DEEPSEEK_CODER_V2_LITE_WEIGHTS_PATH="/home/weights/DeepSeek-Coder-V2-Lite-Instruct"

class TestMoreRunnerBackendTriton(CustomTestCase):
    """Testcase：Verify set --moe-runner-backend, the inference request is successfully processed.

    [Test Category] Parameter
    [Test Target] --moe-runner-backend
    """

    model = DEEPSEEK_CODER_V2_LITE_WEIGHTS_PATH
    moe_runner_backend = "triton"

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                "0.85",
                "--tp-size",
                "2",
                "--ep",
                "2",
                "--enable-eplb",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "normal",
                "--ep-num-redundant-experts",
                "4",
                "--eplb-rebalance-num-iterations",
                "50",
                "--expert-distribution-recorder-buffer-size",
                "50",
                # "--expert-distribution-recorder-mode",
                # "stat",
                "--moe-runner-backend",
                cls.moe_runner_backend,
                "--base-gpu-id",
                "8",
            ],
            env={
                "SGLANG_NPUDISABLE_ACL_FORMAT_WEIGHT": "1",
                "HCCL_BUFFSIZE": "1024",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_moe_runner_backend(self):
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
        self.assertEqual(
            response.status_code, 200, "The request status code is not 200."
        )
        self.assertIn(
            "Paris", response.text, "The inference result does not include Paris."
        )
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/server_info")
        self.assertEqual(
            response.status_code, 200, "The request status code is not 200."
        )
        self.assertEqual(
            response.json()["moe_runner_backend"],
            self.moe_runner_backend,
            "--moe-runner-backend is not taking effect.",
        )


class TestMoreRunnerBackendTritonDefault(TestMoreRunnerBackendTriton):
    moe_runner_backend = "auto"


if __name__ == "__main__":
    unittest.main()
