import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
# from sglang.test.ascend.test_ascend_utils import DEEPSEEK_R1_0528_W8A8_WEIGHTS_PATH as MODEL_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=200, suite="nightly-16-npu-a3", nightly=True)


class TestEplbAlgorithm(CustomTestCase):
    """
    Testcase：Verify the correctness of the inference service when --eplb-algorithm is set

    [Test Category] Parameter
    [Test Target] --eplb-algorithm
    """

    eplb_algorithm = "deepseek"

    @classmethod
    def setUpClass(cls):
        # cls.model = MODEL_PATH
        cls.model = "/root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-R1-0528-W8A8"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "16",
                "--dp-size",
                "1",
                "--attention-backend",
                "ascend",
                "--quantization",
                "modelslim",
                "--mem-fraction-static",
                "0.9",
                "--enable-dp-attention",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "normal",
                "--disable-cuda-graph",
                "--enable-eplb",
                "--ep-num-redundant-experts",
                "16",
                "--eplb-rebalance-num-iterations",
                "50",
                "--expert-distribution-recorder-buffer-size",
                "50",
                "--enable-expert-distribution-metrics",
                "--expert-distribution-recorder-mode",
                "stat",
                "--ep-dispatch-algorithm",
                "static",
                "--eplb-algorithm",
                cls.eplb_algorithm,
            ],
            env={
                "SGL_ENABLE_JIT_DEEPGEMM": "0",
                "HCCL_BUFFSIZE": "512",
                **os.environ,
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_eplb_algorithm(self):
        response = requests.get(f"{self.base_url}/get_server_info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.eplb_algorithm, response.json().get("eplb_algorithm"))

        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 64,
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)


if __name__ == "__main__":
    unittest.main()
