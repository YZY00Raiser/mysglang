import unittest
from types import SimpleNamespace
import requests
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestEnableMultimodalNonMlm(CustomTestCase):
    """Testcase: Verify that when the --enable-multimodal parameter is set, mmlu accuracy greater than or equal 0.37

        [Test Category] Parameter
        [Test Target] --enable-multimodal
        """
    model = QWEN3_32B_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    score_with_param = None
    score_without_param = None
    accuracy=0.37

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--config", "comfig.yaml"
        ]

        cls.process = popen_launch_server(
            "--config", "comfig.yaml",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_enable_multimodal_func(self):
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



if __name__ == "__main__":
    unittest.main(verbosity=2)
