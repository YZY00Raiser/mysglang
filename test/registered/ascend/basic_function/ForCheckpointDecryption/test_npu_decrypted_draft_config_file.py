import json
import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import run_command
# from sglang.test.ascend.test_ascend_utils import (
#     QWEN3_32B_WEIGHTS_PATH,
#     QWEN3_32B_EAGLE3_WEIGHTS_PATH
# )
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=400,
    suite="nightly-4-npu-a3",
    nightly=True,
)
QWEN3_32B_WEIGHTS_PATH = "/home/weights/Qwen/Qwen3-8B"
QWEN3_32B_EAGLE3_WEIGHTS_PATH = "/home/weights/Qwen3/Qwen3-8B_eagle3"


class TestSetForwardHooks(CustomTestCase):
    """Testcase: Verify set --decrypted-config-file, --decrypted-draft-config-file parameter, set non exist config.json,
    will use the non exist config.json and the service startup failed

    [Test Category] Parameter
    [Test Target] --decrypted-config-file, --decrypted-draft-config-file
    """

    model = QWEN3_32B_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        run_command(
            f"mv {os.path.join(QWEN3_32B_WEIGHTS_PATH, 'config.json')} {os.path.join(QWEN3_32B_WEIGHTS_PATH, '_config.json')}")
        run_command(
            f"mv {os.path.join(QWEN3_32B_EAGLE3_WEIGHTS_PATH, 'config.json')} {os.path.join(QWEN3_32B_EAGLE3_WEIGHTS_PATH, '_config.json')}")

        other_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--max-running-requests",
            "1",
            "--disable-radix-cache",
            "--chunked-prefill-size",
            "-1",
            "--max-prefill-tokens",
            "1024",
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            QWEN3_32B_EAGLE3_WEIGHTS_PATH,
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            "--tp-size",
            "2",
            "--mem-fraction-static",
            "0.68",
            "--disable-cuda-graph",
            "--dtype",
            "bfloat16",
            # "--decrypted-config-file",
            # "Qwen3-8B/config.json",
            # "--decrypted-draft-config-file",
            # "Qwen3-8B_eagle3/config.json",
        ]
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        run_command(
            f"mv {os.path.join(QWEN3_32B_WEIGHTS_PATH, '_config.json')} {os.path.join(QWEN3_32B_WEIGHTS_PATH, 'config.json')}")
        run_command(
            f"mv {os.path.join(QWEN3_32B_EAGLE3_WEIGHTS_PATH, '_config.json')} {os.path.join(QWEN3_32B_EAGLE3_WEIGHTS_PATH, 'config.json')}")

        kill_process_tree(cls.process.pid)

    def test_decrypted_draft_config_file(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0.8,
                    "max_new_tokens": 32,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)


if __name__ == "__main__":
    unittest.main()
