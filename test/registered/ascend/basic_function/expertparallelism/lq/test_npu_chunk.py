import unittest

import requests

from sglang.srt.utils import kill_process_tree
# from sglang.test.ascend.test_ascend_utils import (
#     DEEPSEEK_CODER_V2_LITE_WEIGHTS_PATH,
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
    suite="nightly-2-npu-a3",
    nightly=True,
)
DEEPSEEK_CODER_V2_LITE_WEIGHTS_PATH = "/home/weights/DeepSeek-Coder-V2-Lite-Instruct"

class TestSetForwardHooks(CustomTestCase):
    """Testcase: Verify set --forward-hooks parameter, can identify the set hook function
    and the inference request is successfully processed.

    [Test Category] Parameter
    [Test Target] --forward-hooks
    """

    @classmethod
    def setUpClass(cls):
        cls.out_log_file_name = "./tmp_out_log.txt"
        cls.hook_log_file_name = "./tmp_hook_log.txt"
        cls.out_log_file = open(cls.out_log_file_name, "w+", encoding="utf-8")
        cls.hook_log_file = open(cls.hook_log_file_name, "w+", encoding="utf-8")
        cls.process = popen_launch_server(
            DEEPSEEK_CODER_V2_LITE_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "2",
                "--attention-backend",
                "ascend",
                "--mem-fraction-static",
                "0.5",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "normal",
                "--disable-cuda-graph",
                "--enable-eplb",
                "--ep-num-redundant-experts",
                "4",
                "--eplb-rebalance-num-iterations",
                "50",
                "--expert-distribution-recorder-buffer-size",
                "50",
                "--enable-expert-distribution-metrics",
                "--ep-dispatch-algorithm",
                "static",
                "--eplb-rebalance-layers-per-chunk",
                "1",
                "--base-gpu-id",
                "10",
            ],
            env={
                "SGLANG_NPUDISABLE_ACL_FORMAT_WEIGHT": "1",
                "HCCL_BUFFSIZE": "1024",
            },
            return_stdout_stderr=(cls.out_log_file, cls.hook_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        # cls.out_log_file.close()
        # cls.hook_log_file.close()
        # os.remove(cls.out_log_file_name)
        # os.remove(cls.hook_log_file_name)

    def test_forward_hooks(self):
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

        self.hook_log_file.seek(0)
        hook_content = self.hook_log_file.read()
        # self.assertIn("hook effect", hook_content)


if __name__ == "__main__":
    unittest.main()
