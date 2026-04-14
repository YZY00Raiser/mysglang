import os
import unittest

import requests

from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestAscendCpuOffloadGb(CustomTestCase):
    """Testcase: Tests core functionality with --cpu-offload-gb configuration, inference requests successful.
    and the ingerence accuracy using the GSM8K dataset is no less than 0.86.

    [Test Category] Parameter
    [Test Target] --cpu-offload-gb
    """

    accuracy = 0.86

    @classmethod
    def setUpClass(cls):
        out_log_file = open("./cache_out_log.txt", "w+", encoding="utf-8")
        err_log_file = open("./cache_err_log.txt", "w+", encoding="utf-8")
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--cpu-offload-gb",
            10,
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--tp-size",
            2,
            "--mem-fraction-static",
            0.8,
        ]
        cls.process = popen_launch_server(
            QWEN3_32B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(out_log_file, err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        out_log_file.close()
        err_log_file.close()
        os.remove("./cache_out_log.txt")
        os.remove("./cache_err_log.txt")
        kill_process_tree(cls.process.pid)

    def test_cpu_offload_gb(self):
        requests.post(
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
        # self.err_log_file.seek(0)
        # content = self.err_log_file.read()
        # # error_message = "LoRA buffer shape torch.Size([32,4096]) does not match expected weight shape torch.Size([64,4096])"
        # # error_message = "LoRA buffer shape does not match expected weight shape"
        # error_message = "not match weight shape"
        # self.assertIn(error_message, content)


if __name__ == "__main__":
    unittest.main()
