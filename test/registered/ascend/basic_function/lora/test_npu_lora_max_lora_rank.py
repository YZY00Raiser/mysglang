import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
# from sglang.test.ascend.test_ascend_utils import (
#     LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH,
#     LLAMA_3_2_1B_WEIGHTS_PATH,
# )
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

LLAMA_3_2_1B_WEIGHTS_PATH = "/home/weights/LLM-Research/Llama-3.2-1B-Instruct"
LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH = "/home/weights/codelion/Llama-3.2-1B-Instruct-tool-calling-lora"
LLAMA_3_2_1B_INSTRUCT_TOOL_FAST_LORA_WEIGHTS_PATH = "/home/weights/codelion/FastLlama-3.2-LoRA"

'''
class TestLoraMaxLoraRank(CustomTestCase):
    """Testcase：Verify set the --max-load-rank parameter, can load lora corresponding to the number of ranks, inference request succeeded.

    [Test Category] Parameter
    [Test Target] --max-load-rank
    """

    lora_a = LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH
    max_lora_rank = "64"

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--tp-size",
            "1",
            "--enable-lora",
            "--lora-path",
            f"lora_a={cls.lora_a}",
            "--max-lora-rank",
            cls.max_lora_rank,
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]
        cls.process = popen_launch_server(
            LLAMA_3_2_1B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_lora_max_lora_rank(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "lora_path": "lora_a",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)
        response = requests.get(DEFAULT_URL_FOR_TEST + "/server_info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["max_lora_rank"], 64)

'''


class TestLoraMaxLoraRank(CustomTestCase):
    """Testcase：Verify set the --max-load-rank parameter, can load lora corresponding to the number of ranks, inference request succeeded.

    [Test Category] Parameter
    [Test Target] --max-load-rank
    """

    lora_a = LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH
    max_lora_rank = "64"
    def test_max_loaded_loras_error(self):
        error_message = "The number of LoRA paths should not exceed max_loaded_loras."
        other_args = [
            "--tp-size",
            "1",
            "--enable-lora",
            "--lora-path",
            f"lora_a={self.lora_a}",
            "--max-lora-rank",
            self.max_lora_rank,
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]
        out_log_file = open("./cache_out_log.txt", "w+", encoding="utf-8")
        err_log_file = open("./cache_err_log.txt", "w+", encoding="utf-8")
        try:
            popen_launch_server(
                LLAMA_3_2_1B_WEIGHTS_PATH,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=other_args,
                return_stdout_stderr=(out_log_file, err_log_file),
            )
        except Exception as e:
            print(f"Server launch failed as expects:{e}")
        finally:
            err_log_file.seek(0)
            content = err_log_file.read()
            # error_message information is recorded in the error log
            self.assertIn(error_message, content)
            out_log_file.close()
            err_log_file.close()
            os.remove("./cache_out_log.txt")
            os.remove("./cache_err_log.txt")

if __name__ == "__main__":
    unittest.main()
