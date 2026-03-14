import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH,
    LLAMA_3_2_1B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestMaxLoadedLorasError(CustomTestCase):
    """Testcase: Test The number of LoRA paths should exceed max_loaded_loras

    [Test Category] --lora-backend
    [Test Target] --max-loaded-loras
    """

    def test_max_loaded_loras_error(self):

        other_args = [
            "--enable-lora",
            "--max-loaded-loras",
            3,
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            0.8,
            "--max-loras-per-batch",
            1,
            "--lora-path",
            f"lora_1={LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH}",
            f"lora_2={LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH}",
            f"lora_3={LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH}",
            f"lora_4={LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH}",
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
            self.assertIn(
                "Server process exited with code 1. Check server logs for errors.",
                str(e),
            )
            print(e)
        finally:
            err_log_file.seek(0)
            content = err_log_file.read()
            # error_message information is recorded in the error log
            error_message = "The number of LoRA paths should not exceed max_loaded_loras."
            self.assertIn(error_message, content)
            out_log_file.close()
            err_log_file.close()
            os.remove("./cache_out_log.txt")
            os.remove("./cache_err_log.txt")

        with self.assertRaises(Exception) as ctx:
            popen_launch_server(
                LLAMA_3_2_1B_WEIGHTS_PATH,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=other_args,
                return_stdout_stderr=(out_log_file, err_log_file),
            )
        self.assertIn("Server process exited with code 1. Check server logs for errors.", str(ctx.exception))



if __name__ == "__main__":
    unittest.main()
