import unittest
import openai
import requests
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestRequestLengthValidationGenerate(CustomTestCase):
    """
    Test threshold
    """
    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--max-total-tokens", "110",
            "--context-length", "100"
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

    def test_context_length_success(self):
        print("==============startoooo====================================")
        input_ids = [1] * 99
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                },
            },
        )
        print("==============respob=====================================")
        print(response.json())
        print("=============finshhhhhhhhhhhhhhh====================================")
        self.assertEqual(response.status_code, 200)

    # def test_max_token_success(self):
    #     input_ids = [1] * 999
    #     response = requests.post(
    #         f"{DEFAULT_URL_FOR_TEST}/generate",
    #         json={
    #             "input_ids": input_ids,
    #             "sampling_params": {
    #                 "temperature": 0,
    #                 "max_tokens": 1,
    #             },
    #         },
    #     )
    #     self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
