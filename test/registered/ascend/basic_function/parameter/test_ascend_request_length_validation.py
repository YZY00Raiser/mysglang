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


class TestRequestLengthValidation(CustomTestCase):
    """Testcase：Verify set --max-total-tokens and --context-length, can correctly reject inference requests
    that exceed the limits and throw the specified exceptions.

       [Test Category] Parameter
       [Test Target] --max-total-tokens, --context-length
       """
    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=["--max-total-tokens", "1000", "--context-length", "1000"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_input_length_no_longer_than_context_length_success(self):
        client = openai.Client(api_key=self.api_key, base_url=f"{self.base_url}/v1")
        long_text = "hello " * 500
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": long_text},
            ],
            temperature=0,
        )
        completions_tokens = response.usage.completion_tokens
        self.assertGreater(completions_tokens, 0)

    def test_input_length_longer_than_context_length(self):
        client = openai.Client(api_key=self.api_key, base_url=f"{self.base_url}/v1")
        long_text = "hello " * 1200
        with self.assertRaises(openai.BadRequestError) as cm:
            client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": long_text},
                ],
                temperature=0,
            )
        self.assertIn("is longer than the model's context length", str(cm.exception))

    def test_not_longer_max_tokens_validation_success(self):
        client = openai.Client(api_key=self.api_key, base_url=f"{self.base_url}/v1")
        long_text = "hello "
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": long_text},
            ],
            temperature=0,
            max_tokens=800,
        )
        completions_tokens = response.usage.completion_tokens
        self.assertGreater(completions_tokens, 0)

    def test_longer_max_tokens_validation(self):
        client = openai.Client(api_key=self.api_key, base_url=f"{self.base_url}/v1")
        long_text = "hello "
        with self.assertRaises(openai.BadRequestError) as cm:
            client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": long_text},
                ],
                temperature=0,
                max_tokens=1001,
            )
        self.assertIn(
            "max_completion_tokens is too large",
            str(cm.exception),
        )


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
            "--max-total-tokens", "1000",
            "--context-length", "1000"
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

    # def test_context_length_success(self):
    #     input_ids = [100] * 999
    #     response = requests.post(
    #         f"{DEFAULT_URL_FOR_TEST}/generate",
    #         json={
    #             "input_ids": input_ids,
    #             "sampling_params": {
    #                 "temperature": 0,
    #             },
    #         },
    #     )
    #     self.assertEqual(response.status_code, 200)

    def test_max_token_success(self):
        input_ids = [1] * 999
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_tokens": 1,
                },
            },
        )
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
