import json
import unittest

import requests

from sglang.srt.utils import kill_process_tree
# from sglang.test.ascend.test_ascend_utils import (
#     LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH,
#     LLAMA_3_2_1B_INSTRUCT_TOOL_FAST_LORA_WEIGHTS_PATH,
#     LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
# )
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)
LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH = "/home/weights/LLM-Research/Llama-3.2-1B-Instruct"
LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH = "/home/weights/codelion/Llama-3.2-1B-Instruct-tool-calling-lora"
LLAMA_3_2_1B_INSTRUCT_TOOL_FAST_LORA_WEIGHTS_PATH = "/home/weights/codelion/FastLlama-3.2-LoRA"


class TestLoRAOpenAI(CustomTestCase):
    """Testcase：Verify the use different lora, inference request succeeded.

    [Test Category] Parameter
    [Test Target] --enable-lora, --lora-path,
    """

    lora_a = LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH
    lora_b = LLAMA_3_2_1B_INSTRUCT_TOOL_FAST_LORA_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--enable-lora",
            "--lora-path",
            f"lora_a={cls.lora_a}",
            f"lora_b={cls.lora_b}",
            "--max-loaded-loras",
            "2",
            "--max-loras-per-batch",
            "2",
            "--lora-target-modules",
            "all",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.3",
            "--base-gpu-id",
            "4",
        ]
        cls.process = popen_launch_server(
            LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_completions_with_lora(self):
        # Test temperature parameter; temperature=0 yields identical outputs across requests, temperature=2 yields varied outputs
        response1 = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/v1/completions",
            json={"prompt": "who are you?", "temperature": 0, "lora_path": "lora_a"},
        )

        self.assertEqual(response1.status_code, 200, f"Failed with: {response1.text}")

        response2 = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/v1/completions",
            json={"prompt": "who are you?", "temperature": 0, "lora_path": "lora_b"},
        )

        self.assertEqual(response2.status_code, 200, f"Failed with: {response2.text}")
        # Asser that the configuration temperature is the same and the output response is the same
        print("------------------111111111111------------------------")
        print(response1.json())
        print("--------------------2222222222----------------------")
        print(response2.json())
        self.assertNotEqual(
            response1.json()["choices"][0]["text"],
            response2.json()["choices"][0]["text"],
        )

    def test_completions_chat_with_lora(self):
        # Test temperature parameter; temperature=0 yields identical outputs across requests, temperature=2 yields varied outputs
        response1 = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/v1/chat/completions",
            json={
                # "model": self.model,
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
                "temperature": 0,
                "lora_path": "lora_a"
            },
        )
        self.assertEqual(response1.status_code, 200, f"Failed with: {response1.text}")
        content1 = response1.json()["choices"][0]["message"]["content"]

        response2 = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/v1/chat/completions",
            json={
                # "model": self.model,
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
                "temperature": 0,
                "lora_path": "lora_b"
            },
        )
        print("------------------111111111111------------------------")
        print(response1.json())
        print("--------------------2222222222----------------------")
        print(response2.json())
        self.assertEqual(response2.status_code, 200, f"Failed with: {response2.text}")
        content2 = response2.json()["choices"][0]["message"]["content"]
        self.assertNotEqual(content1, content2)


if __name__ == "__main__":
    unittest.main()
