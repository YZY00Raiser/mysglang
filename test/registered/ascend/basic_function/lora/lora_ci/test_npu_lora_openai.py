import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH,
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
    LLAMA_3_2_1B_INSTRUCT_TOOL_FAST_LORA_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestLoRAOpenAICompatible(CustomTestCase):
    """Testcase：Test that model:adapter syntax takes precedence over explicit lora_path.

    [Test Category] Parameter
    [Test Target] --enable-lora;--lora-paths
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    lora_a = LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH
    lora_b = LLAMA_3_2_1B_INSTRUCT_TOOL_FAST_LORA_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--max-running-requests",
            "10",
            "--enable-lora",
            "--lora-paths",
            f"lora_a={cls.lora_a}",
            f"lora_b={cls.lora_b}",
            "--disable-radix-cache",
        ]

        cls.process = popen_launch_server(
            LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        cls.client = openai.Client(api_key="EMPTY", base_url=f"{DEFAULT_URL_FOR_TEST}/v1")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_priority_model_over_explicit_with_chat_completions_api(self):
        """Test that model:adapter syntax takes precedence over explicit lora_path."""
        # This test verifies the priority logic in _resolve_lora_path
        response1 = self.client.chat.completions.create(
            model=f"{self.model}:lora_a",
            messages=[{"role": "user", "content": "What tools do you have available?"}],
            max_tokens=32,
            temperature=0,
        )

        response2 = self.client.chat.completions.create(
            model=f"{self.model}:lora_a",
            messages=[{"role": "user", "content": "What tools do you have available?"}],
            extra_body={"lora_path": "lora_b"},
            max_tokens=32,
            temperature=0,
        )
        # Should use lora_a adapter (model parameter takes precedence)
        self.assertEqual(response1.choices[0].message.content, response2.choices[0].message.content)

        print("--------------------lora_a---------------------------")
        print(response1.choices[0].message.content)

    def test_priority_model_over_explicit_with_completions_api(self):
        """Test that model:adapter syntax takes precedence over explicit lora_path."""
        response1 = self.client.completions.create(
            model=f"{self.model}:lora_b",  # ← Using model:adapter syntax
            prompt="What tools do you have available?",
            max_tokens=32,
            temperature=0,
        )

        response2 = self.client.completions.create(
            model=f"{self.model}:lora_b",  # ← Using model:adapter syntax
            prompt="What tools do you have available?",
            extra_body={"lora_path": "lora_a"},
            max_tokens=32,
            temperature=0,
        )
        # Should use lora_a adapter (model parameter takes precedence)
        self.assertEqual(response1.choices[0].text, response2.choices[0].text)
        # Should use lora_b adapter (model parameter takes precedence)

        print("--------------------lora_b---------------------------")

        print(response1.choices[0].text)


if __name__ == "__main__":
    unittest.main()
