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


class TestLoraOverlapLoadingDisabled(CustomTestCase):
    """Testcase：Verify LoRA works properly without --enable-lora-overlap-loading, Switch lora TTFT < Switch lora TTFT with
    --enable-lora-overlap-loading.

    [Test Category] Parameter
    [Test Target] --enable-lora-overlap-loading
    """

    unable_overlap_loading_time = 0
    other_args = [
        "--tp-size",
        "2",
        "--enable-lora",
        "--lora-path",
        f"lora_a={LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH}",
        f"lora_b={LLAMA_3_2_1B_INSTRUCT_TOOL_FAST_LORA_WEIGHTS_PATH}",
        "--max-loaded-loras",
        "2",
        "--max-loras-per-batch",
        "2",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--base-gpu-id",
        "2",
    ]

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_lora_without_overlap_loading(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is" * 800,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1,
                },
                "lora_path": "lora_a",
            },
        )
        self.assertEqual(response.status_code, 200)
        # self.assertIn("Paris", response.text)
        print("--------------------e2e-latency------1--------------------------")
        print(response.json()["meta_info"]["e2e_latency"])

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is" * 800,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1,
                },
                "lora_path": "lora_b",
            },
        )
        self.assertEqual(response.status_code, 200)
        # self.assertIn("Paris", response.text)
        TestLoraOverlapLoadingDisabled.unable_overlap_loading_time = response.json()["meta_info"]["e2e_latency"]
        print("--------------------e2e-latency------2--------------------------")
        print(response.json()["meta_info"]["e2e_latency"])


class TestLoraOverlapLoadingEnabled(CustomTestCase):
    """Testcase：Verify LoRA works properly without --enable-lora-overlap-loading, Switch lora TTFT < Switch lora TTFT with
    --enable-lora-overlap-loading.

    [Test Category] Parameter
    [Test Target] --enable-lora-overlap-loading
    """

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--tp-size",
            "2",
            "--enable-lora",
            "--lora-path",
            f"lora_a={LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH}",
            f"lora_b={LLAMA_3_2_1B_INSTRUCT_TOOL_FAST_LORA_WEIGHTS_PATH}",
            "--max-loaded-loras",
            "2",
            "--max-loras-per-batch",
            "2",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--enable-lora-overlap-loading",
            "--base-gpu-id",
            "2",
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

    def test_lora_with_overlap_loading(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is" * 800,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1,
                },
                "lora_path": "lora_a",
            },
        )
        self.assertEqual(response.status_code, 200)
        # self.assertIn("Paris", response.text)
        print("--------------------e2e-latency------3--------------------------")
        print(response.json()["meta_info"]["e2e_latency"])
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is" * 800,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1,
                },
                "lora_path": "lora_b",
            },
        )
        self.assertEqual(response.status_code, 200)
        # self.assertIn("Paris", response.text)
        enable_overlap_loading_time = response.json()["meta_info"]["e2e_latency"]
        print("--------------------e2e-latency------4--------------------------")
        print(response.json()["meta_info"]["e2e_latency"])
        self.assertGreaterEqual(TestLoraOverlapLoadingDisabled.unable_overlap_loading_time, enable_overlap_loading_time)


if __name__ == "__main__":
    unittest.main()
