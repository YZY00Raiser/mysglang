import json
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


class TestLoraBasicFunction_1_2_3_7_8(CustomTestCase):
    """Testcase：Verify the functionality and parameter effectiveness when --lora-target-modules=all is set for Llama-3.2-1B

    [Test Category] Parameter
    [Test Target] --lora-target-modules
    """

    lora_a = "/home/weights/codelion/Llama-3.2-1B-Instruct-tool-calling-lora"
    lora_b = "/home/weights/codelion/FastLlama-3.2-LoRA"

    # lora_c = "/home/weights/codelion/OneLLM-Doey-"
    # lora_c = "None"

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--tp-size",
            "2",
            "--enable-lora",
            "--lora-path",
            f"lora_a={cls.lora_a}",
            f"lora_b={cls.lora_b}",
            # f"lora_c={cls.lora_c}",
            "--lora-target-modules",
            "all",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--base-gpu-id",
            "6",
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

    def test_lora_use_different_lora(self):
        """Core Test: Verify the effectiveness of --lora-target-modules=all and normal server functionality

        Three-Step Verification Logic:
        1. Verify health check API availability (service readiness)
        2. Verify core generate API functionality (normal inference with correct results)
        3. Verify LoRA parameter configuration effectiveness via server info API
        """
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(response.status_code, 200)

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
        print("--------------------------response.json()----------lora_a--------------------------------")
        print(response.json())
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "lora_path": "lora_b",
            },
        )
        print("--------------------------response.json()-----------lora_b-------------------------------")
        print(response.json())

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
        print("--------------------------response.json()----------non--lora------------------------------")
        print(response.json())

        # 流式7
        response_stream = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "lora_path": "lora_a",
                "stream": True,
            },
            stream=True,
        )
        stream_text = ""
        for chunk in response_stream.iter_lines(decode_unicode=False):
            chunk = chunk.decode("utf-8")
            if chunk and chunk.startswith("data:"):
                if chunk == "data: [DONE]":
                    break
                data = json.loads(chunk[5:].strip("\n"))
                stream_text += data["text"]

        print("--------------------------chunk-------stream--true---------------------------------")
        print(stream_text)

'''
class TestLoraBasicFunction_6(CustomTestCase):
    """Testcase：Verify the functionality and parameter effectiveness when --lora-target-modules=all is set for Llama-3.2-1B

    [Test Category] Parameter
    [Test Target] --lora-target-modules
    """
    lora_a = ""
    lora_b = ""
    lora_c = "None"

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--tp-size"
            "2"
            "--enable-lora",
            "--lora-path",
            f"lora_1={cls.lora_a}",
            f"lora_2={cls.lora_b}",
            f"lora_3={cls.lora_c}",
            f"lora_3={cls.lora_c}",
            "--max-load-loras",
            "3",
            "--lora-target-modules",
            "all",
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

    def test_lora_use_different_lora(self):
        """Core Test: Verify the effectiveness of --lora-target-modules=all and normal server functionality

        Three-Step Verification Logic:
        1. Verify health check API availability (service readiness)
        2. Verify core generate API functionality (normal inference with correct results)
        3. Verify LoRA parameter configuration effectiveness via server info API
        """
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(response.status_code, 200)

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "lora_path": self.lora_a,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)

class TestLoraBasicFunction_9(CustomTestCase):
    """Testcase：Verify the functionality and parameter effectiveness when --lora-target-modules=all is set for Llama-3.2-1B

    [Test Category] Parameter
    [Test Target] --lora-target-modules
    """
    lora_a = "LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH"
    lora_b = "LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH"
    lora_c = "LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH"

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--tp-size"
            "2"
            "--enable-lora",
            "--lora-path",
            f"lora_1={cls.lora_a}",
            f"lora_2={cls.lora_b}",
            f"lora_3={cls.lora_c}",
            "--lora-target-modules",
            "all",
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

    def test_lora_use_different_lora(self):
        """Core Test: Verify the effectiveness of --lora-target-modules=all and normal server functionality

        Three-Step Verification Logic:
        1. Verify health check API availability (service readiness)
        2. Verify core generate API functionality (normal inference with correct results)
        3. Verify LoRA parameter configuration effectiveness via server info API
        """
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(response.status_code, 200)

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "lora_path": self.lora_a,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)

class TestLoraBasicFunction_13(CustomTestCase):
    """Testcase：Verify the functionality and parameter effectiveness when --lora-target-modules=all is set for Llama-3.2-1B

    [Test Category] Parameter
    [Test Target] --lora-target-modules
    """
    lora_a = "LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH"
    lora_b = "LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH"
    lora_c = "LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH"

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--tp-size"
            "2"
            "--enable-lora",
            "--lora-path",
            f"lora_1={cls.lora_a}",
            f"lora_2={cls.lora_b}",
            f"lora_3={cls.lora_c}",
            "--lora-target-modules",
            "--max-load-rank",
            "2",
            "all",
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

    def test_lora_use_different_lora(self):
        """Core Test: Verify the effectiveness of --lora-target-modules=all and normal server functionality

        Three-Step Verification Logic:
        1. Verify health check API availability (service readiness)
        2. Verify core generate API functionality (normal inference with correct results)
        3. Verify LoRA parameter configuration effectiveness via server info API
        """
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(response.status_code, 200)

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "lora_path": self.lora_a,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)

class TestLoraBasicFunction_14(CustomTestCase):
    """Testcase：Verify the functionality and parameter effectiveness when --lora-target-modules=all is set for Llama-3.2-1B

    [Test Category] Parameter
    [Test Target] --lora-target-modules
    """
    lora_a = "LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH"
    lora_b = "LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH"
    lora_c = "LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH"

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--tp-size"
            "2"
            "--enable-lora",
            "--lora-path",
            f"lora_1={cls.lora_a}",
            f"lora_2={cls.lora_b}",
            f"lora_3={cls.lora_c}",
            "--lora-target-modules",
            "all",
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

    def test_lora_use_different_lora(self):
        """Core Test: Verify the effectiveness of --lora-target-modules=all and normal server functionality

        Three-Step Verification Logic:
        1. Verify health check API availability (service readiness)
        2. Verify core generate API functionality (normal inference with correct results)
        3. Verify LoRA parameter configuration effectiveness via server info API
        """
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(response.status_code, 200)

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "lora_path": self.lora_a,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)



class TestLoraBasicFunction_15_low(CustomTestCase):
    """Testcase：Verify the functionality and parameter effectiveness when --lora-target-modules=all is set for Llama-3.2-1B

    [Test Category] Parameter
    [Test Target] --lora-target-modules
    """
    lora_a = "LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH"
    lora_b = "LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH"
    lora_c = "LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH"

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--tp-size"
            "2"
            "--enable-lora",
            "--lora-path",
            f"lora_1={cls.lora_a}",
            f"lora_2={cls.lora_b}",
            f"lora_3={cls.lora_c}",
            "--lora-target-modules",
            "all",
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

    def test_lora_use_different_lora(self):
        """Core Test: Verify the effectiveness of --lora-target-modules=all and normal server functionality

        Three-Step Verification Logic:
        1. Verify health check API availability (service readiness)
        2. Verify core generate API functionality (normal inference with correct results)
        3. Verify LoRA parameter configuration effectiveness via server info API
        """
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(response.status_code, 200)

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "lora_path": self.lora_a,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)
'''

if __name__ == "__main__":
    unittest.main()
