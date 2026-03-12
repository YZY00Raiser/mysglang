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

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)

LLAMA_3_2_1B_WEIGHTS_PATH = "/home/weights/LLM-Research/Llama-3.2-1B-Instruct"


class TestLoraBasicFunction(CustomTestCase):
    """Testcase：Verify the lora functionality

    [Test Category] Parameter
    [Test Target] --enable-lora, --lora-path
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
            "--lora-target-modules",
            "all",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--base-gpu-id",
            "6",
            "--max-loras-per-batch",
            "3",
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
        # case1 case2
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
        self.assertEqual(response.status_code, 200)
        text_no_lora = response.text

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
        text_lora_a = response.text

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
        text_lora_b = response.text

        print("--------------------------response.json()-----------lora_b-------------------------------")
        print(response.json())

        self.assertNotEqual(
            text_no_lora,
            text_lora_a,
            f"same response.text"
        )

        self.assertNotEqual(
            text_no_lora,
            text_lora_b,
            f"same response.text"
        )

        self.assertNotEqual(
            text_lora_a,
            text_lora_b,
            f"same response.text"
        )

    '''
    def test_openai_with_different_loras(self):
        #case 11
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/chat/completions",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "lora_path": "lora_a",
            },
        )
        text_lora_a = response.text


    def test_batch_with_different_loras(self):
        # case3
        prompts = [
            "What is AI",
            "Explain neural network",
            "What is deep learning",
        ]
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": prompts,
                "sampling_params": {
                    "temperature": 0.7,
                    "max_new_tokens": 64,
                },
                "lora_path": ["lora_a", "lora_b"],
            },
        )
        results = response.json()

        self.assertEqual(len(results), len(prompts))

        for i, result in enumerate(results):
            self.assertEqual("text", result)
            self.assertGreater(len(result["text"]), 0)
            print(f"Prompt {i + 1} result: {result['text'][:50]}...")

        # print("--------------------------response.json()----------non--lora------------------------------")
        # print(response.json())
        # self.assertEqual(response.status_code, 200)

    # # 对比流式，非流式结果一致性
    # response_stream = requests.post(
    #     f"{DEFAULT_URL_FOR_TEST}/generate",
    #     json={
    #         "text": "The capital of France is",
    #         "sampling_params": {
    #             "temperature": 0,
    #             "max_new_tokens": 32,
    #         },
    #         "lora_path": "lora_a",
    #         "stream": True,
    #     },
    #     stream=True,
    # )
    # stream_text = ""
    # for chunk in response_stream.iter_lines(decode_unicode=False):
    #     chunk = chunk.decode("utf-8")
    #     if chunk and chunk.startswith("data:"):
    #         if chunk == "data: [DONE]":
    #             break
    #         data = json.loads(chunk[5:].strip("\n"))
    #         stream_text += data.get("text", "")
    # print("--------------------------chunk-------stream--true---------------------------------")
    # print(stream_text)




    def test_lora_with_temperature(self):
    # case4
    response_texts = []
    for i in range(2):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0.8,
                    "max_new_tokens": 32,
                },
                "lora_path": "lora_a",
            },
        )
        self.assertEqual(response.status_code, 200)
        response_text = response.json()["text"]
        response_texts.append(response_text)
    first_text = response_texts[0]
    for idx, text in enumerate(response_texts[1:], start=2):
        self.assertNotEqual(text, first_text, f"same response_text")

    def test_lora_with_json_schema(self):
        #case5
        json_schema = json.dumps({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "city": {"type": "string"},
            },
            "required": ["name", "age", "city"],

        })

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "Generate person information",
                "sampling_params": {
                    "temperature": 0.3,
                    "max_new_tokens": 128,
                    "json_schema": json_schema,
                },
                "lora_path": "lora_a",
            },
        )
        print("--------------------------response.json()----------lora_a--------------------------------")
        print(response.json())
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("text", result)
        parsed_json = json.loads(result["text"])
        self.assertIn("name", parsed_json)
        self.assertIn("age", parsed_json)
        self.assertIn("city", parsed_json)
        print(f"Valid JSON generate: {parsed_json}")
'''

'''
class TestLoraKVCache(CustomTestCase):
    """Testcase：Verify the LoRA adapter can work properly with Radix Cache

    [Test Category] Parameter
    [Test Target] --enable-lora
    """
    lora_a = "/home/weights/codelion/Llama-3.2-1B-Instruct-tool-calling-lora"
    lora_b = "/home/weights/codelion/FastLlama-3.2-LoRA"
    lora_c = "/home/weights/codelion/FastLlama-3.2-LoRA"
    lora_d = "/home/weights/codelion/Llama-3.2-1B-Instruct-tool-calling-lora"

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--tp-size"
            "2"
            "--enable-lora",
            "--lora-path",
            f"lora_1={cls.lora_a}",
            f"lora_2={cls.lora_b}",
            "--enable-radix-cache",
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

    def test_lora(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "lora_path": self.lora_a,
            },
        )
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
                "lora_path": self.lora_b,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)



class TestLoraMemoryEviction(CustomTestCase):
    """Testcase：Verify the functionality and parameter effectiveness when --lora-target-modules=all is set for Llama-3.2-1B

    [Test Category] Parameter
    [Test Target] --lora-target-modules
    """
    lora_a = "/home/weights/codelion/Llama-3.2-1B-Instruct-tool-calling-lora"
    lora_b = "/home/weights/codelion/FastLlama-3.2-LoRA"
    lora_c = "/home/weights/codelion/FastLlama-3.2-LoRA"
    lora_d = "/home/weights/codelion/Llama-3.2-1B-Instruct-tool-calling-lora"

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--tp-size"
            "2"
            "--enable-lora",
            "--lora-path",
            f"lora_1={cls.lora_a}",
            f"lora_2={cls.lora_b}",
            "--max-load-loras",
            "3",
            "--lora-eviction-policy"
            "fifo"
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

    def test_lora(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "lora_path": self.lora_c,
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


'''


if __name__ == "__main__":
    unittest.main()
