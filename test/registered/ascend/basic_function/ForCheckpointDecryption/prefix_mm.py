import base64
import logging
import os
import random
import shutil
import tempfile
import time
import unittest
from typing import Dict, List

import requests

from sglang.bench_serving import get_tokenizer
from sglang.test.ascend.test_ascend_utils import QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class PrefixMMBase(PDDisaggregationServerBase):
    """Test prefix MM cache with encoder-only + language-only mode."""

    @classmethod
    def setUpClass(cls):
        super(PrefixMMBase, cls).setUpClass()

        cls.model = QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH
        cls.tokenizer = get_tokenizer(cls.model)
        cls.temp_dir = tempfile.mkdtemp()
        cls.encoder_port = "30000"
        cls.language_port = "30002"
        cls.encoder_url = f"http://{cls.base_host}:{cls.encoder_port}"
        cls.language_url = f"http://{cls.base_host}:{cls.language_port}"
        cls.start_encoder()
        cls.start_language()

        cls.wait_server_ready(cls.encoder_url + "/health")
        cls.wait_server_ready(cls.language_url + "/health")
        time.sleep(5)

    @classmethod
    def start_encoder(cls):
        encoder_args = [
            "--trust-remote-code",
            "--attention-backend", "ascend",
            "--encoder-only",
            "--encoder-transfer-backend", "zmq_to_scheduler",
            "--port", cls.encoder_port,
            "--enable-prefix-mm-cache",
            "--tp-size", "1",
            "--mem-fraction-static", "0.9",
            "--disable-cuda-graph",
            "--base-gpu-id", "12",
        ]
        env = {
            **os.environ,
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
        }
        cls.process_encoder = popen_launch_pd_server(
            cls.model,
            cls.encoder_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=encoder_args,
            env=env,
        )

    @classmethod
    def start_language(cls):
        language_args = [
            "--trust-remote-code",
            "--attention-backend", "ascend",
            "--language-only",
            "--encoder-urls", cls.encoder_url,
            "--encoder-transfer-backend", "zmq_to_scheduler",
            "--port", cls.language_port,
            "--tp-size", "1",
            "--mem-fraction-static", "0.9",
            "--disable-cuda-graph",
            "--base-gpu-id", "14",
        ]
        env = {
            **os.environ,
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
        }
        cls.process_language = popen_launch_pd_server(
            cls.model,
            cls.language_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=language_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'process_language') and cls.process_language:
            cls.process_language.terminate()
            cls.process_language.wait()
        if hasattr(cls, 'process_encoder') and cls.process_encoder:
            cls.process_encoder.terminate()
            cls.process_encoder.wait()
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def gen_prompt(self, token_num: int) -> str:
        all_available_tokens = list(self.tokenizer.get_vocab().values())
        selected_tokens = random.choices(all_available_tokens, k=token_num)
        return self.tokenizer.decode(selected_tokens)

    def send_text_request(self, prompt: str, max_tokens: int = 100) -> Dict:
        response = requests.post(
            f"{self.language_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": max_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def send_image_request(
        self, text_prompt: str, image_paths: List[str], max_tokens: int = 100
    ) -> Dict:
        image_data = [
            f"data:image/jpeg;base64,{encode_image_to_base64(img_path)}"
            for img_path in image_paths
        ]

        content = [
            {"type": "image_url", "image_url": {"url": img_base64}}
            for img_base64 in image_data
        ]
        content.append({"type": "text", "text": text_prompt})

        response = requests.post(
            f"{self.language_url}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": content}],
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": 0.0,
            },
            timeout=120,
        )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def trigger_offloading_and_flush(self):
        self.send_text_request(self.gen_prompt(1), max_tokens=150)
        time.sleep(2)
        requests.post(self.encoder_url + "/flush_cache")


class TestPrefixMMCache(PrefixMMBase):
    def test_image_request_cache_reuse(self):
        test_image_path = os.path.join(os.path.dirname(__file__), "test_image.jpg")
        if not os.path.exists(test_image_path):
            self.skipTest(f"Test image not found: {test_image_path}")

        text_prompt = "描述这张图片的内容"

        logging.warning("========== First image request (cold start) ==========")
        response1 = self.send_image_request(text_prompt, [test_image_path])
        cached_tokens_1 = response1.get("usage", {}).get("prompt_tokens", 0)
        logging.warning(f"First request cached/prompt tokens: {cached_tokens_1}")

        self.trigger_offloading_and_flush()

        logging.warning("========== Second image request (should hit cache) ==========")
        response2 = self.send_image_request(text_prompt, [test_image_path])
        cached_tokens_2 = response2.get("usage", {}).get("prompt_tokens", 0)
        logging.warning(f"Second request cached/prompt tokens: {cached_tokens_2}")

        content1 = response1.get("choices", [{}])[0].get("message", {}).get("content", "")
        content2 = response2.get("choices", [{}])[0].get("message", {}).get("content", "")

        logging.warning(f"First response: {content1[:100]}...")
        logging.warning(f"Second response: {content2[:100]}...")

        self.assertTrue(len(content1) > 0, "First request should return content")
        self.assertTrue(len(content2) > 0, "Second request should return content")
        logging.warning("Image request cache reuse test completed successfully")


if __name__ == "__main__":
    unittest.main()
