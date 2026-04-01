import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH = "/root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V2-Lite-W8A8"


class TestAscendMlaHicache(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            0.8,
            "--attention-backend",
            "ascend",
            "--tp-size",
            4,
            "--enable-hierarchical-cache",
            "--hicache-ratio",
            1.2,
        ]

        cls.process = popen_launch_server(
            DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_l1_cache_reuse(self):
        input_ids_first = [1] * 200
        input_ids_second = input_ids_first + [2] * 70

        def make_request(input_ids, expected_cached_tokens):
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "input_ids": input_ids,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 32,
                    },
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["meta_info"]["cached_tokens"], expected_cached_tokens)

        # For the first request, using lora_a, the expected cache size is 0.
        make_request(input_ids_first, 0)

        # The second request uses lora_b, expecting a cache of 0 (different lora types do not share cache).
        make_request(input_ids_first, 128)


if __name__ == "__main__":
    unittest.main()
