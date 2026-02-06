import os
import random
import tempfile
import time
import unittest
from typing import Dict

import requests

from sglang.bench_serving import get_tokenizer
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)


class TestDisaggregationDecodeWithHiCache(DisaggregationHiCacheBase):
    """Test disaggregation with HiCache enabled on both Prefill and Decode sides"""

    @classmethod
    def start_decode(cls):
        # Decode with HiCache offload enabled
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--tp-size",
            "1",
            "--page-size",
            "64",
            "--mem-fraction-static",
            "0.8",
            "--base-gpu-id",
            "1",
            "--disaggregation-decode-enable-offload-kvcache",
            "--hicache-ratio",
            "1.2",
            "--hicache-size",
            "0",
            "--hicache-storage-backend",
            "file",
            "--hicache-storage-prefetch-policy",
            "wait_complete",
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        env = {
            **os.environ,
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
        }
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=env,
        )

    def test_multi_turn_conversation_cache(self):
        """Test multi-turn conversation scenario with cache hit improvement"""

        print("=== Multi-turn Conversation Cache Test ===")

        # Turn 1
        initial_prompt = self.gen_prompt(300)

        response1 = self.send_request(initial_prompt, max_tokens=200, temperature=0.1)
        current_context = initial_prompt + response1["text"]

        # Turns 2-4: Continue generation based on previous context
        previous_cached_tokens = 0

        for turn in range(2, 5):
            print(f"\nTurn {turn}: Continuing from previous context")

            response = self.send_request(
                current_context, max_tokens=200, temperature=0.1
            )
            cached_tokens = response["meta_info"]["cached_tokens"]

            print(f"Turn {turn} cached tokens: {cached_tokens}")
            print(f"Improvement: {cached_tokens - previous_cached_tokens} tokens")

            # Assert cache improvement
            self.assertGreater(
                cached_tokens,
                previous_cached_tokens,
                f"Turn {turn} should have more cached tokens than turn {turn - 1}",
            )

            # Update context and cached tokens for next iteration
            current_context += response["text"]
            previous_cached_tokens = cached_tokens

            # Flush prefill cache
            self.trigger_offloading_and_flush()


if __name__ == "__main__":
    unittest.main()
