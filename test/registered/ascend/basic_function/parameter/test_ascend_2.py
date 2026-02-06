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


class TestDisaggregationPrefillWithHiCache(DisaggregationHiCacheBase):
    """Test disaggregation with HiCache enabled only on Prefill side"""

    @classmethod
    def start_decode(cls):
        # Decode without HiCache offload
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

    def test_prefill_cache_hit(self):
        """Test that prefill cache works with repeated queries"""

        repeated_prompt = self.gen_prompt(800)

        # First request - should miss cache
        self.send_request(repeated_prompt, max_tokens=100)

        # Flush cache
        self.trigger_offloading_and_flush()

        # Second request - should hit cache (faster)
        response2 = self.send_request(repeated_prompt, max_tokens=100)

        # Assert cached tokens cnt
        self.assertGreater(response2["meta_info"]["cached_tokens"], 700)


if __name__ == "__main__":
    unittest.main()
