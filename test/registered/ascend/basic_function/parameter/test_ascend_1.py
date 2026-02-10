import unittest
import requests
import os
import glob
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
)
DEFAULT_URL_FOR_TEST="http://127.0.0.1:6666"
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

class DisaggregationHiCacheBase(CustomTestCase):
    def test_prefill_cache_hit(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
            timeout=30
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)

if __name__ == "__main__":
    unittest.main()
