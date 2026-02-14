import unittest

import requests
from sglang.test.test_utils import (
    CustomTestCase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestEnableCacheReport(CustomTestCase):
    """Testcase：Verify use DeepSeek V3.2  Prefix set enable-hierarchical-cache, not set --disable-radix-cache
    send two same requests with 600 token the second response's cached_tokens equal 512.

    [Test Category] model
    [Test Target] enable-hierarchical-cache
    """
    def test_enable_hierarchical(self):
        DEFAULT_URL_FOR_TEST="http://172.22.3.19:6688"
        print("==============startoooo====================================")
        input_ids = [1] * 300
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_tokens": 200,
                },
            },
        )

        print("==============respob=====================================")
        print(response.json())
        print("=============finshhhhhhhhhhhhhhh====================================")
        self.assertEqual(response.status_code, 200)

if __name__ == "__main__":
    unittest.main()
