import unittest

import requests
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

RESPONSE_TEXT1 = None
RESPONSE_TEXT2 = None


class TestRandomSeedZero(CustomTestCase):
    """Testcase：Verify set --random-seed parameter, different random_seed will affect the model's output (response.text)
    and when random_seed is the same, the response.text will same.

       [Test Category] Parameter
       [Test Target] --random-seed
       """
    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    random_seed = 0

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--random-seed",
            cls.random_seed,
        ]
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_random_seed(self):
        response_text1 = None
        response_text2 = None
        for i in range(2):
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {
                        "temperature": 1,
                        "max_new_tokens": 32,
                    },
                },
            )
            self.assertEqual(response.status_code, 200)
            if i == 0:
                response_text1 = response.json()["text"]
                print("-------0000000000000000000-------------")
                print(response_text1)
            else:
                response_text2 = response.json()["text"]
                print("-------1111111111111111111-------------")
                print(response_text2)
        self.assertEqual(response_text1, response_text2)

class TestRandomSeedOne(TestRandomSeedZero):
    random_seed = 42


# class TestRandomSeed(CustomTestCase):
#     self.assertEqual(response_text1, response_text2)


if __name__ == "__main__":
    unittest.main()
