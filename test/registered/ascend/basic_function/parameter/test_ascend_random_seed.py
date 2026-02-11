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


class TestRandomSeed(CustomTestCase):
    """Testcase：Verify set --random-seed parameter, the inference request is successfully processed.

       [Test Category] Parameter
       [Test Target] --random-seed
       """
    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    random_seed = 0

    @classmethod
    def get_server_args(cls):
        """Return the arguments for the server launch. Override in subclasses."""
        other_args = (
            [
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--random-seed",
                cls.random_seed,
            ]

        )
        return other_args

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.get_server_args(),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_random_seed(self):
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
        self.assertEqual(
            response.status_code, 200, "The request status code is not 200."
        )
        print("------------------respo1111111111111nseeeeee---------------------------------")
        print(response.json())
        self.assertIn(
            "Paris", response.text, "The inference result does not include Paris."
        )

class TestRandomSeedOne(CustomTestCase):
    """Testcase：Verify set --random-seed parameter, the inference request is successfully processed.

           [Test Category] Parameter
           [Test Target] --random-seed
           """
    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    random_seed = 0

    @classmethod
    def get_server_args(cls):
        """Return the arguments for the server launch. Override in subclasses."""
        other_args = (
            [
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--random-seed",
                cls.random_seed,
            ]

        )
        return other_args

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.get_server_args(),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_random_seed(self):
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
        self.assertEqual(
            response.status_code, 200, "The request status code is not 200."
        )
        print("------------------respo1111111111111nseeeeee---------------------------------")
        print(response.json())
        self.assertIn(
            "Paris", response.text, "The inference result does not include Paris."
        )


if __name__ == "__main__":
    unittest.main()
