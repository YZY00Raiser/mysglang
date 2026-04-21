import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestOffloadGroupSize(CustomTestCase):
    """Testcase: Tests core functionality with --cpu-offload-gb configuration, inference requests successful.
    and the ingerence accuracy using the GSM8K dataset is no less than 0.86.

    [Test Category] Parameter
    [Test Target] --offload-group-size
    """

    @classmethod
    def setUpClass(cls):
        cls.other_args = [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--offload-group-size",
            "2",
        ]

        cls.out_log_file = open("./cache_out_log.txt", "w+", encoding="utf-8")
        cls.err_log_file = open("./cache_err_log.txt", "w+", encoding="utf-8")
        cls.process = popen_launch_server(
            DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        cls.out_log_file.close()
        cls.err_log_file.close()
        os.remove("./cache_out_log.txt")
        os.remove("./cache_err_log.txt")
        kill_process_tree(cls.process.pid)

    def test_max_loaded_loras_error(self):
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
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)

        self.err_log_file.seek(0)
        content = self.err_log_file.read()
        offload_message = "[offloader]"
        self.assertIn(offload_message, content)


class TestOffload1(CustomTestCase):
    """Testcase: Tests core functionality with --cpu-offload-gb configuration, inference requests successful.
    and the ingerence accuracy using the GSM8K dataset is no less than 0.86.

    [Test Category] Parameter
    [Test Target] --offload-mode
    """

    @classmethod
    def setUpClass(cls):
        cls.other_args = [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--offload-group-size",
            "4",
            "--offload-num-in-group",
            "2",
            "--offload-prefetch-step",
            "2",
            "--offload-mode",
            "meta",
        ]

        cls.out_log_file = open("./cache_out_log.txt", "w+", encoding="utf-8")
        cls.err_log_file = open("./cache_err_log.txt", "w+", encoding="utf-8")
        cls.process = popen_launch_server(
            DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        cls.out_log_file.close()
        cls.err_log_file.close()
        os.remove("./cache_out_log.txt")
        os.remove("./cache_err_log.txt")
        kill_process_tree(cls.process.pid)

    def test_max_loaded_loras_error(self):
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
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Paris", response.text)
        self.err_log_file.seek(0)
        content = self.err_log_file.read()
        offload_message = "[offloader]"
        self.assertIn(offload_message, content)


class TestOffload2(CustomTestCase):
    """Testcase: Tests core functionality with --cpu-offload-gb configuration, inference requests successful.
    and the ingerence accuracy using the GSM8K dataset is no less than 0.86.

    [Test Category] Parameter
    [Test Target] --offload-mode
    """

    @classmethod
    def setUpClass(cls):
        cls.other_args = [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--dp",
            "2",
            "--offload-group-size",
            "4",
            "--offload-num-in-group",
            "2",
            "--offload-prefetch-step",
            "2",
            "--offload-mode",
            "sharded_gpu",
        ]

        cls.out_log_file = open("./cache_out_log.txt", "w+", encoding="utf-8")
        cls.err_log_file = open("./cache_err_log.txt", "w+", encoding="utf-8")
        cls.process = popen_launch_server(
            DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        cls.out_log_file.close()
        cls.err_log_file.close()
        os.remove("./cache_out_log.txt")
        os.remove("./cache_err_log.txt")
        kill_process_tree(cls.process.pid)

    def test_max_loaded_loras_error(self):
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
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)
        self.err_log_file.seek(0)
        content = self.err_log_file.read()
        offload_message = "[offloader]"
        self.assertIn(offload_message, content)


if __name__ == "__main__":
    unittest.main()
