import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import popen_launch_server_config
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestConfig(CustomTestCase):
    """Testcase: Verify set --config parameter, can identify the set config and inference request is successfully processed.

    [Test Category] Parameter
    [Test Target] --config
    """

    model = None
    config = "config.yaml"

    @classmethod
    def _build_other_args(cls):
        return [
            "--config",
            cls.config,
        ]

    @classmethod
    def _launch_server(cls):
        other_args = cls._build_other_args()
        cls.process = popen_launch_server_config(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_config(self):
        self._launch_server()
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


class TestConfigPriority(TestConfig):
    """Testcase: Verify set the parameter set in the command line have a higher priority than set in config.yaml,
    set false model path in in the command, set right model path in in the config.yaml,
    will use false model path service start fail .

    [Test Category] Parameter
    [Test Target] --config
    """

    model = "/data/Qwen/Qwen3-32B"

    @classmethod
    def _launch_server(cls):
        other_args = cls._build_other_args()
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(cls.out_log_file, cls.hook_log_file),
        )

    @classmethod
    def setUpClass(cls):
        cls.out_log_file_name = "./tmp_out_log.txt"
        cls.hook_log_file_name = "./tmp_hook_log.txt"
        cls.out_log_file = open(cls.out_log_file_name, "w+", encoding="utf-8")
        cls.hook_log_file = open(cls.hook_log_file_name, "w+", encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.out_log_file.close()
        cls.hook_log_file.close()
        os.remove(cls.out_log_file_name)
        os.remove(cls.hook_log_file_name)

    def test_config(self):
        with self.assertRaises(Exception) as ctx:
            self._launch_server()
        self.assertIn(
            "Server process exited with code 1. Check server logs for errors.",
            str(ctx.exception),
        )
        self.hook_log_file.seek(0)
        hook_content = self.hook_log_file.read()
        self.assertIn(
            "make sure '/data/Qwen/Qwen3-32B' is the correct path", hook_content
        )


class TestConfigValidation(TestConfig):
    """Testcase: Verify set --config exception param the service start fail.

    [Test Category] Parameter
    [Test Target] --config
    """

    test_cases = [
        "abc",
        3.14,
        -2,
        None,
        "!@#$",
        "config1.yaml",
    ]
    for config in test_cases:

        @classmethod
        def _build_other_args(cls):
            return [
                "--config",
                cls.config,
            ]

        def test_config(self):
            with self.assertRaises(Exception) as ctx:
                self._launch_server()
            self.assertIn(
                "Server process exited with code 1. Check server logs for errors.",
                str(ctx.exception),
            )


class TestConfigFileModeValidation(TestConfig):
    """Testcase: Verify set --config non yaml file format the service start fail.

    [Test Category] Parameter
    [Test Target] --config
    """

    test_cases = [
        "config.ini",
        "config.txt",
        "config.xml",
    ]
    for config in test_cases:

        @classmethod
        def _build_other_args(cls):
            return [
                "--config",
                cls.config,
            ]

        def test_config(self):
            with self.assertRaises(Exception) as ctx:
                self._launch_server()
            self.assertIn(
                "Server process exited with code 1. Check server logs for errors.",
                str(ctx.exception),
            )


class TestConfigParamValidation(TestConfig):
    """Testcase: Verify set exception param in config file the service start fail.

    [Test Category] Parameter
    [Test Target] --config
    """

    config = "config_valid.yaml"

    def test_config(self):
        with self.assertRaises(Exception) as ctx:
            self._launch_server()
        self.assertIn(
            "Server process exited with code 2. Check server logs for errors.",
            str(ctx.exception),
        )


if __name__ == "__main__":
    unittest.main()
