import os
import subprocess
import unittest

import requests

from sglang.srt.utils import kill_process_tree
# from sglang.test.ascend.test_ascend_utils import popen_launch_server
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server, _create_clean_subprocess_env, _wait_for_server_health,
)

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)

MODEL_PATH = "/home/weights/Qwen3-0.6B"
PATH = "/home/l30081563/prtest/sglang/test/registered/ascend/basic_function/parameter"
CONFIG_YAML_PATH = f"{PATH}/config.yaml"
CONFIG_INVALID_YAML_PATH = f"{PATH}/config_invalid.yaml"


class TestAscendConfig(CustomTestCase):
    """Testcase: Verify set --config parameter, can identify the set config and inference request is successfully processed.

    [Test Category] Parameter
    [Test Target] --config
    """

    config = CONFIG_YAML_PATH

    @classmethod
    def _launch_server_with_config_yaml(cls, config_file, url, timeout):
        command = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--config",
            config_file,
        ]
        process = subprocess.Popen(command, stdout=None, stderr=None,
                                   env=_create_clean_subprocess_env(os.environ.copy()))
        _wait_for_server_health(process, url, None, timeout)
        return process

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

        # TODO：或许应该在这里生成config文件

        cls.process = cls._launch_server_with_config_yaml(cls.config, cls.base_url, DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH)
        # command = [
        #     "python3",
        #     "-m",
        #     "sglang.launch_server",
        #     "--config",
        #     cls.config,
        # ]
        # cls.process = subprocess.Popen(command, stdout=None, stderr=None,
        #                            env=_create_clean_subprocess_env(os.environ.copy()))
        # _wait_for_server_health(cls.process, cls.base_url, None, DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_config(self):
        response = requests.post(
            f"{self.base_url}/generate",
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


class TestAscendConfigInValidConfigFileType(CustomTestCase):
    """Testcase: Verify set --config non yaml file format the service start fail.

    [Test Category] Parameter
    [Test Target] --config
    """

    invalid_config_file_list = [
        "config.ini",
        "config.txt",
        "config.xml",
    ]

    # for config in test_cases:

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def tearDownClass(cls):
        pass

    def test_config(self):
        process = None
        for config in self.invalid_config_file_list:
            try:
                # with self.assertRaises(Exception) as ctx:

                self.other_args = [
                    "--config",
                    config,
                ]

                process = popen_launch_server(
                    self.model,
                    self.base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=self.other_args,
                )
            except Exception as e:
                self.assertIn(
                    "Server process exited with code 1. Check server logs for errors.",
                    str(e),
                )
                print(e)
            finally:
                if process:
                    kill_process_tree(process.pid)


# class TestConfigParamValidation(TestConfig):
#     """Testcase: Verify set exception param in config file the service start fail.
#
#     [Test Category] Parameter
#     [Test Target] --config
#     """
#
#     config = CONFIG_VALID_YAML_PATH
#
#     def test_config(self):
#         with self.assertRaises(Exception) as ctx:
#             self._launch_server()
#         self.assertIn(
#             "Server process exited with code 2. Check server logs for errors.",
#             str(ctx.exception),
#         )


if __name__ == "__main__":
    # unittest.main()
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestAscendConfig))
    runner = unittest.TextTestRunner()
    runner.run(suite)
