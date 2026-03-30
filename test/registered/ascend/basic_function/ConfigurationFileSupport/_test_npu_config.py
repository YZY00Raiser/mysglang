import os
import subprocess
import unittest

import requests

from sglang.srt.utils import kill_process_tree
# from sglang.test.ascend.test_ascend_utils import (
#     CONFIG_YAML_PATH,
# )
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server, _wait_for_server_health, _create_clean_subprocess_env,
)

register_npu_ci(
    est_time=400,
    suite="nightly-4-npu-a3",
    nightly=True,
)

CONFIG_YAML_PATH = (
    "/data/y30082119/mysglang/test/registered/ascend/basic_function/ConfigurationFileSupport/config.yaml"
)
class TestConfig(CustomTestCase):
    """Testcase: Verify set --config parameter, can identify the set config and inference request is successfully processed.

    [Test Category] Parameter
    [Test Target] --config
    """

    config = CONFIG_YAML_PATH

    @classmethod
    def launch_server_with_config_yaml(cls, config_file, url, timeout):
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
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = cls.launch_server_with_config_yaml(cls.config, cls.base_url, DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH)

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

'''
class TestConfigPriority(TestConfig):
    """Testcase: Verify set the parameter set in the command line have a higher priority than set in config.yaml,
    set false model path in the command, set right model path in the config.yaml,
    will use false model path service start fail .

    [Test Category] Parameter
    [Test Target] --config
    """

    model = "/nonexistent/Qwen/Qwen3-32B"

    @classmethod
    def _launch_server(cls):
        other_args = cls._build_other_args()
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_config(self):
        with self.assertRaises(Exception) as ctx:
            self._launch_server()
        self.assertIn(
            "Server process exited with code 1. Check server logs for errors.",
            str(ctx.exception),
        )

'''


if __name__ == "__main__":
    unittest.main()
