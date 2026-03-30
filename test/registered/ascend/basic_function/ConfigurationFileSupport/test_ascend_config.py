import unittest

import requests

from sglang.srt.utils import kill_process_tree
# from sglang.test.ascend.test_ascend_utils import (
#     CONFIG_YAML_PATH,
#     popen_launch_server_config,
# )
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_32B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=400,
    suite="nightly-4-npu-a3",
    nightly=True,
)
CONFIG_YAML_PATH = (
    "/data/y30082119/mysglang/test/registered/ascend/basic_function/ConfigurationFileSupport/config.yaml"
)
'''
class TestConfig(CustomTestCase):
    """Testcase: Verify set --config parameter, can identify the set config and inference request is successfully processed.

    [Test Category] Parameter
    [Test Target] --config
    """

    model = None
    config = CONFIG_YAML_PATH

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


'''


class TestConfigPriority(CustomTestCase):
    """Testcase: Verify set the parameter set in the command line have a higher priority than set in config.yaml,
    set false model path in the command, set right model path in the config.yaml,
    will use false model path service start fail .

    [Test Category] Parameter
    [Test Target] --config
    """

    # model = "/data/Qwen/Qwen3-32B"
    model = QWEN3_32B_WEIGHTS_PATH
    config = CONFIG_YAML_PATH

    @classmethod
    def _launch_server(cls):
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--config", cls.config, ],
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


if __name__ == "__main__":
    unittest.main()
