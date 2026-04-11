import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestMoreRunnerBackendTriton(CustomTestCase):
    """Testcase：Verify set --moe-runner-backend, the inference request is successfully processed.

    [Test Category] Parameter
    [Test Target] --moe-runner-backend
    """

    model = DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
    moe_runner_backend = "triton"


    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_moe_runner_backend(cls):
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                "0.8",
                "--tp-size",
                "16",
                "--ep",
                "16",
                "--enable-eplb",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "normal",
                "--ep-num-redundant-experts",
                "16",
                "--expert-distribution-recorder-buffer-size",
                "50",
                "--moe-runner-backend",
                cls.moe_runner_backend,
            ],
            env={
                "SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT": "1",
                "HCCL_BUFFSIZE": "1024",
            },
        )


'''
class TestMoreRunnerBackendTritonDefault(TestMoreRunnerBackendTriton):
    moe_runner_backend = "auto"
'''

if __name__ == "__main__":
    unittest.main()
