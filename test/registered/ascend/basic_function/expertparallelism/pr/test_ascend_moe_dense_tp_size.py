import os
import unittest
from types import SimpleNamespace
from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_CODER_V2_LITE_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestAscendMoeDenseTPSize(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the model accuracy remains uncompromised when the parameter --moe-dense-tp-size is configured to 1.

    [Test Category] Parameter
    [Test Target] --moe-dense-tp-size
    """

    model = DEEPSEEK_CODER_V2_LITE_WEIGHTS_PATH
    accuracy = 0.9
    other_args = [
        "--trust-remote-code",
        "--tp",
        "2",
        "--moe-dense-tp-size",
        "1",
        "--moe-a2a-backend",
        "deepep",
        "--max-running-requests",
        "512",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--mem-fraction-static",
        "0.85",
    ],
    env = {
        "SGLANG_NPUDISABLE_ACL_FORMAT_WEIGHT": "1",
        "HCCL_BUFFSIZE": "1024",
    },


if __name__ == "__main__":
    unittest.main()
