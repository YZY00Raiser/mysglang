import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
# from sglang.test.ascend.test_ascend_utils import DEEPSEEK_CODER_V2_LITE_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)
DEEPSEEK_CODER_V2_LITE_WEIGHTS_PATH = "/mnt/nfs_share/weights/DeepSeek-Coder-V2-Lite-Instruct"


class TestEPLBDispatchAlgorithmStatic(CustomTestCase):
    """Testcase: Verify that the model accuracy remains uncompromised when the parameter --moe-dense-tp-size is configured to 1.

    [Test Category] Parameter
    [Test Target] --ep-dispatch-algorithm
    """

    ep_dispatch_algorithm = "static"

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            DEEPSEEK_CODER_V2_LITE_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                "0.5",
                "--tp-size",
                "2",
                "--expert-parallel-size",
                "2",
                "--enable-eplb",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "normal",
                "--ep-num-redundant-experts",
                "4",
                "--ep-dispatch-algorithm",
                cls.ep_dispatch_algorithm,
            ],
            env={
                "SGLANG_NPUDISABLE_ACL_FORMAT_WEIGHT": "1",
                "HCCL_BUFFSIZE": "1024",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path="/home/y30082119/test.jsonl",
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            base_url=DEFAULT_URL_FOR_TEST,
            eval_name="gsm8k",
            api="completion",
        )
        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.79)


class TestEPLBDispatchAlgorithmDynamic(TestEPLBDispatchAlgorithmStatic):
    ep_dispatch_algorithm = "dynamic"


class TestEPLBDispatchAlgorithmFake(TestEPLBDispatchAlgorithmStatic):
    ep_dispatch_algorithm = "fake"


if __name__ == "__main__":
    unittest.main()
