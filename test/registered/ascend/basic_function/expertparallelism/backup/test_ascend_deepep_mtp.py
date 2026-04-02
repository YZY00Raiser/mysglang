import os
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_R1_0528_W8A8_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestAscendDeepEP(CustomTestCase):
    """
    Testcase：Verify the correctness and performance of DeepSeek Model when the MTP technology and deepep are used

    [Test Category] Parameter
    [Test Target] use MTP by test model DeepSeek R1, --scheduler-recv-interval 10, --moe-a2a-backend deepep,
    --deepep-mode auto
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_R1_0528_W8A8_WEIGHTS_PATH
        cls.accuracy = 0.95

        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(cls.base_url)

        cls.common_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--quantization",
            "modelslim",
            "--mem-fraction-static",
            0.8,
            "--max-running-requests",
            32,
            "--disable-radix-cache",
            "--chunked-prefill-size",
            32768,
            "--disable-cuda-graph",
            "--tp-size",
            16,
            "--dp-size",
            1,
            "--ep-size",
            16,
            "--moe-a2a-backend",
            "deepep",
            "--deepep-mode",
            "auto",
            "--speculative-algorithm",
            "NEXTN",
            "--speculative-num-steps",
            1,
            "--speculative-eagle-topk",
            1,
            "--speculative-num-draft-tokens",
            2,
            "--scheduler-recv-interval",
            10,
        ]

        cls.extra_envs = {
            "HCCL_BUFFSIZE": "1024",
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
            "SGLANG_NPU_USE_MLAPO": "1",
            "SGLANG_NPU_USE_EINSUM_MM": "1",
            "SLANG_ENABLE_SPEC_V2": "1",
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
        }
        os.environ.update(cls.extra_envs)

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=1500,
            other_args=[
                *cls.common_args,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(self):
        print(f"##=== Testing accuracy: {model} ===##")

        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host=f"http://{self.url.hostname}",
            port=int(self.url.port),
        )

        metrics = run_eval_few_shot_gsm8k(args)
        self.assertGreaterEqual(
            metrics["accuracy"],
            self.accuracy,
        )


if __name__ == "__main__":
    unittest.main()
