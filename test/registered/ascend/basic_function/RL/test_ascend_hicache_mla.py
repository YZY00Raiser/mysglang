import unittest

from types import SimpleNamespace
from sglang.test.run_eval import run_eval

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH = "/root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V2-Lite-W8A8"


class TestRL(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.other_args = [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--tp-size",
            2,
            "--mem-fraction-static",
            0.8,
            "--enable-hierarchical-cache",
            "--hicache-ratio",
            1.2,
            "--base-gpu-id",
            "12",
        ]
        cls.process = popen_launch_server(
            QWEN3_32B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_l1_cache_reuse(self):
        args = SimpleNamespace(
            max_new_tokens=512,
            base_url=DEFAULT_URL_FOR_TEST,
            model=QWEN3_32B_WEIGHTS_PATH,
            eval_name="gsm8k",
            api="completion",
            num_examples=200,
            num_threads=128,
            num_shots=5,
        )
        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.86)


if __name__ == "__main__":
    unittest.main()
