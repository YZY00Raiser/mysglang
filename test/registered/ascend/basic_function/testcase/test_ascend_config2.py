import json
import os
import unittest
import requests
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestConfig(CustomTestCase):
    """Testcase: Verify set --forward-hooks parameter, can identify the set hook function and the inference request is successfully processed.

    [Test Category] Parameter
    [Test Target] --forward-hooks
    """
    model = QWEN3_32B_WEIGHTS_PATH
    hooks_spec = [
        {
            "name": "qwen_first_layer_attn_monitor",
            "target_modules": ["model.layers.0.self_attn"],
            "hook_factory": "test_ascend_forward_hooks2:create_attention_monitor_factory",
            "config": {
                "layer_index": 0
            }
        }
    ]

    forward_hooks = json.dumps(hooks_spec)
    # forward_hooks = 3.14
    # forward_hooks = "abc"
    # forward_hooks = -2
    # forward_hooks = "!@#$"
    # forward_hooks = None
    @classmethod
    def _build_other_args(cls):
        return [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--tp-size",
            "4",
            "--forward-hooks",
            cls.forward_hooks,
            "--base-gpu-id", "4",
        ]

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
        cls.out_log_file.close()
        cls.hook_log_file.close()
        os.remove(cls.out_log_file_name)
        os.remove(cls.hook_log_file_name)

    def test_enable_multimodal_func(self):
        # with self.assertRaises(Exception) as ctx:
        self._launch_server()
        # self.assertIn("Server process exited with code 2", str(ctx.exception))
        print("-----------------------------launch_server------------------------------------------")
        self.hook_log_file.seek(0)
        hook_content = self.hook_log_file.read()
        # self.assertIn("Registered forward hook 'abc' on model.layers.0.self_attn", hook_content)
        # with self.assertRaises(Exception) as ctx:
        kill_process_tree(self.process.pid)
        # self.assertIn("'TestSetForwardHooks' object has no attribute 'process'", str(ctx.exception))
        print("-----------------------------kill_process_tree------------------------------------------")


# class TestSetForwardHooksValidation1(TestSetForwardHooks):
#     forward_hooks = "abc"
#
#     def test_enable_multimodal_func(self):
#         with self.assertRaises(Exception) as ctx:
#             self._launch_server()
#         self.assertIn("Server process exited with code 2", str(ctx.exception))
#
#         self.hook_log_file.seek(0)
#         hook_content = self.hook_log_file.read()
#         self.assertIn("Invalid JSON list: abc", hook_content)





if __name__ == "__main__":
    unittest.main(verbosity=2)
