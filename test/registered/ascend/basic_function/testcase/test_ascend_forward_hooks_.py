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

import logging
import time


def create_attention_monitor_factory(config):
    """
    hook factory
    config: from --forward hooks
    """
    layer_index = config.get("layer_index", 0)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",  # 添加时间戳和日志级别
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    def attention_monitor_hook(module, inputs, output):
        """
        The actual hook function is called during the forward propagation of the self-attention layer.
        """
        timestamp = time.time()

        hidden_states = inputs[1] if inputs else None

        monitor_record = {
            "timestamp": timestamp,
            "layer_index": layer_index,
            "module_type": type(module).__name__,
            "inputs": hidden_states.sum(-1)[:5] if hidden_states is not None else None,
            "outputs": output.sum(-1)[:5],
        }

        logging.info(f"hook effect: {monitor_record}")

        return output

    return attention_monitor_hook


class TestSetForwardHooks(CustomTestCase):
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
        kill_process_tree(cls.process.pid)
        cls.out_log_file.close()
        cls.hook_log_file.close()
        os.remove(cls.out_log_file_name)
        os.remove(cls.hook_log_file_name)

    def test_enable_multimodal_func(self):
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

        self.hook_log_file.seek(0)
        hook_content = self.hook_log_file.read()
        self.assertIn("hook effect", hook_content)


# --forward-hooks参数
'''
class TestSetForwardHooksValidation1(TestSetForwardHooks):

    forward_hooks = "abc"

    def test_enable_multimodal_func(self):
        with self.assertRaises(Exception) as ctx:
            self._launch_server()
        self.assertIn("Server process exited with code 2", str(ctx.exception))
        self.hook_log_file.seek(0)
        hook_content = self.hook_log_file.read()
        self.assertIn("Invalid JSON list: abc", hook_content)


class TestSetForwardHooksValidation2(TestSetForwardHooks):
    forward_hooks = 3.14

    def test_enable_multimodal_func(self):
        with self.assertRaises(Exception) as ctx:
            self._launch_server()
        self.assertIn("Server process exited with code -9", str(ctx.exception))
        self.hook_log_file.seek(0)
        hook_content = self.hook_log_file.read()
        self.assertIn("'float' object is not iterable", hook_content)


class TestSetForwardHooksValidation3(TestSetForwardHooks):
    forward_hooks = -2

    def test_enable_multimodal_func(self):
        with self.assertRaises(Exception) as ctx:
            self._launch_server()
        self.assertIn("Server process exited with code -9", str(ctx.exception))
        self.hook_log_file.seek(0)
        hook_content = self.hook_log_file.read()
        self.assertIn("'int' object is not iterable", hook_content)


class TestSetForwardHooksValidation4(TestSetForwardHooks):
    forward_hooks = "!@#$"

    def test_enable_multimodal_func(self):
        with self.assertRaises(Exception) as ctx:
            self._launch_server()
        self.assertIn("Server process exited with code 2", str(ctx.exception))
        self.hook_log_file.seek(0)
        hook_content = self.hook_log_file.read()
        self.assertIn("Invalid JSON list: !@#$", hook_content)

class TestSetForwardHooksValidation5(TestSetForwardHooks):
    forward_hooks = None

    def test_enable_multimodal_func(self):
        with self.assertRaises(Exception) as ctx:
            self._launch_server()
        self.assertIn("Server process exited with code 2", str(ctx.exception))
        self.hook_log_file.seek(0)
        hook_content = self.hook_log_file.read()
        self.assertIn("Invalid JSON list: None", hook_content)
'''

# --forward-hooks参数字段名称
"""
class TestSetForwardHooksFieldNameValidation(TestSetForwardHooks):
    hooks_spec = [
        {
            "NAME": "qwen_first_layer_attn_monitor",
            "target_modules": ["model.layers.0.self_attn"],
            "hook_factory": "test_ascend_forward_hooks2:create_attention_monitor_factory",
            "config": {
                "layer_index": 0
            }
        }
    ]

    def test_enable_multimodal_func(self):
        self._launch_server()
        self.hook_log_file.seek(0)
        hook_content = self.hook_log_file.read()
        self.assertIn("Registered forward hook '' on model.layers.0.self_attn", hook_content)



class TestSetForwardHooksFieldNameValidation(TestSetForwardHooks):
    hooks_spec = [
        {
            "name": "qwen_first_layer_attn_monitor",
            "TARGET_modules": ["model.layers.0.self_attn"],
            "hook_factory": "test_ascend_forward_hooks2:create_attention_monitor_factory",
            "config": {
                "layer_index": 0
            }
        }
    ]

    def test_enable_multimodal_func(self):
        self._launch_server()
        self.hook_log_file.seek(0)
        hook_content = self.hook_log_file.read()
        self.assertIn("has no 'target_modules', skipping", hook_content)



class TestSetForwardHooksFieldNameValidation(TestSetForwardHooks):
    hooks_spec = [
        {
            "name": "qwen_first_layer_attn_monitor",
            "target_modules": ["model.layers.0.self_attn"],
            "hook_Factory": "test_ascend_forward_hooks2:create_attention_monitor_factory",
            "config": {
                "layer_index": 0
            }
        }
    ]

    def test_enable_multimodal_func(self):
        self._launch_server()
        self.hook_log_file.seek(0)
        hook_content = self.hook_log_file.read()
        self.assertIn("has no 'hook_factory', skipping", hook_content)


class TestSetForwardHooksFieldNameValidation(TestSetForwardHooks):
    hooks_spec = [
        {
            "name": "qwen_first_layer_attn_monitor",
            "target_modules": ["model.layers.0.self_attn"],
            "hook_factory": "test_ascend_forward_hooks2:create_attention_monitor_factory",
            "Config": {
                "layer_index": 0
            }
        }
    ]
"""

# --forward-hooks参数字段 name target_modules hook_factory config 异常参数
'''
class TestSetForwardHooksFieldValidation1(TestSetForwardHooks):

    hooks_spec = [
        {
            "name": "abc",
            "target_modules": ["model.layers.0.self_attn"],
            "hook_factory": "test_ascend_forward_hooks2:create_attention_monitor_factory",
            "config": {
                "layer_index": 0
            }
        }
    ]

    def test_enable_multimodal_func(self):
        self._launch_server()
        self.hook_log_file.seek(0)
        hook_content = self.hook_log_file.read()
        self.assertIn("Registered forward hook 'abc' on model.layers.0.self_attn", hook_content)

class TestSetForwardHooksFieldValidation2(TestSetForwardHooks):
    hooks_spec = [
        {
            "name": 3.14,
            "target_modules": ["model.layers.0.self_attn"],
            "hook_factory": "test_ascend_forward_hooks2:create_attention_monitor_factory",
            "config": {
                "layer_index": 0
            }
        }
    ]

    def test_enable_multimodal_func(self):
        self._launch_server()
        self.hook_log_file.seek(0)
        hook_content = self.hook_log_file.read()
        self.assertIn("Registered forward hook '3.14' on model.layers.0.self_attn", hook_content)

class TestSetForwardHooksFieldValidation3(TestSetForwardHooks):
   hooks_spec = [
        {
            "name": -2,
            "target_modules": ["model.layers.0.self_attn"],
            "hook_factory": "test_ascend_forward_hooks2:create_attention_monitor_factory",
            "config": {
                "layer_index": 0
            }
        }
    ]

    def test_enable_multimodal_func(self):
        self._launch_server()
        self.hook_log_file.seek(0)
        hook_content = self.hook_log_file.read()
        self.assertIn("Registered forward hook '-2' on model.layers.0.self_attn", hook_content)

'''

'''
class TestSetForwardHooksFieldValidation4(TestSetForwardHooks):
    hooks_spec = [
        {
            "name": None,
            "target_modules": ["model.layers.0.self_attn"],
            "hook_factory": "test_ascend_forward_hooks2:create_attention_monitor_factory",
            "config": {
                "layer_index": 0
            }
        }
    ]

    def test_enable_multimodal_func(self):
        with self.assertRaises(Exception) as ctx:
            self._launch_server()
        self.assertIn("Server process exited with code 2", str(ctx.exception))
        self.hook_log_file.seek(0)
        hook_content = self.hook_log_file.read()
        self.assertIn("Invalid JSON list: None", hook_content)



class TestSetForwardHooksFieldValidation5(TestSetForwardHooks):
    hooks_spec = [
        {
            "name": !@#$,
            "target_modules": ["model.layers.0.self_attn"],
            "hook_factory": "test_ascend_forward_hooks2:create_attention_monitor_factory",
            "config": {
                "layer_index": 0
            }
        }
    ]

    def test_enable_multimodal_func(self):
        with self.assertRaises(Exception) as ctx:
            self._launch_server()
        self.assertIn("Server process exited with code 2", str(ctx.exception))
        self.hook_log_file.seek(0)
        hook_content = self.hook_log_file.read()
        self.assertIn("Invalid JSON list: None", hook_content)
'''

# --forward-hooks参数字段 name target_modules hook_factory config
import unittest


class TestSetForwardHooksFieldNameValidation(TestSetForwardHooks):
    # 定义所有测试用例：(name值, 期望的断言字符串)
    test_cases = [
        ("abc", "Registered forward hook 'abc' on model.layers.0.self_attn"),
        (3.14, "Registered forward hook '3.14' on model.layers.0.self_attn"),
        (-2, "Registered forward hook '-2' on model.layers.0.self_attn"),
        (None, "Registered forward hook 'None' on model.layers.0.self_attn"),
        ("!@#$", "Registered forward hook 'None' on model.layers.0.self_attn"),
    ]

    def test_enable_multimodal_func(self):
        # 遍历所有测试用例，逐个执行测试
        for name, expected_log in self.test_cases:
            # 为每个用例动态构建 hooks_spec
            self.hooks_spec = [
                {
                    "name": name,
                    "target_modules": ["model.layers.0.self_attn"],
                    "hook_factory": "test_ascend_forward_hooks2:create_attention_monitor_factory",
                    "config": {
                        "layer_index": 0
                    }
                }
            ]

            # 执行核心测试逻辑
            self._launch_server()
            self.hook_log_file.seek(0)
            hook_content = self.hook_log_file.read()

            # 断言，若失败则明确提示是哪个 name 值出了问题
            self.assertIn(
                expected_log,
                hook_content,
                msg=f"测试 name={name} 失败：未找到期望的日志内容"
            )


'''
class TestSetForwardHooksFieldTargetModulesValidation(TestSetForwardHooks):
    # 定义所有测试用例：(target_modules值, 期望的断言字符串)
    test_cases = [
        ("abc", "Registered forward hook 'abc' on model.layers.0.self_attn"),
        (3.14, "Registered forward hook '3.14' on model.layers.0.self_attn"),
        (-2, "Registered forward hook '-2' on model.layers.0.self_attn"),
        (None, "Registered forward hook 'None' on model.layers.0.self_attn"),
        ("!@#$", "Registered forward hook '!@#$' on model.layers.0.self_attn"),
    ]

    def test_enable_multimodal_func(self):
        # 遍历所有测试用例，逐个执行测试
        for target_modules, expected_log in self.test_cases:
            # 为每个用例动态构建 hooks_spec
            self.hooks_spec = [
                {
                    "name": "qwen_first_layer_attn_monitor",
                    "target_modules": target_modules,
                    "hook_factory": "test_ascend_forward_hooks2:create_attention_monitor_factory",
                    "config": {
                        "layer_index": 0
                    }
                }
            ]

            # 执行核心测试逻辑
            self._launch_server()
            self.hook_log_file.seek(0)
            hook_content = self.hook_log_file.read()

            # 断言，若失败则明确提示是哪个 name 值出了问题
            self.assertIn(
                expected_log,
                hook_content,
                msg=f"测试 target_modules={target_modules} 失败：未找到期望的日志内容"
            )

class TestSetForwardHooksFieldHookFactoryValidation(TestSetForwardHooks):
    # 定义所有测试用例：(hook_factory值, 期望的断言字符串)
    test_cases = [
        ("abc", "Registered forward hook 'abc' on model.layers.0.self_attn"),
        (3.14, "Registered forward hook '3.14' on model.layers.0.self_attn"),
        (-2, "Registered forward hook '-2' on model.layers.0.self_attn"),
        (None, "Registered forward hook 'None' on model.layers.0.self_attn"),
        ("!@#$", "Registered forward hook '!@#$' on model.layers.0.self_attn"),
    ]

    def test_enable_multimodal_func(self):
        # 遍历所有测试用例，逐个执行测试
        for hook_factory, expected_log in self.test_cases:
            # 为每个用例动态构建 hooks_spec
            self.hooks_spec = [
                {
                    "name": "qwen_first_layer_attn_monitor",
                    "target_modules": ["model.layers.0.self_attn"],
                    "hook_factory": hook_factory,
                    "config": {
                        "layer_index": 0
                    }
                }
            ]

            # 执行核心测试逻辑
            self._launch_server()
            self.hook_log_file.seek(0)
            hook_content = self.hook_log_file.read()

            # 断言，若失败则明确提示是哪个 name 值出了问题
            self.assertIn(
                expected_log,
                hook_content,
                msg=f"测试 hook_factory={hook_factory} 失败：未找到期望的日志内容"
            )


class TestSetForwardHooksFieldConfigValidation(TestSetForwardHooks):
    # 定义所有测试用例：(config值, 期望的断言字符串)
    test_cases = [
        ("abc", "Registered forward hook 'abc' on model.layers.0.self_attn"),
        (3.14, "Registered forward hook '3.14' on model.layers.0.self_attn"),
        (-2, "Registered forward hook '-2' on model.layers.0.self_attn"),
        (None, "Registered forward hook 'None' on model.layers.0.self_attn"),
        ("!@#$", "Registered forward hook '!@#$' on model.layers.0.self_attn"),
    ]

    def test_enable_multimodal_func(self):
        # 遍历所有测试用例，逐个执行测试
        for config, expected_log in self.test_cases:
            # 为每个用例动态构建 hooks_spec
            self.hooks_spec = [
                {
                    "name": "qwen_first_layer_attn_monitor",
                    "target_modules": ["model.layers.0.self_attn"],
                    "hook_factory": "test_ascend_forward_hooks2:create_attention_monitor_factory",
                    "config": config
                }
            ]

            # 执行核心测试逻辑
            self._launch_server()
            self.hook_log_file.seek(0)
            hook_content = self.hook_log_file.read()

            # 断言，若失败则明确提示是哪个 name 值出了问题
            self.assertIn(
                expected_log,
                hook_content,
                msg=f"测试 config={config} 失败：未找到期望的日志内容"
            )

'''

if __name__ == "__main__":
    unittest.main()
