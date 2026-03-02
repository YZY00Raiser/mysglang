import json
import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)

import logging
import time


def create_attention_monitor_factory(config):
    # hook factory
    layer_index = config.get("layer_index", 0)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    def attention_monitor_hook(module, inputs, output):
        # The actual hook function is called during the forward propagation of the self-attention layer.
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
    """Testcase: Verify set --forward-hooks parameter, can identify the set hook function
    and the inference request is successfully processed.

    [Test Category] Parameter
    [Test Target] --forward-hooks
    """

    model = QWEN3_32B_WEIGHTS_PATH
    hooks_spec = [
        {
            "name": "qwen_first_layer_attn_monitor",
            "target_modules": ["model.layers.0.self_attn"],
            "hook_factory": "test_ascend_forward_hooks:create_attention_monitor_factory",
            "config": {"layer_index": 0},
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

    def test_forward_hooks(self):
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


# test set --forward-hooks exception parameter
class TestSetForwardHooksValidation(TestSetForwardHooks):
    """Testcase: Verify set --forward-hooks parameter exception parameter, service start fail.

    [Test Category] Parameter
    [Test Target] --forward-hooks
    """

    def test_forward_hooks_invalid_values(self):
        test_cases = [
            ("abc", 2, "Invalid JSON list: abc"),
            (3.14, -9, "'float' object is not iterable"),
            (-2, -9, "'int' object is not iterable"),
            ("!@#$", 2, "Invalid JSON list: !@#$"),
            (None, 2, "Invalid JSON list: None"),
        ]
        for value, expected_code, expected_msg in test_cases:
            with self.subTest(forward_hooks=value):
                self.forward_hooks = value

                with self.assertRaises(Exception) as ctx:
                    self._launch_server()

                self.assertIn(
                    f"Server process exited with code {expected_code}",
                    str(ctx.exception),
                )

                self.hook_log_file.seek(0)
                hook_content = self.hook_log_file.read()
                self.assertIn(expected_msg, hook_content)


# test set --forward-hooks error parameter field name
class TestSetForwardHooksFieldNameValidation1(TestSetForwardHooks):
    """Testcase: Verify set --forward-hooks parameter field name exception parameter, service start success,
    prompt county official information.

    [Test Category] Parameter
    [Test Target] --forward-hooks
    """

    hooks_spec = [
        {
            "NAME": "qwen_first_layer_attn_monitor",
            "target_modules": ["model.layers.0.self_attn"],
            "hook_factory": "test_ascend_forward_hooks:create_attention_monitor_factory",
            "config": {"layer_index": 0},
        }
    ]

    def test_forward_hooks(self):
        self._launch_server()
        self.hook_log_file.seek(0)
        hook_content = self.hook_log_file.read()
        self.assertIn(
            "Registered forward hook '' on model.layers.0.self_attn", hook_content
        )


class TestSetForwardHooksFieldNameValidation2(TestSetForwardHooks):
    hooks_spec = [
        {
            "name": "qwen_first_layer_attn_monitor",
            "TARGET_modules": ["model.layers.0.self_attn"],
            "hook_factory": "test_ascend_forward_hooks:create_attention_monitor_factory",
            "config": {"layer_index": 0},
        }
    ]

    def test_forward_hooks(self):
        self._launch_server()
        self.hook_log_file.seek(0)
        hook_content = self.hook_log_file.read()
        self.assertIn("has no 'target_modules', skipping", hook_content)


class TestSetForwardHooksFieldNameValidation3(TestSetForwardHooks):
    hooks_spec = [
        {
            "name": "qwen_first_layer_attn_monitor",
            "target_modules": ["model.layers.0.self_attn"],
            "hook_Factory": "test_ascend_forward_hooks:create_attention_monitor_factory",
            "config": {"layer_index": 0},
        }
    ]

    def test_forward_hooks(self):
        self._launch_server()
        self.hook_log_file.seek(0)
        hook_content = self.hook_log_file.read()
        self.assertIn("has no 'hook_factory', skipping", hook_content)


class TestSetForwardHooksFieldNameValidation4(TestSetForwardHooks):
    hooks_spec = [
        {
            "name": "qwen_first_layer_attn_monitor",
            "target_modules": ["model.layers.0.self_attn"],
            "hook_factory": "test_ascend_forward_hooks:create_attention_monitor_factory",
            "Config": {"layer_index": 0},
        }
    ]


# test --forward-hooks parameter fields name, target_modules, hook_factory, config set exception parameters
class TestSetForwardHooksFieldNameParameterValidation(TestSetForwardHooks):
    """Testcase: Verify set --forward-hooks parameter field exception parameter, service start success,
    prompt county official information.

    [Test Category] Parameter
    [Test Target] --forward-hooks
    """

    test_cases = [
        ("abc", "Registered forward hook 'abc' on model.layers.0.self_attn"),
        (3.14, "Registered forward hook '3.14' on model.layers.0.self_attn"),
        (-2, "Registered forward hook '-2' on model.layers.0.self_attn"),
        (None, "Registered forward hook 'None' on model.layers.0.self_attn"),
        ("!@#$", "Registered forward hook 'None' on model.layers.0.self_attn"),
    ]

    def test_forward_hooks(self):
        for name, expected_log in self.test_cases:
            self.hooks_spec = [
                {
                    "name": name,
                    "target_modules": ["model.layers.0.self_attn"],
                    "hook_factory": "test_ascend_forward_hooks:create_attention_monitor_factory",
                    "config": {"layer_index": 0},
                }
            ]

            self._launch_server()
            self.hook_log_file.seek(0)
            hook_content = self.hook_log_file.read()

            self.assertIn(
                expected_log,
                hook_content,
                msg=f"tset config={name} fail：expected log content not found",
            )


class TestSetForwardHooksFieldParameterTargetModulesValidation(TestSetForwardHooks):
    test_cases = [
        ("abc", "Registered forward hook 'abc' on model.layers.0.self_attn"),
        (3.14, "Registered forward hook '3.14' on model.layers.0.self_attn"),
        (-2, "Registered forward hook '-2' on model.layers.0.self_attn"),
        (None, "Registered forward hook 'None' on model.layers.0.self_attn"),
        ("!@#$", "Registered forward hook '!@#$' on model.layers.0.self_attn"),
    ]

    def test_forward_hooks(self):
        for target_modules, expected_log in self.test_cases:
            self.hooks_spec = [
                {
                    "name": "qwen_first_layer_attn_monitor",
                    "target_modules": target_modules,
                    "hook_factory": "test_ascend_forward_hooks:create_attention_monitor_factory",
                    "config": {"layer_index": 0},
                }
            ]

            self._launch_server()
            self.hook_log_file.seek(0)
            hook_content = self.hook_log_file.read()

            self.assertIn(
                expected_log,
                hook_content,
                msg=f"tset config={target_modules} fail：expected log content not found",
            )


class TestSetForwardHooksFieldParameterHookFactoryValidation(TestSetForwardHooks):
    test_cases = [
        ("abc", "Registered forward hook 'abc' on model.layers.0.self_attn"),
        (3.14, "Registered forward hook '3.14' on model.layers.0.self_attn"),
        (-2, "Registered forward hook '-2' on model.layers.0.self_attn"),
        (None, "Registered forward hook 'None' on model.layers.0.self_attn"),
        ("!@#$", "Registered forward hook '!@#$' on model.layers.0.self_attn"),
    ]

    def test_forward_hooks(self):
        for hook_factory, expected_log in self.test_cases:
            self.hooks_spec = [
                {
                    "name": "qwen_first_layer_attn_monitor",
                    "target_modules": ["model.layers.0.self_attn"],
                    "hook_factory": hook_factory,
                    "config": {"layer_index": 0},
                }
            ]

            self._launch_server()
            self.hook_log_file.seek(0)
            hook_content = self.hook_log_file.read()

            self.assertIn(
                expected_log,
                hook_content,
                msg=f"tset config={hook_factory} fail：expected log content not found",
            )


class TestSetForwardHooksFieldParameterConfigValidation(TestSetForwardHooks):
    test_cases = [
        ("abc", "Registered forward hook 'abc' on model.layers.0.self_attn"),
        (3.14, "Registered forward hook '3.14' on model.layers.0.self_attn"),
        (-2, "Registered forward hook '-2' on model.layers.0.self_attn"),
        (None, "Registered forward hook 'None' on model.layers.0.self_attn"),
        ("!@#$", "Registered forward hook '!@#$' on model.layers.0.self_attn"),
    ]

    def test_forward_hooks(self):
        for config, expected_log in self.test_cases:
            self.hooks_spec = [
                {
                    "name": "qwen_first_layer_attn_monitor",
                    "target_modules": ["model.layers.0.self_attn"],
                    "hook_factory": "test_ascend_forward_hooks:create_attention_monitor_factory",
                    "config": config,
                }
            ]

            self._launch_server()
            self.hook_log_file.seek(0)
            hook_content = self.hook_log_file.read()

            self.assertIn(
                expected_log,
                hook_content,
                msg=f"tset config={config} fail：expected log content not found",
            )


if __name__ == "__main__":
    unittest.main()
