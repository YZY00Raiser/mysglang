import os
import shlex
from typing import Optional

import requests
import unittest
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    auto_config_device, _try_enable_offline_mode_if_cache_complete, _launch_server_process,
    _wait_for_server_health, popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)
def popen_launch_server_config(
    model: str,
    base_url: str,
    timeout: float,
    api_key: Optional[str] = None,
    other_args: Optional[list[str]] = None,
    env: Optional[dict] = None,
    return_stdout_stderr: Optional[tuple] = None,
    device: str = "auto",
    pd_separated: bool = False,
    num_replicas: Optional[int] = None,
):
    """Launch a server process with automatic device detection and offline/online retry.

    Args:
        model: Model path or identifier
        base_url: Base URL for the server
        timeout: Timeout for server startup
        api_key: Optional API key for authentication
        other_args: Additional command line arguments
        env: Environment dict for subprocess
        return_stdout_stderr: Optional tuple for output capture
        device: Device type ("auto", "cuda", "rocm" or "cpu")
        pd_separated: Whether to use PD separated mode
        num_replicas: Number of replicas for mixed PD mode

    Returns:
        Started subprocess.Popen object
    """
    other_args = other_args or []

    # Auto-detect device if needed
    if device == "auto":
        device = auto_config_device()
        other_args = list(other_args)
        other_args += ["--device", str(device)]

    # CI-specific: Validate cache and enable offline mode if complete
    if env is None:
        env = os.environ.copy()
    else:
        env = env.copy()

    # Store per-run marker path for potential invalidation
    per_run_marker_path = None
    try:
        from sglang.utils import is_in_ci

        if is_in_ci():
            per_run_marker_path = _try_enable_offline_mode_if_cache_complete(
                model, env, other_args
            )
    except Exception as e:
        print(f"CI cache validation failed (non-fatal): {e}")

    # Build server command
    _, host, port = base_url.split(":")
    host = host[2:]

    use_mixed_pd_engine = not pd_separated and num_replicas is not None
    if pd_separated or use_mixed_pd_engine:
        command = "sglang.launch_pd_server"
    else:
        command = "sglang.launch_server"

    command = [
        "python3",
        "-m",
        command,
        # "--model-path",
        # model,
        *[str(x) for x in other_args],
    ]

    if pd_separated or use_mixed_pd_engine:
        command.extend(["--lb-host", host, "--lb-port", port])
    else:
        command.extend(["--host", host, "--port", port])

    if use_mixed_pd_engine:
        command.extend(["--mixed", "--num-replicas", str(num_replicas)])

    if api_key:
        command += ["--api-key", api_key]

    print(f"command={shlex.join(command)}")

    # Track if offline mode was enabled for potential retry
    offline_enabled = env.get("HF_HUB_OFFLINE") == "1"

    # First launch attempt
    process = _launch_server_process(command, env, return_stdout_stderr, model)
    success, error_msg = _wait_for_server_health(process, base_url, api_key, timeout)

    # If offline launch failed and offline was enabled, retry with online mode
    if not success and offline_enabled:
        print(
            f"CI_OFFLINE: Offline launch failed ({error_msg}), retrying with online mode..."
        )

        # Kill failed process
        try:
            if process.poll() is None:
                kill_process_tree(process.pid)
            else:
                process.wait(timeout=5)
        except Exception as e:
            print(f"CI_OFFLINE: Error cleaning up failed offline process: {e}")

        # Invalidate per-run marker to prevent subsequent tests from using offline
        if per_run_marker_path and os.path.exists(per_run_marker_path):
            try:
                os.remove(per_run_marker_path)
                print("CI_OFFLINE: Invalidated per-run marker due to offline failure")
            except Exception as e:
                print(f"CI_OFFLINE: Failed to remove per-run marker: {e}")

        # Retry with online mode
        env["HF_HUB_OFFLINE"] = "0"
        process = _launch_server_process(command, env, return_stdout_stderr, model)
        success, error_msg = _wait_for_server_health(
            process, base_url, api_key, timeout
        )

        if success:
            print("CI_OFFLINE: Online retry succeeded")
            return process

        # Online retry also failed
        try:
            kill_process_tree(process.pid)
        except Exception as e:
            print(f"CI_OFFLINE: Error killing process after online retry failure: {e}")

        if "exited" in error_msg:
            raise Exception(error_msg + ". Check server logs for errors.")
        raise TimeoutError(error_msg)

    # First attempt succeeded or offline was not enabled
    if success:
        return process

    # First attempt failed and offline was not enabled
    try:
        kill_process_tree(process.pid)
    except Exception as e:
        print(f"CI_OFFLINE: Error killing process after first attempt failure: {e}")

    if "exited" in error_msg:
        raise Exception(error_msg + ". Check server logs for errors.")
    raise TimeoutError(error_msg)

class TestConfig(CustomTestCase):
    """Testcase: Verify that when the --enable-multimodal parameter is set, mmlu accuracy greater than or equal 0.37

    [Test Category] Parameter
    [Test Target] --config
    """
    model = None

    @classmethod
    def _build_other_args(cls):
        return [
            "--config", "config.yaml",
            "--base-gpu-id", "4",
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
#命令行设置的参数优先级更高
class TestConfigCmd(TestConfig):
    model = "/data/Qwen/Qwen3-32B"
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

    def test_config(self):
        with self.assertRaises(Exception) as ctx:
            self._launch_server()
        self.assertIn("Server process exited with code 1", str(ctx.exception))
        self.hook_log_file.seek(0)
        hook_content = self.hook_log_file.read()
        self.assertIn("Can't load the configuration of '/data/Qwen/Qwen3-32B'", hook_content)
'''
#--config异常参数
class TestConfigValidation1(TestConfig):
    @classmethod
    def _build_other_args(cls):
        return [
            "--config", "abc",
            "--base-gpu-id", "4",
        ]
    def test_config(self):
        with self.assertRaises(Exception) as ctx:
            self._launch_server()
        self.assertIn("Config file must be YAML format, got: 'abc'", str(ctx.exception))

class TestConfigValidation2(TestConfig):
    @classmethod
    def _build_other_args(cls):
        return [
            "--config", 3.14,
            "--base-gpu-id", "4",
        ]

    def test_config(self):
        # with self.assertRaises(Exception) as ctx:
        self._launch_server()
        # self.assertIn("Config file must be YAML format, got: 3.14", str(ctx.exception))


class TestConfigValidation3(TestConfig):
    @classmethod
    def _build_other_args(cls):
        return [
            "--config", -2,
            "--base-gpu-id", "4",
        ]

    def test_config(self):
        # with self.assertRaises(Exception) as ctx:
        self._launch_server()
        # self.assertIn("Config file must be YAML format, got: -2", str(ctx.exception))



class TestConfigValidation4(TestConfig):
    @classmethod
    def _build_other_args(cls):
        return [
            "--config", None,
            "--base-gpu-id", "4",
        ]

    def test_config(self):
        # with self.assertRaises(Exception) as ctx:
        self._launch_server()
        # self.assertIn("Config file must be YAML format", got: , str(ctx.exception))



class TestConfigValidation5(TestConfig):
    @classmethod
    def _build_other_args(cls):
        return [
            "--config", "!@#$",
            "--base-gpu-id", "4",
        ]

    def test_config(self):
        # with self.assertRaises(Exception) as ctx:
        self._launch_server()
        # self.assertIn("Config file must be YAML format", str(ctx.exception))


class TestConfigValidation6(TestConfig):

    @classmethod
    def _build_other_args(cls):
        return [
            "--config", "config.yaml",
            "--base-gpu-id", "4",
        ]

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
    def test_config(self):
        # with self.assertRaises(Exception) as ctx:
        self._launch_server()
        # self.assertIn("Server process exited with code 2", str(ctx.exception))
        self.hook_log_file.seek(0)
        hook_content = self.hook_log_file.read()
        self.assertIn("Config file not found", hook_content)

#非yaml文件格式
class TestConfigFileModeValidation1(TestConfig):
    @classmethod
    def _build_other_args(cls):
        return [
            "--config", "config.ini",
            "--base-gpu-id", "4",
        ]

    def test_config(self):
        # with self.assertRaises(Exception) as ctx:
        self._launch_server()
        # self.assertIn("must be YAML format", str(ctx.exception))
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

class TestConfigFileModeValidation2(TestConfig):
    @classmethod
    def _build_other_args(cls):
        return [
            "--config", "config.txt",
            "--base-gpu-id", "4",
        ]

    def test_config(self):
        # with self.assertRaises(Exception) as ctx:
        self._launch_server()
        # self.assertIn("must be YAML format", str(ctx.exception))
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

class TestConfigFileModeValidation3(TestConfig):
    @classmethod
    def _build_other_args(cls):
        return [
            "--config", "config.xml",
            "--base-gpu-id", "4",
        ]

    def test_config(self):
        # with self.assertRaises(Exception) as ctx:
        self._launch_server()
        # self.assertIn("must be YAML format", str(ctx.exception))
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

#配置错误的参数

class TestConfigParamValidation(TestConfig):
    @classmethod
    def _build_other_args(cls):
        return [
            "--config", "_config.yaml",
            "--base-gpu-id", "4",
        ]

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

    def test_config(self):
        # with self.assertRaises(Exception) as ctx:
        self._launch_server()
        # self.assertIn("must be YAML format", str(ctx.exception))
        self.hook_log_file.seek(0)
        hook_content = self.hook_log_file.read()
        self.assertIn("--tp-size: invalid int value: 'abcd'", hook_content)
'''

if __name__ == "__main__":
    unittest.main()
