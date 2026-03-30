import tempfile
import os
import subprocess
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    CONFIG_YAML_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server, _wait_for_server_health, _create_clean_subprocess_env,
)

register_npu_ci(
    est_time=400,
    suite="nightly-4-npu-a3",
    nightly=True,
)
CONFIG_YAML_PATH = (
    "/data/y30082119/mysglang/test/registered/ascend/basic_function/ConfigurationFileSupport/config.yaml"
)


class TestSGLangConfigServer:
    config = CONFIG_YAML_PATH

    @classmethod
    def launch_server_with_config_yaml(cls, config_file, base_url, timeout):
        """
        从 yaml 配置文件启动 SGLang 服务，参考原版 popen_launch_server 逻辑完善
        """
        # --------------------------
        # 1. 从 base_url 自动解析 host / port（和原版完全一致）
        # --------------------------
        _, host, port = base_url.split(":")
        host = host[2:]  # 去掉 "//"

        # --------------------------
        # 2. 构建启动命令（支持 --config + host/port）
        # --------------------------
        command = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--config", config_file,
            "--host", host,
            "--port", port,
        ]

        # # 打印命令，方便调试（原版逻辑）
        # print(f"[Launch Server] command={shlex.join(command)}")

        # --------------------------
        # 3. 创建干净的环境变量
        # --------------------------
        env = _create_clean_subprocess_env(os.environ.copy())

        # --------------------------
        # 4. 启动进程
        # --------------------------
        process = subprocess.Popen(
            command,
            stdout=None,
            stderr=None,
            env=env
        )

        # --------------------------
        # 5. 等待服务健康检查（原版核心逻辑）
        # --------------------------
        try:
            success, error_msg = _wait_for_server_health(process, base_url, None, timeout)
        except Exception as e:
            # 启动失败，杀死进程
            kill_process_tree(process.pid)
            raise RuntimeError(f"服务健康检查失败: {str(e)}") from e

        # --------------------------
        # 6. 失败处理
        # --------------------------
        if not success:
            kill_process_tree(process.pid)
            if "exited" in error_msg:
                raise Exception(error_msg + ". 请检查服务日志")
            raise TimeoutError(f"服务启动超时: {error_msg}")

        return process

    @classmethod
    def setUpClass(cls):
        """测试类启动时：拉起服务"""
        print(f"🚀 启动 SGLang 服务，配置文件: {cls.config}")
        cls.process = cls.launch_server_with_config_yaml(
            cls.config,
            DEFAULT_URL_FOR_TEST,
            DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        )
        print("✅ 服务启动成功")

    @classmethod
    def tearDownClass(cls):
        """测试类结束时：安全杀死整个进程树（原版逻辑）"""
        if hasattr(cls, "process") and cls.process:
            print("🛑 正在关闭服务...")
            try:
                kill_process_tree(cls.process.pid)
                cls.process.wait(timeout=5)
            except Exception as e:
                print(f"⚠️ 关闭服务时出现异常: {e}")
            print("✅ 服务已关闭")

    def test_config_yaml_server_generate(self):
        """测试 yaml 配置启动的服务是否正常生成文本"""
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
            timeout=30
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)
        print("✅ 测试通过：配置文件启动服务正常工作")


'''
class TestConfig(CustomTestCase):
    """Testcase: Verify set --config parameter, can identify the set config and inference request is successfully processed.

    [Test Category] Parameter
    [Test Target] --config
    """

    config = CONFIG_YAML_PATH

    @classmethod
    def launch_server_with_config_yaml(cls, config_file, url, timeout):
        command = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--config",
            config_file,
        ]
        process = subprocess.Popen(command, stdout=None, stderr=None,
                                   env=_create_clean_subprocess_env(os.environ.copy()))
        _wait_for_server_health(process, url, None, timeout)
        return process






    @classmethod
    def setUpClass(cls):
        cls.process = cls.launch_server_with_config_yaml(cls.config, DEFAULT_URL_FOR_TEST,
                                                         DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_config(self):
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


class TestConfigPriority(CustomTestCase):
    """Testcase: Verify set the parameter set in the command line have a higher priority than set in config.yaml,
    set false model path in the command, set right model path in the config.yaml,
    will use false model path service start fail .

    [Test Category] Parameter
    [Test Target] --config
    """

    model = "/nonexistent/Qwen/Qwen3-32B"
    config = CONFIG_YAML_PATH

    def test_config_priority(self):
        error_message = "/nonexistent/Qwen/Qwen3-32B"
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=True, suffix="out.log"
        ) as out_log_file, tempfile.NamedTemporaryFile(
            mode="w+", delete=True, suffix="out.log"
        ) as err_log_file:
            try:
                popen_launch_server(
                    self.model,
                    DEFAULT_URL_FOR_TEST,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=["--config", self.config],
                    return_stdout_stderr=(out_log_file, err_log_file),
                )
            except Exception as e:
                self.assertIn(
                    "Server process exited with code 1.",
                    str(e),
                )
            finally:
                err_log_file.seek(0)
                content = err_log_file.read()
                # error_message information is recorded in the error log
                self.assertIn(error_message, content)
'''

if __name__ == "__main__":
    unittest.main()
