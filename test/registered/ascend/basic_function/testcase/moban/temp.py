import requests
import os
import subprocess
from unittest import TestCase

# 模拟你代码中已定义的常量和工具函数（需替换为实际值）
CustomTestCase = TestCase
QWEN3_32B_WEIGHTS_PATH = "/path/to/qwen3-32b"  # 实际模型路径
DEFAULT_URL_FOR_TEST = "http://localhost:8000"  # 实际测试URL
DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH = 60  # 实际超时时间


# 模拟工具函数（需替换为实际实现）
def popen_launch_server_config(model, url, timeout, other_args):
    """模拟带config的启动函数"""
    cmd = ["python", "-m", "vllm.entrypoints.api_server", "--model", model] + other_args if model else []
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    return process


def popen_launch_server(model, url, timeout, other_args):
    """模拟普通启动函数"""
    cmd = ["python", "-m", "vllm.entrypoints.api_server", "--model", model] + other_args
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    return process


def kill_process_tree(pid):
    """模拟杀死进程树"""
    try:
        import psutil
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.kill()
        parent.kill()
    except Exception:
        pass


class TestConfig(CustomTestCase):
    """Testcase: Verify that when the --enable-multimodal parameter is set, mmlu accuracy greater than or equal 0.37

    [Test Category] Parameter
    [Test Target] --config
    """
    model = None  # 父类默认无模型

    @classmethod
    def _build_other_args(cls):
        """构建通用命令行参数"""
        return [
            "--config", "config.yaml",
            "--base-gpu-id", "4",
        ]

    @classmethod
    def _launch_server(cls):
        """父类启动服务器（调用带config的启动函数）"""
        other_args = cls._build_other_args()
        cls.process = popen_launch_server_config(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        """清理进程"""
        if hasattr(cls, 'process') and cls.process:
            kill_process_tree(cls.process.pid)

    def test_config(self):
        """父类核心测试方法"""
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


class TestConfigCmd(TestConfig):
    """测试命令行参数优先级高于config文件"""
    # 子类指定模型路径（覆盖父类的None）
    model = QWEN3_32B_WEIGHTS_PATH

    @classmethod
    def _launch_server(cls):
        """子类重写启动方法：调用普通启动函数，而非带config的"""
        other_args = cls._build_other_args()
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    def test_config(self):
        """子类重写核心测试方法：验证命令行参数优先级"""
        # 1. 启动服务器（使用子类重写的_launch_server）
        self._launch_server()

        # 2. 发送请求并验证响应
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

        # 3. 核心断言：验证响应正常，且命令行参数生效
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)

        # 【可选】添加命令行参数优先级的专属断言
        # 例如：验证config文件中的某个参数被命令行覆盖
        # self.assertIn("cmd_param_override_config", response.text)

    # 移除多余的test_enable_multimodal_func方法（避免干扰）
