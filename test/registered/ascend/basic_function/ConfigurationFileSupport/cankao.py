import os
import shlex
import subprocess
import requests
from typing import Optional

# 你原来的常量
CONFIG_YAML_PATH = "your_config.yaml"  # 你的配置文件路径
DEFAULT_URL_FOR_TEST = "http://127.0.0.1:30000"
DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH = 300.0


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

        # 打印命令，方便调试（原版逻辑）
        print(f"[Launch Server] command={shlex.join(command)}")

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
