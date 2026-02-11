import unittest
import requests
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

class TestRandomSeedBase(CustomTestCase):
    """基础测试类：封装通用逻辑"""
    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    random_seed = None

    @classmethod
    def setUpClass(cls):
        if cls.random_seed is None:
            raise ValueError("必须设置 random_seed 属性")
        other_args = [
            "--attention-backend", "ascend",
            "--disable-cuda-graph",
            "--random-seed", str(cls.random_seed),  # 确保参数为字符串
        ]
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def get_model_output(self, prompt):
        """封装请求接口的逻辑，增加异常处理"""
        try:
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 1,
                        "max_new_tokens": 32,
                    },
                },
                timeout=30  # 设置请求超时
            )
            response.raise_for_status()  # 抛出 HTTP 状态码异常
            return response.json()["text"]
        except requests.exceptions.RequestException as e:
            self.fail(f"请求接口失败: {str(e)}")
        except KeyError as e:
            self.fail(f"响应格式错误，缺少 {e} 字段")

    def test_same_seed_same_output(self):
        """测试相同随机种子输出一致"""
        prompt = "The capital of France is"
        output1 = self.get_model_output(prompt)
        output2 = self.get_model_output(prompt)
        self.assertEqual(output1, output2, "相同随机种子输出不一致")

class TestRandomSeedZero(TestRandomSeedBase):
    """测试随机种子为 0 的情况"""
    random_seed = 0

class TestRandomSeedOne(TestRandomSeedBase):
    """测试随机种子为 1 的情况"""
    random_seed = 1

class TestDifferentSeedDifferentOutput(CustomTestCase):
    """测试不同随机种子输出不同"""
    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

    def test_different_seed_different_output(self):
        # 启动 seed=0 的服务并获取输出
        process0 = self._launch_server(0)
        output0 = self._get_output(process0)
        kill_process_tree(process0.pid)

        # 启动 seed=1 的服务并获取输出
        process1 = self._launch_server(1)
        output1 = self._get_output(process1)
        kill_process_tree(process1.pid)

        # 断言不同种子输出不同
        self.assertNotEqual(output0, output1, "不同随机种子输出一致")

    def _launch_server(self, seed):
        """辅助方法：启动指定种子的服务"""
        other_args = [
            "--attention-backend", "ascend",
            "--disable-cuda-graph",
            "--random-seed", str(seed),
        ]
        return popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    def _get_output(self, process):
        """辅助方法：获取模型输出"""
        try:
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"temperature": 1, "max_new_tokens": 32},
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["text"]
        except Exception as e:
            kill_process_tree(process.pid)
            self.fail(f"获取输出失败: {str(e)}")

if __name__ == "__main__":
    unittest.main()
