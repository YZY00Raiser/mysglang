import os
import random
import tempfile
import time
import unittest
from typing import Dict  # 类型注解，限定字典返回值

# 从SGLang库导入基准测试/服务相关工具
from sglang.bench_serving import get_tokenizer  # 获取模型的分词器
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)
import requests
from sglang.test.test_utils import (
    CustomTestCase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)
class DisaggregationHiCacheBase(PDDisaggregationServerBase):
    """
    解耦架构+HiCache功能的测试基类
    封装公共的服务启动、工具方法、请求发送逻辑，供子类继承复用
    继承自PDDisaggregationServerBase（提供解耦服务的基础配置：端口、URL、传输后端等）
    """

    @classmethod
    def setUpClass(cls):
        print("setUpClass")

    @classmethod
    def start_prefill(cls):
        print("start_prefill")

    @classmethod
    def start_decode(cls):
        """
        Decode服务的启动方法（基类中仅占位，由子类重写）
        原因：不同测试的Decode节点配置不同（是否启用HiCache、GPU ID等）
        """
        pass

    def gen_prompt(self, token_num: int) -> str:
        """
        生成指定token数量的随机测试提示词（prompt）
        用于模拟不同长度的用户请求，触发缓存的加载/卸载
        :param token_num: 要生成的token数量
        :return: 解码后的字符串prompt
        """
        # 获取分词器的词表所有token的ID值
        all_available_tokens = list(self.tokenizer.get_vocab().values())
        # 随机选择指定数量的token（可重复）
        selected_tokens = random.choices(all_available_tokens, k=token_num)
        # 将token ID解码为人类可读的字符串
        return self.tokenizer.decode(selected_tokens)

    def send_request(
        self, prompt: str, max_tokens: int = 100, temperature: float = 0.0
    ) -> Dict:
        """
        向负载均衡服务发送生成请求，封装请求逻辑并做基础校验
        :param prompt: 输入的提示词
        :param max_tokens: 最大生成新token数
        :param temperature: 采样温度（0表示确定性生成，无随机）
        :return: 服务返回的json字典
        """
        DEFAULT_URL_FOR_TEST = "http://172.22.3.19:6688"
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",  # 负载均衡的生成接口
            json={
                "text": prompt,  # 输入提示词
                "sampling_params": {  # 采样参数
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "ignore_eos": True,  # 忽略结束符，强制生成到max_tokens
                },
            },
            timeout=60,  # HTTP请求超时时间60秒
        )

        # 断言请求成功（状态码200），失败则打印错误信息
        self.assertEqual(
            response.status_code,
            200,
            f"Request failed: {response.status_code} - {response.text}",
        )
        # 返回json格式的响应结果
        return response.json()

    def trigger_offloading_and_flush(self):
        """
        工具方法：触发KV缓存的**卸载（Offloading）** 并刷盘
        核心目的：将GPU内存中的缓存刷到HiCache的文件存储，模拟缓存冷启动后的命中场景
        """
        # 发送一个短请求，触发Prefill节点的推理和KV缓存生成（进而触发缓存卸载）
        self.send_request(self.gen_prompt(1), max_tokens=150)

        # 等待2秒，确保缓存有足够时间完成内存到文件的卸载
        time.sleep(2)
        # 调用Prefill服务的刷缓存接口，强制将设备缓存刷到远程存储（HiCache）
        requests.post(self.prefill_url + "/flush_cache")


class TestEnableCacheReport(CustomTestCase):
    """Testcase：Verify use DeepSeek V3.2  Prefix set enable-hierarchical-cache, not set --disable-radix-cache
    send two same requests with 600 token the second response's cached_tokens equal 512.

    [Test Category] model
    [Test Target] enable-hierarchical-cache
    """
    def test_enable_hierarchical(self):
        DEFAULT_URL_FOR_TEST="http://172.22.3.19:6688"
        print("==============startoooo====================================")
        input_ids = [1] * 300
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_tokens": 200,
                },
            },
        )

        self.trigger_offloading_and_flush



        print("==============respob=====================================")
        print(response.json())
        print("=============finshhhhhhhhhhhhhhh====================================")
        self.assertEqual(response.status_code, 200)

if __name__ == "__main__":
    unittest.main()
