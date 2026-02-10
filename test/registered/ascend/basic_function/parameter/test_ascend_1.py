import os
import random
import tempfile
import time
import unittest
from typing import Dict

import requests
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.bench_serving import get_tokenizer
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)

# 测试用的模型路径（替换了原默认模型，指定昇腾VLLM的DeepSeek-V3.2量化模型）
DEFAULT_MODEL_NAME_FOR_TEST = "/root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V3.2-W8A8"


class DisaggregationHiCacheBase(CustomTestCase):
    """Base class for disaggregation with HiCache tests"""
    @classmethod
    def setUpClass(cls):
        # 设置测试使用的模型路径
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        # 初始化模型对应的分词器（用于生成测试prompt、解码token）
        cls.tokenizer = get_tokenizer(cls.model)

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
        """Send a generate request and return response"""
        response = requests.post(
            f"{self.lb_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=60,
        )

        self.assertEqual(
            response.status_code,
            200,
            f"Request failed: {response.status_code} - {response.text}",
        )
        print("----------------------------------respasjisda---------------------------------------------")
        print(response.json())
        return response.json()

    def test_prefill_cache_hit(self):
        """
        核心测试用例：验证Prefill节点的缓存命中功能
        逻辑：相同prompt的两次请求，第二次在缓存刷盘后应命中大量缓存token
        """
        print("-----------------------------------test_prefill_cache_hit-----------------------------------")
        repeated_prompt = self.gen_prompt(800)
        # 第一次请求：缓存未命中（冷启动），Prefill会生成KV缓存并卸载到文件
        self.send_request(repeated_prompt, max_tokens=100)

        # 第二次请求：相同prompt，应从HiCache文件加载缓存，触发**缓存命中**
        response2 = self.send_request(repeated_prompt, max_tokens=100)

        # 断言缓存命中的token数大于700（验证高缓存命中率）
        self.assertGreater(response2["meta_info"]["cached_tokens"], 700)


if __name__ == "__main__":
    unittest.main()
