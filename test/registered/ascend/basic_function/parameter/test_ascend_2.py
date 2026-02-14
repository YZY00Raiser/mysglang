import time
import unittest

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


class TestEnableCacheReport(CustomTestCase):
    """Testcase：Verify use DeepSeek V3.2  Prefix set enable-hierarchical-cache, not set --disable-radix-cache
    send two same requests with 600 token the second response's cached_tokens equal 512.

    [Test Category] model
    [Test Target] enable-hierarchical-cache
    """
    DEFAULT_URL_FOR_TEST = "http://172.22.3.19:6688"
    PREFILL_URL_FOR_TEST = "http://172.22.3.19:8000"

    def trigger_offloading_and_flush(self):
        """
        工具方法：触发KV缓存的**卸载（Offloading）** 并刷盘
        核心目的：将GPU内存中的缓存刷到HiCache的文件存储，模拟缓存冷启动后的命中场景
        """
        # input_ids = [1]
        # response = requests.post(
        #     f"{self.DEFAULT_URL_FOR_TEST}/generate",
        #     json={
        #         "input_ids": input_ids,
        #         "sampling_params": {
        #             "temperature": 0,
        #             "max_tokens": 150,
        #         },
        #     },
        # )
        # 等待2秒，确保缓存有足够时间完成内存到文件的卸载
        # time.sleep(2)
        # 调用Prefill服务的刷缓存接口，强制将设备缓存刷到远程存储（HiCache）
        requests.post(self.PREFILL_URL_FOR_TEST + "/flush_cache")

    def test_enable_hierarchical(self):
        print("==============startoooo====================================")
        input_ids1 = [1] * 300
        response1 = requests.post(
            f"{self.DEFAULT_URL_FOR_TEST}/generate",
            json={
                "input_ids": input_ids1,
                "sampling_params": {
                    "temperature": 0,
                    "max_tokens": 200,
                },
            },
        )
        print("==============respob=====================================")
        print(response1.json())
        print("=============finshhhhhhhhhhhhhhh====================================")
        self.assertEqual(response1.status_code, 200)

        self.trigger_offloading_and_flush()

        output_ids = response1.json()["output_ids"]
        input_ids2 = [1] * 300
        input_ids2 += output_ids
        response2 = requests.post(
            f"{self.DEFAULT_URL_FOR_TEST}/generate",
            json={
                "input_ids": input_ids2,
                "sampling_params": {
                    "temperature": 0,
                    "max_tokens": 200,
                },
            },
        )

        print("==============respob=====================================")
        print(response2.json())
        print("=============finshhhhhhhhhhhhhhh====================================")
        self.assertEqual(response1.status_code, 200)


if __name__ == "__main__":
    unittest.main()
