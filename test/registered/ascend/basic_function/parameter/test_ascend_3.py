# 导入基础系统/工具库
import os
import random
import tempfile
import time
import unittest
from typing import Dict  # 类型注解，限定字典返回值

# 导入HTTP请求库
import requests

# 从SGLang库导入基准测试/服务相关工具
from sglang.bench_serving import get_tokenizer  # 获取模型的分词器
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,  # 解耦架构Prefill-Decode服务的基础测试类
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,  # 服务启动的默认超时时间
    popen_launch_pd_server,  # 以子进程方式启动Prefill/Decode解耦服务
)

from sglang.test.ascend.performance.test_ascend_performance_utils import (
    TestAscendMultiNodePdSepTestCaseBase,
    DEEPSEEK_V32_W8A8_MODEL_PATH,
    NIC_NAME
)


# 测试用的模型路径（替换了原默认模型，指定昇腾VLLM的DeepSeek-V3.2量化模型）
DEFAULT_MODEL_NAME_FOR_TEST = "/root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V3.2-W8A8"


class DisaggregationHiCacheBase(PDDisaggregationServerBase):
    """
    解耦架构+HiCache功能的测试基类
    封装公共的服务启动、工具方法、请求发送逻辑，供子类继承复用
    继承自PDDisaggregationServerBase（提供解耦服务的基础配置：端口、URL、传输后端等）
    """

    @classmethod
    def setUpClass(cls):
        """
        测试类的初始化方法（unittest类方法，所有测试用例执行前仅运行一次）
        启动Prefill/Decode服务、初始化分词器、创建临时目录、启动负载均衡、检查服务健康性
        """
        # 调用父类的初始化方法，加载基础配置（如prefill_url/decode_url/lb_url等）
        super(DisaggregationHiCacheBase, cls).setUpClass()

        # 设置测试使用的模型路径
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        # 初始化模型对应的分词器（用于生成测试prompt、解码token）
        cls.tokenizer = get_tokenizer(cls.model)
        # 创建临时目录，用于HiCache的文件后端存储（缓存的KV数据会存在这里）
        cls.temp_dir = tempfile.mkdtemp()
        # 启动Prefill服务（子类可重写start_prefill定制启动参数）
        cls.start_prefill()
        # 启动Decode服务（子类必须重写，因为不同测试的Decode配置不同）
        cls.start_decode()

        # 阻塞等待Prefill/Decode服务启动完成（检查健康检查接口）
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        # 启动负载均衡服务（解耦架构中，客户端请求通过LB转发到Prefill/Decode）
        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        """
        启动**启用HiCache**的Prefill服务
        Prefill节点负责文本的预填充推理，是解耦架构的核心节点之一
        这里封装了HiCache的核心启动参数，所有子类复用该逻辑
        """
        # Prefill服务的启动参数列表
        prefill_args = [
            "--trust-remote-code",  # 信任模型的远程代码（自定义模型必备）
            "--disaggregation-mode", "prefill",  # 标记为Prefill解耦模式
            "--tp-size", "1",  # 张量并行度为1（单卡运行）
            "--page-size", "64",  # KV缓存的页大小为64token（HiCache基于页管理）
            "--enable-hierarchical-cache",  # 核心：启用分层缓存（HiCache）
            "--hicache-ratio", "1.2",  # HiCache的缓存比例（预留20%缓存空间）
            "--hicache-size", "0",  # HiCache内存大小（0表示自动适配）
            "--hicache-write-policy", "write_through",  # 写策略：直写（内存写同时刷到存储）
            "--hicache-storage-backend", "file",  # HiCache的存储后端：文件系统（临时目录）
            "--hicache-storage-prefetch-policy", "wait_complete",  # 预取策略：等待预取完成再推理
            "--mem-fraction-static", "0.8",  # 模型静态占用GPU内存的比例（80%）
        ]
        # 添加传输后端参数+RDMA设备参数（父类定义，如TCP/RDMA传输方式）
        prefill_args += cls.transfer_backend + cls.rdma_devices
        # 设置环境变量：指定HiCache文件后端的存储目录（前面创建的临时目录）
        env = {
            **os.environ,  # 继承系统原有环境变量
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
        }
        # 以子进程方式启动Prefill服务，返回进程句柄（便于后续管理）
        cls.process_prefill = popen_launch_pd_server(
            cls.model,  # 测试模型路径
            cls.prefill_url,  # Prefill服务的访问URL
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,  # 服务启动超时时间
            other_args=prefill_args,  # 自定义启动参数
            env=env,  # 环境变量
        )

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
        response = requests.post(
            f"{self.lb_url}/generate",  # 负载均衡的生成接口
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



NIC_NAME="enp23s0f3"
MODEL_CONFIG = {
    "model_path": DEEPSEEK_V32_W8A8_MODEL_PATH,
    "prefill_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "HCCL_BUFFSIZE": "1024",
        "DEEPEP_NORMAL_LONG_SEQ_ROUND": "5",
        "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "512",
        "SGLANG_NPU_USE_MLAPO": "1",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
        "SGLANG_NPU_USE_MULTI_STREAM": "1",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
    },
    "decode_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_NPU_USE_MULTI_STREAM": "1",
        "SGLANG_NPU_USE_MLAPO": "1",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "SGLANG_SCHEDULER_SKIP_ALL_GATHER": "1",
        "TASK_QUEUE_ENABLE": "0",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
        "SGLANG_ENABLE_SPEC_V2": "1",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
        "HCCL_BUFFSIZE": "400",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "8",
    },
    "router_envs": {
        "SGLANG_DP_ROUND_ROBIN": "1",
    },
    "prefill_args": [
        "--nnodes", 2,
        "--disaggregation-mode", "prefill",
        "--tp", 32,
        "--watchdog-timeout", 9000,
        "--mem-fraction-static", 0.73,
        "--disable-radix-cache",
        "--chunked-prefill-size", -1,
        "--max-prefill-tokens", 68000,
        "--max-running-requests", 1,
        "--moe-a2a-backend", "deepep",
        "--deepep-mode", "normal",
        "--quantization", "modelslim",
        "--disable-cuda-graph",
        "--enable-nsa-prefill-context-parallel",
        "--moe-dense-tp-size", 1,
        "--speculative-algorithm", "NEXTN",
        "--speculative-num-steps", 1,
        "--speculative-eagle-topk", 1,
        "--speculative-num-draft-tokens", 2,
    ],
    "decode_args": [
        "--nnodes", 2,
        "--disaggregation-mode", "decode",
        "--tp", 32,
        "--dp", 8,
        "--ep", 32,
        "--moe-dense-tp-size", 1,
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--watchdog-timeout", 9000,
        "--mem-fraction-static", 0.79,
        "--disable-radix-cache",
        "--chunked-prefill-size", -1,
        "--max-prefill-tokens", 68000,
        "--max-running-requests", 32,
        "--cuda-graph-max-bs", 4,
        "--moe-a2a-backend", "deepep",
        "--deepep-mode", "low_latency",
        "--quantization", "modelslim",
        "--speculative-algorithm", "NEXTN",
        "--speculative-num-steps", 3,
        "--speculative-eagle-topk", 1,
        "--speculative-num-draft-tokens", 4,
        "--prefill-round-robin-balance",
        "--load-balance-method", "round_robin",
    ],
    "router_args": [
        "--mini-lb",
    ],
}

class TestDeepSeekV32(TestAscendMultiNodePdSepTestCaseBase):
    model_config = MODEL_CONFIG
    dataset_name = "random"
    request_rate = "inf"
    max_concurrency = 32
    num_prompts = 64
    input_len = 64000
    output_len = 3000
    random_range_ratio = 1
    tpot = 27.3
    # T: 4.7@26ms        800I: None          Dev-800I: 471/ 32
    output_token_throughput = 433

    def test_throughput(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
