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
        cls.prefill_nodes = ["http://192.168.0.188:24667"]
        cls.decode_nodes = ["http://192.168.0.244:24667"]
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


class TestDisaggregationPrefillWithHiCache(DisaggregationHiCacheBase):
    """
    测试场景1：**仅Prefill节点启用HiCache**，Decode节点不启用缓存卸载
    验证单节点HiCache的基础缓存命中功能，是最基础的HiCache解耦测试
    """

    @classmethod
    def start_decode(cls):
        """
        重写基类的start_decode：启动**不启用HiCache**的Decode服务
        Decode节点负责token的逐一生成，这里指定使用GPU 1，与Prefill的GPU 0隔离
        """
        decode_args = [
            "--trust-remote-code",  # 信任远程代码
            "--disaggregation-mode", "decode",  # 标记为Decode解耦模式
            "--tp-size", "1",  # 张量并行度1
            "--page-size", "64",  # 与Prefill保持一致的页大小（解耦架构必须统一）
            "--mem-fraction-static", "0.8",  # GPU内存占用比例80%
            "--base-gpu-id", "1",  # 指定Decode服务使用GPU 1（避免与Prefill的GPU 0冲突）
        ]
        # 添加传输后端+RDMA设备参数
        decode_args += cls.transfer_backend + cls.rdma_devices
        # 环境变量：复用HiCache的临时存储目录（虽不启用，但保持配置统一）
        env = {
            **os.environ,
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
        }
        # 启动Decode服务，返回进程句柄
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=env,
        )

    def test_prefill_cache_hit(self):
        """
        核心测试用例：验证Prefill节点的缓存命中功能
        逻辑：相同prompt的两次请求，第二次在缓存刷盘后应命中大量缓存token
        """
        # 生成800个token的长prompt（足够触发HiCache的缓存机制）
        repeated_prompt = self.gen_prompt(800)

        # 第一次请求：缓存未命中（冷启动），Prefill会生成KV缓存并卸载到文件
        self.send_request(repeated_prompt, max_tokens=100)

        # 触发缓存卸载并刷盘（将GPU内存的缓存写到HiCache文件存储）
        self.trigger_offloading_and_flush()

        # 第二次请求：相同prompt，应从HiCache文件加载缓存，触发**缓存命中**
        response2 = self.send_request(repeated_prompt, max_tokens=100)

        # 断言缓存命中的token数大于700（验证高缓存命中率）
        self.assertGreater(response2["meta_info"]["cached_tokens"], 700)


class TestDisaggregationDecodeWithHiCache(DisaggregationHiCacheBase):
    """
    测试场景2：**Prefill+Decode双节点均启用HiCache**
    Decode节点额外启用KV缓存卸载，验证更复杂的多轮对话场景下的缓存优化效果
    """

    @classmethod
    def start_decode(cls):
        """
        重写基类的start_decode：启动**启用HiCache**的Decode服务
        新增Decode节点的缓存卸载参数，指定使用GPU 1，与Prefill隔离
        """
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode", "decode",
            "--tp-size", "1",
            "--page-size", "64",  # 与Prefill页大小统一
            "--mem-fraction-static", "0.8",
            "--base-gpu-id", "1",  # Decode使用GPU 1
            # Decode节点核心参数：启用KV缓存卸载
            "--disaggregation-decode-enable-offload-kvcache",
            # HiCache相关配置（与Prefill保持一致）
            "--hicache-ratio", "1.2",
            "--hicache-size", "0",
            "--hicache-storage-backend", "file",
            "--hicache-storage-prefetch-policy", "wait_complete",
        ]
        # 添加传输后端+RDMA设备参数
        decode_args += cls.transfer_backend + cls.rdma_devices
        # 环境变量：指定HiCache文件存储目录
        env = {
            **os.environ,
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
        }
        # 启动启用HiCache的Decode服务
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=env,
        )

    def test_multi_turn_conversation_cache(self):
        """
        核心测试用例：验证**多轮对话**场景下的缓存命中优化
        逻辑：多轮对话的上下文不断累积，后续轮次应命中更多缓存token，提升推理效率
        """
        print("=== Multi-turn Conversation Cache Test ===")

        # 第一轮：生成300个token的初始prompt，作为对话的起始上下文
        initial_prompt = self.gen_prompt(300)
        # 发送第一轮请求，生成200个新token，采样温度0.1（轻微随机）
        response1 = self.send_request(initial_prompt, max_tokens=200, temperature=0.1)
        # 拼接初始prompt+生成的文本，作为下一轮的上下文（模拟多轮对话）
        current_context = initial_prompt + response1["text"]

        # 初始化上一轮的缓存命中数为0，用于对比后续轮次的提升
        previous_cached_tokens = 0

        # 执行2-4轮对话（共3轮），基于上一轮的上下文继续生成
        for turn in range(2, 5):
            print(f"\nTurn {turn}: Continuing from previous context")

            # 发送请求，基于累积的上下文生成文本
            response = self.send_request(
                current_context, max_tokens=200, temperature=0.1
            )
            # 获取当前轮次的缓存命中token数（从响应的元信息中提取）
            cached_tokens = response["meta_info"]["cached_tokens"]

            # 打印当前轮次的缓存命中数和相比上一轮的提升量
            print(f"Turn {turn} cached tokens: {cached_tokens}")
            print(f"Improvement: {cached_tokens - previous_cached_tokens} tokens")

            # 断言：当前轮次的缓存命中数必须大于上一轮（验证多轮缓存优化）
            self.assertGreater(
                cached_tokens,
                previous_cached_tokens,
                f"Turn {turn} should have more cached tokens than turn {turn-1}",
            )

            # 更新上下文（拼接上一轮的上下文+本次生成的文本）
            current_context += response["text"]
            # 更新上一轮的缓存命中数，用于下一轮对比
            previous_cached_tokens = cached_tokens

            # 每轮结束后刷盘Prefill缓存，模拟真实场景的缓存持久化
            self.trigger_offloading_and_flush()


# 程序入口：执行所有unittest测试用例
if __name__ == "__main__":
    unittest.main()
