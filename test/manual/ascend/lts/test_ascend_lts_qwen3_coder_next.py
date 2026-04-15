import datetime
import logging
import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.lts_utils import TestAscendLtsTestCaseBase
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    QWEN3_NEXT_80B_A3B_MODEL_PATH,
)
from sglang.test.test_utils import (
    popen_launch_server,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-Coder-Next_W8A8"

ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "160",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "20",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "3280",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "TASK_QUEUE_ENABLE": "1",
    "ASCEND_USE_FIA": "1",
    "SGLANG_NPU_USE_MULTI_STREAM": "0",
    "ASCEND_LAUNCH_BLOCKING": "1",
    "SGLANG_WARMUP_TIMEOUT": "3600",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "FORCE_DRAFT_MODEL_NON_QUANT": "1",
    "HCCL_BUFFSIZE": "2000",
    "ZBCCL_LOCAL_MEM_SIZE": "60416",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "ZBCCL_NPU_ALLOC_CONF": "use_vmm_for_static_memory:True",
    "ZBCCL_ENABLE_GRAPH": "1",
}

OTHER_ARGS = [
    "--page-size",
    128,
    "--tp-size",
    4,
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--watchdog-timeout",
    9000,
    "--mem-fraction-static",
    0.75,
    "--disable-radix-cache",
    "--max-prefill-tokens",
    65536,
    # "--context-length",
    # 26384,
    "--max-total-tokens",
    1700000,
    "--dp-size",
    2,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--speculative-draft-model-quantization",
    "unquant",
    "--chunked-prefill-size",
    -1,
    "--max-running-requests",
    80,
    "--cuda-graph-bs",
    50,
    "--mamba-ssm-dtype",
    "bfloat16",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--speculative-draft-model-path",
    QWEN3_NEXT_80B_A3B_MODEL_PATH,
    "--quantization",
    "modelslim",
    "--base-gpu-id",
    12,
]


class TestLTSQwen3CoderNext(TestAscendLtsTestCaseBase):
    model = MODEL_PATH
    other_args = OTHER_ARGS
    envs = ENVS
    max_concurrency = 80
    num_prompts = int(max_concurrency) * 4
    input_len = 3500
    output_len = 1500
    random_range_ratio = 0.5
    ttft = 10000
    tpot = 100
    output_token_throughput = 500
    accuracy = {"gsm8k": 0.90, "mmlu": 0.80}

    @classmethod
    def setUpClass(cls):
        cls.host = "0.0.0.0"
        cls.port = 30010
        cls.base_url = f"http://{cls.host}:{cls.port}"
        env = os.environ.copy()
        for key, value in env.items():
            logger.info(f"ENV_VAR_SYS {key}:{value}")
        if cls.envs:
            for key, value in cls.envs.items():
                logger.info(f"ENV_VAR_CASE {key}:{value}")
                env[key] = value

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=cls.timeout,
            other_args=cls.other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def testLtsQwen3CoderNext(self):
        i = 0
        while True:
            i = i + 1
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(
                f"====={current_time}  Execute the {i}-th long-term stability test====="
            )

            # self.run_long_seq_testcase()
            self.run_mmlu()
            self.run_gsm8k()
            self.run_throughput()
            # self.run_evalscope()


if __name__ == "__main__":
    unittest.main()
