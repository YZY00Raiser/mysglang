import unittest

from test_ascend_single_mix_utils import NIC_NAME
from test_ascend_disaggregation_utils import TestAscendDisaggregationUtils

MODEL_PATH = "/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/DeepSeek-V3.2-Exp-W8A8"

MODEL_CONFIG = {
    "model_path": MODEL_PATH,
    "prefill_envs": {
        "SGLANG_SET_CPU_AFFIMITY": "1",
        "LD_LIBRARY_PATH": "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}",
        "PATH": "/usr/local/Ascend/8.5.0/compiler/bisheng/bin:$PATH",
        "ASCEND_HOME_PATH": "/usr/local/Ascend/ascend-toolkit/latest",
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
        "SGLANG_SET_CPU_AFFIMITY": "1",
        "LD_LIBRARY_PATH": "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}",
        "PATH": "/usr/local/Ascend/8.5.0/compiler/bisheng/bin:$PATH",
        "ASCEND_HOME_PATH": "/usr/local/Ascend/ascend-toolkit/latest",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_NPU_USE_MULTI_STREAM": "1",
        "SGLANG_NPU_USE_MLAPO": "1",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "SGLANG_SCHEDULER_SKIP_ALL_GATHER": "1",
        "TASK_QUEUE_ENABLE": "0",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
        "SGLANG_ENABLE_SEPC_V2": "1",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
    },
    "prefill_args": [
        "--nnodes",
        2,
        "--disaggregation-mode",
        "prefill",
        "--tp",
        32,
        "--watchdog-timeout",
        9000,
        "--mem-fraction-static",
        0.73,
        "--disable-radix-cache",
        "--chunked-prefill-size",
        -1,
        "--max-prefill-tokens",
        68000,
        "--max-running-requests",
        1,
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "normal",
        "--quantization",
        "modelslim",
        "--disable-cuda-graph",
        "--enable-nsa-prefill-context-parallel",
        "--moe-dense-tp-size",    
        1,
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        1,
        "--speculative-eagle-topk",
        1,
        "--speculative-num-draft-tokens",
        2,
    ],
    "decode_args": [
        "--nnodes",
        "2",
        "--disaggregation-mode",
        "decode",
        "--tp",
        32,
        "--dp",
        8,
        "--ep",
        32,
        "--moe-dense-tp-size",
        1,
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--watchdog-timeout",
        9000,
        "--mem-fraction-static",
        0.79,
        "--disable-radix-cache",
        "--chunked-prefill-size",
        -1,
        "--max-prefill-tokens",
        68000,
        "--max-running-requests",
        "32",
        "--cuda-graph-max-bs",
        "4",
        "--moe-a2a-backend",
        "ascend_fuseep",
        "--deepep-mode",
        "low_latency",
        "--quantization",
        "modelslim",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        3,
        "--speculative-eagle-topk",
        1,
        "--speculative-num-draft-tokens",
        4,
        "--prefill-round-robin-balance",  
        "--load-balance-method",
        "round_robin",
    ],
}

class Test_DeepSeek_V3_2_W8A8_1P1D(TestAscendDisaggregationUtils):
    model_config = MODEL_CONFIG
    dataset_name = "random"
    request_rate = "inf"
    max_concurrency = 1
    num_prompts = 1
    input_len = 64000
    output_len = 1000
    random_range_ratio = 1
    ttft = 10000
    tpot = 30
    output_token_throughput = 1

    def test_throughput(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
