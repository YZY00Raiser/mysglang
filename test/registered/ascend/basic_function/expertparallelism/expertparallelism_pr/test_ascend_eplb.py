import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import sglang as sgl
from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
# from sglang.test.ascend.test_ascend_utils import DEEPSEEK_R1_0528_W8A8_WEIGHTS_PATH as MODEL_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=200, suite="nightly-16-npu-a3", nightly=True)

class _BaseTestDynamicEPLB(CustomTestCase):
    """
    Testcase：Verify the correctness of the function when eplb is enabled and the accuracy of DeepSeek-R1-W8A8 on MMLU
    dataset

    [Test Category] Parameter
    [Test Target] --enable-eplb， --ep-num-redundant-experts, --eplb-rebalance-num-iterations,
    --expert-distribution-recorder-buffer-size, --enable-expert-distribution-metrics,
    --expert-distribution-recorder-mode, --ep-dispatch-algorithm
    """

    extra_args = []

    @classmethod
    def setUpClass(cls):
        # cls.model = MODEL_PATH
        cls.model = "/root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-R1-0528-W8A8"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "16",
                "--dp-size",
                "1",
                "--attention-backend",
                "ascend",
                "--quantization",
                "modelslim",
                "--mem-fraction-static",
                "0.9",
                "--enable-dp-attention",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "normal",
                "--disable-cuda-graph",
                "--enable-eplb",
                "--ep-num-redundant-experts",
                "16",
                "--eplb-rebalance-num-iterations",
                "50",
                "--expert-distribution-recorder-buffer-size",
                "50",
                "--enable-expert-distribution-metrics",
                "--expert-distribution-recorder-mode",
                "stat",
                "--ep-dispatch-algorithm",
                "static",
                *cls.extra_args,
            ],
            env={
                "SGL_ENABLE_JIT_DEEPGEMM": "0",
                "HCCL_BUFFSIZE": "500",
                **os.environ,
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=8,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.5)


class TestDynamicEPLBSimple(_BaseTestDynamicEPLB):
    pass


class TestDynamicEPLBMultiChunk(_BaseTestDynamicEPLB):
    extra_args = ["--eplb-rebalance-layers-per-chunk", "1"]


class TestStaticEPLB(CustomTestCase):
    def test_save_expert_distribution_and_init_expert_location(self):
        os.environ["SGL_ENABLE_JIT_DEEPGEMM"] = "0"
        os.environ["HCCL_BUFFSIZE"] = "500"

        with tempfile.TemporaryDirectory() as tmp_dir:
            engine_kwargs = dict(
                model_path=MODEL_PATH,
                trust_remote_code=True,
                attention_backend="ascend",
                quantization="modelslim",
                mem_fraction_static=0.9,
                deepep_mode="normal",
                ep_num_redundant_experts=16,
                enable_dp_attention=True,
                moe_a2a_backend="deepep",
                disable_cuda_graph=True,
                expert_distribution_recorder_mode="stat",
                tp_size=16,
                dp_size=1,
                log_level="info",
                enable_expert_distribution_metrics=True,
            )

            print(f"Action: start engine")
            os.environ["SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR"] = tmp_dir
            engine = sgl.Engine(
                **engine_kwargs,
                disable_overlap_schedule=True,
            )
            engine.start_expert_distribution_record()
            self._assert_engine_generate_correct(engine)

            print(f"Action: dump_expert_distribution_record")
            engine.dump_expert_distribution_record()
            snapshot_path = list(Path(tmp_dir).glob("*.pt"))[0]
            assert snapshot_path is not None
            print(f"{snapshot_path=}")

            print(f"Action: shutdown engine")
            engine.shutdown()
            del engine

            print(f"Action: start engine with init_expert_location")
            engine = sgl.Engine(
                **engine_kwargs,
                init_expert_location=str(snapshot_path),
                port=21000,
                ep_dispatch_algorithm="static",
            )
            self._assert_engine_generate_correct(engine)
            print(f"Action: shutdown engine")
            engine.shutdown()
            del engine

    def _assert_engine_generate_correct(self, engine: sgl.Engine):
        output = engine.generate(
            prompt=["1+1=2, 2+2=4", "One plus one is two, two plus two is four"],
            sampling_params=dict(max_new_tokens=8, temperature=0.0),
        )
        print(f"engine.generate {output=}")
        self.assertEqual(
            [x["text"] for x in output],
            [", 4+4=8,", ", four plus four is eight, eight"],
        )


if __name__ == "__main__":
    unittest.main()
