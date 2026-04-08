# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0

"""
Test cases for --ep-dispatch-algorithm parameter
"""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import sglang as sgl
from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestEpDispatchAlgorithmStatic(CustomTestCase):
    """
    Testcase: Verify ep-dispatch-algorithm=static with EPLB enabled

    [Test Category] Parameter
    [Test Target] --ep-dispatch-algorithm=static
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "2",
                "--dp",
                "2",
                "--enable-dp-attention",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "normal",
                "--disable-cuda-graph",
                "--enable-eplb",
                "--ep-num-redundant-experts",
                "4",
                "--eplb-rebalance-num-iterations",
                "50",
                "--expert-distribution-recorder-buffer-size",
                "50",
                "--expert-distribution-recorder-mode",
                "stat",
                "--ep-dispatch-algorithm",
                "static",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu_with_static_dispatch(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.5)


class TestEpDispatchAlgorithmDynamic(CustomTestCase):
    """
    Testcase: Verify ep-dispatch-algorithm=dynamic with EPLB enabled

    [Test Category] Parameter
    [Test Target] --ep-dispatch-algorithm=dynamic
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "2",
                "--dp",
                "2",
                "--enable-dp-attention",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "normal",
                "--disable-cuda-graph",
                "--enable-eplb",
                "--ep-num-redundant-experts",
                "4",
                "--eplb-rebalance-num-iterations",
                "50",
                "--expert-distribution-recorder-buffer-size",
                "50",
                "--expert-distribution-recorder-mode",
                "stat",
                "--ep-dispatch-algorithm",
                "dynamic",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu_with_dynamic_dispatch(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.5)


class TestEpDispatchAlgorithmFake(CustomTestCase):
    """
    Testcase: Verify ep-dispatch-algorithm=fake (for testing/debugging)

    [Test Category] Parameter
    [Test Target] --ep-dispatch-algorithm=fake
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "2",
                "--dp",
                "2",
                "--enable-dp-attention",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "normal",
                "--disable-cuda-graph",
                "--ep-dispatch-algorithm",
                "fake",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_basic_generation_with_fake_dispatch(self):
        """Test basic generation works with fake dispatch algorithm"""
        import requests

        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"max_new_tokens": 8, "temperature": 0.0},
            },
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("text", result)


class TestEpDispatchAlgorithmWithInitExpertLocation(CustomTestCase):
    """
    Testcase: Verify ep-dispatch-algorithm=static with init-expert-location

    [Test Category] Parameter
    [Test Target] --ep-dispatch-algorithm=static with --init-expert-location
    """

    def test_static_dispatch_with_saved_expert_location(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            engine_kwargs = dict(
                model_path=DEFAULT_MLA_MODEL_NAME_FOR_TEST,
                trust_remote_code=True,
                ep_num_redundant_experts=4,
                enable_dp_attention=True,
                moe_a2a_backend="deepep",
                disable_cuda_graph=True,
                expert_distribution_recorder_mode="stat",
                tp_size=2,
                dp_size=2,
                log_level="info",
            )

            # Phase 1: Record expert distribution
            print("Phase 1: Start engine to record expert distribution")
            os.environ["SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR"] = tmp_dir
            engine = sgl.Engine(
                **engine_kwargs,
                disable_overlap_schedule=True,
            )
            engine.start_expert_distribution_record()

            # Generate some text to collect expert distribution
            output = engine.generate(
                prompt=["1+1=2, 2+2=4"],
                sampling_params=dict(max_new_tokens=8, temperature=0.0),
            )
            self.assertIsNotNone(output)

            engine.dump_expert_distribution_record()
            snapshot_path = list(Path(tmp_dir).glob("*.pt"))[0]
            assert snapshot_path is not None
            print(f"Expert distribution saved to: {snapshot_path}")

            engine.shutdown()
            del engine

            # Phase 2: Load expert location with static dispatch
            print("Phase 2: Start engine with init_expert_location and static dispatch")
            engine2 = sgl.Engine(
                **engine_kwargs,
                init_expert_location=str(snapshot_path),
                port=21001,
                ep_dispatch_algorithm="static",
            )

            output2 = engine2.generate(
                prompt=["1+1=2, 2+2=4"],
                sampling_params=dict(max_new_tokens=8, temperature=0.0),
            )
            self.assertIsNotNone(output2)
            print(f"engine.generate output: {output2}")

            engine2.shutdown()
            del engine2


class TestEpDispatchAlgorithmDefault(CustomTestCase):
    """
    Testcase: Verify default behavior when ep-dispatch-algorithm is not set

    [Test Category] Parameter
    [Test Target] Default behavior (ep-dispatch-algorithm=None)
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        # Test without setting ep-dispatch-algorithm
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "2",
                "--dp",
                "2",
                "--enable-dp-attention",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "normal",
                "--disable-cuda-graph",
                "--enable-eplb",
                "--ep-num-redundant-experts",
                "4",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_default_dispatch_with_eplb(self):
        """When enable-eplb is set but ep-dispatch-algorithm is not set,
        it should default to 'static'"""
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=32,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.5)


if __name__ == "__main__":
    unittest.main()
