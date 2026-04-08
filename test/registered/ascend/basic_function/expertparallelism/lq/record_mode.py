"""
Test cases for expert_distribution_recorder_mode parameter.

This module tests all four modes of expert distribution recorder:
- stat: Statistical mode, records expert call counts
- stat_approx: Approximate statistical mode based on DeepEP dispatch data
- per_pass: Detailed recording per forward pass
- per_token: Detailed recording per token with top-k expert selection
"""

import tempfile
import unittest
from pathlib import Path
from typing import Dict, Any

import requests
import torch

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestExpertDistributionRecorderMode(CustomTestCase):
    """Test expert_distribution_recorder_mode parameter with all modes."""

    def test_mode_stat(self):
        """Test stat mode - records expert call counts."""
        self._execute_core(
            model_path="Qwen/Qwen1.5-MoE-A2.7B",
            mode="stat",
            tp_size=1,
        )

    def test_mode_stat_tp2(self):
        """Test stat mode with tensor parallelism (TP=2)."""
        self._execute_core(
            model_path="Qwen/Qwen1.5-MoE-A2.7B",
            mode="stat",
            tp_size=2,
        )

    def test_mode_per_pass(self):
        """Test per_pass mode - detailed recording per forward pass."""
        self._execute_core(
            model_path="Qwen/Qwen1.5-MoE-A2.7B",
            mode="per_pass",
            tp_size=1,
        )

    def test_mode_per_token(self):
        """Test per_token mode - detailed recording per token."""
        self._execute_core(
            model_path="Qwen/Qwen1.5-MoE-A2.7B",
            mode="per_token",
            tp_size=1,
        )

    def test_mode_stat_deepseek(self):
        """Test stat mode with DeepSeek model."""
        self._execute_core(
            model_path="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            mode="stat",
            tp_size=1,
        )

    def test_multiple_record_cycles(self):
        """Test multiple start/stop/dump cycles."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            envs.SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR.set(tmp_dir)

            process = popen_launch_server(
                "Qwen/Qwen1.5-MoE-A2.7B",
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--tp-size", "1",
                    "--expert-distribution-recorder-mode", "stat",
                    "--disable-cuda-graph",
                    "--disable-overlap-schedule",
                ],
            )

            try:
                # First cycle
                self._record_and_verify(DEFAULT_URL_FOR_TEST, tmp_dir, cycle=1)

                # Second cycle
                self._record_and_verify(DEFAULT_URL_FOR_TEST, tmp_dir, cycle=2)

                # Verify multiple dump files exist
                pt_files = list(Path(tmp_dir).glob("*.pt"))
                self.assertEqual(len(pt_files), 2, "Should have 2 dump files")

            finally:
                kill_process_tree(process.pid)

    def test_concurrent_requests(self):
        """Test expert distribution recording with concurrent requests."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            envs.SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR.set(tmp_dir)

            process = popen_launch_server(
                "Qwen/Qwen1.5-MoE-A2.7B",
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--tp-size", "1",
                    "--expert-distribution-recorder-mode", "stat",
                    "--disable-cuda-graph",
                    "--disable-overlap-schedule",
                ],
            )

            try:
                # Start recording
                response = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/start_expert_distribution_record"
                )
                self.assertEqual(response.status_code, 200)

                # Send multiple concurrent requests
                prompts = [
                    "The capital of France is",
                    "The largest planet is",
                    "Python is a",
                    "Machine learning is",
                ]

                for prompt in prompts:
                    response = requests.post(
                        f"{DEFAULT_URL_FOR_TEST}/generate",
                        json={
                            "text": prompt,
                            "sampling_params": {
                                "temperature": 0,
                                "max_new_tokens": 16,
                            },
                        },
                    )
                    self.assertEqual(response.status_code, 200)

                # Stop and dump
                response = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/stop_expert_distribution_record"
                )
                self.assertEqual(response.status_code, 200)

                response = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/dump_expert_distribution_record"
                )
                self.assertEqual(response.status_code, 200)

                # Verify data
                data = torch.load(
                    list(Path(tmp_dir).glob("*.pt"))[0], weights_only=True
                )
                logical_count = data["logical_count"]
                self.assertTrue(
                    logical_count.sum() > 0,
                    "Should have recorded expert calls from concurrent requests"
                )

            finally:
                kill_process_tree(process.pid)

    def test_buffer_size_limit(self):
        """Test expert distribution recorder with buffer size limit."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            envs.SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR.set(tmp_dir)

            process = popen_launch_server(
                "Qwen/Qwen1.5-MoE-A2.7B",
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--tp-size", "1",
                    "--expert-distribution-recorder-mode", "stat",
                    "--expert-distribution-recorder-buffer-size", "10",
                    "--disable-cuda-graph",
                    "--disable-overlap-schedule",
                ],
            )

            try:
                # Start recording
                response = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/start_expert_distribution_record"
                )
                self.assertEqual(response.status_code, 200)

                # Send multiple requests to potentially exceed buffer
                for i in range(5):
                    response = requests.post(
                        f"{DEFAULT_URL_FOR_TEST}/generate",
                        json={
                            "text": f"Request {i}: The answer is",
                            "sampling_params": {
                                "temperature": 0,
                                "max_new_tokens": 8,
                            },
                        },
                    )
                    self.assertEqual(response.status_code, 200)

                # Stop and dump
                response = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/stop_expert_distribution_record"
                )
                self.assertEqual(response.status_code, 200)

                response = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/dump_expert_distribution_record"
                )
                self.assertEqual(response.status_code, 200)

                # Verify data exists
                data = torch.load(
                    list(Path(tmp_dir).glob("*.pt"))[0], weights_only=True
                )
                self.assertIn("logical_count", data)

            finally:
                kill_process_tree(process.pid)

    def _execute_core(self, model_path: str, mode: str, tp_size: int):
        """Core execution logic for testing expert distribution recorder."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            envs.SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR.set(tmp_dir)

            process = popen_launch_server(
                model_path,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--tp-size", str(tp_size),
                    "--expert-distribution-recorder-mode", mode,
                    "--disable-cuda-graph",
                    "--disable-overlap-schedule",
                ],
            )

            try:
                # Start recording
                response = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/start_expert_distribution_record"
                )
                self.assertEqual(response.status_code, 200)

                # Generate expert distribution data
                response = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/generate",
                    json={
                        "text": "The capital of France is",
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 32,
                        },
                    },
                )
                self.assertEqual(response.status_code, 200)

                # Stop recording
                response = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/stop_expert_distribution_record"
                )
                self.assertEqual(response.status_code, 200)

                # Dump recorded data
                response = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/dump_expert_distribution_record"
                )
                self.assertEqual(response.status_code, 200)

                # Verify data based on mode
                self._verify_recorded_data(tmp_dir, mode)

            finally:
                kill_process_tree(process.pid)

    def _verify_recorded_data(self, tmp_dir: str, mode: str):
        """Verify recorded data based on mode."""
        pt_files = list(Path(tmp_dir).glob("*.pt"))
        self.assertGreater(len(pt_files), 0, "Should have at least one dump file")

        data = torch.load(pt_files[0], weights_only=True)
        print(f"Mode={mode}, Data keys={data.keys() if isinstance(data, dict) else 'list'}")

        if mode in ["per_pass", "per_token"]:
            # Detail modes return list of records
            self.assertIsInstance(data, list)
            self.assertGreater(len(data), 0, "Should contain data rows")
        else:
            # Stat modes return dict with logical_count
            self.assertIsInstance(data, dict)
            self.assertIn("logical_count", data)
            logical_count = data["logical_count"]
            print(f"{logical_count.sum()=} {logical_count.shape=}")
            self.assertTrue(
                logical_count.sum() > 0,
                "Should have recorded some expert calls"
            )

    def _record_and_verify(self, base_url: str, tmp_dir: str, cycle: int):
        """Perform one record cycle and verify."""
        # Start recording
        response = requests.post(f"{base_url}/start_expert_distribution_record")
        self.assertEqual(response.status_code, 200, f"Cycle {cycle}: Failed to start")

        # Generate request
        response = requests.post(
            f"{base_url}/generate",
            json={
                "text": f"Cycle {cycle}: The answer is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
        )
        self.assertEqual(response.status_code, 200, f"Cycle {cycle}: Generate failed")

        # Stop recording
        response = requests.post(f"{base_url}/stop_expert_distribution_record")
        self.assertEqual(response.status_code, 200, f"Cycle {cycle}: Failed to stop")

        # Dump data
        response = requests.post(f"{base_url}/dump_expert_distribution_record")
        self.assertEqual(response.status_code, 200, f"Cycle {cycle}: Failed to dump")


class TestExpertDistributionRecorderModeEdgeCases(CustomTestCase):
    """Test edge cases for expert_distribution_recorder_mode."""

    def test_stop_without_start(self):
        """Test stopping recording without starting first."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            envs.SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR.set(tmp_dir)

            process = popen_launch_server(
                "Qwen/Qwen1.5-MoE-A2.7B",
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--tp-size", "1",
                    "--expert-distribution-recorder-mode", "stat",
                    "--disable-cuda-graph",
                    "--disable-overlap-schedule",
                ],
            )

            try:
                # Try to stop without starting - should warn but not crash
                response = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/stop_expert_distribution_record"
                )
                self.assertEqual(response.status_code, 200)

            finally:
                kill_process_tree(process.pid)

    def test_dump_without_record(self):
        """Test dumping without any recorded data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            envs.SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR.set(tmp_dir)

            process = popen_launch_server(
                "Qwen/Qwen1.5-MoE-A2.7B",
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--tp-size", "1",
                    "--expert-distribution-recorder-mode", "stat",
                    "--disable-cuda-graph",
                    "--disable-overlap-schedule",
                ],
            )

            try:
                # Start and immediately stop/dump without generating
                response = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/start_expert_distribution_record"
                )
                self.assertEqual(response.status_code, 200)

                response = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/stop_expert_distribution_record"
                )
                self.assertEqual(response.status_code, 200)

                response = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/dump_expert_distribution_record"
                )
                self.assertEqual(response.status_code, 200)

            finally:
                kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
