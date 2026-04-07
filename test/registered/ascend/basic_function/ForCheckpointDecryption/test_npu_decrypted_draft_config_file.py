import os
import subprocess
import unittest
from urllib.parse import urlparse

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import run_command
# from sglang.test.ascend.test_ascend_utils import (
#     QWEN3_32B_WEIGHTS_PATH,
#     QWEN3_32B_EAGLE3_WEIGHTS_PATH
# )
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server, _wait_for_server_health, _create_clean_subprocess_env,
)

register_npu_ci(
    est_time=400,
    suite="nightly-4-npu-a3",
    nightly=True,
)
QWEN3_32B_WEIGHTS_PATH = "/home/weights/Qwen/Qwen3-8B"
QWEN3_32B_EAGLE3_WEIGHTS_PATH = "/home/weights/Qwen/Qwen3-8B_eagle3"


class TestSetForwardHooks(CustomTestCase):
    """Testcase: Verify set --decrypted-config-file, --decrypted-draft-config-file parameter,
    will use the specified config.json and the inference request is successfully processed.

    [Test Category] Parameter
    [Test Target] --decrypted-config-file, --decrypted-draft-config-file
    """

    model = QWEN3_32B_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        # run_command(
        #     f"mv {os.path.join(QWEN3_32B_WEIGHTS_PATH, 'config.json')} {os.path.join(QWEN3_32B_WEIGHTS_PATH, '_config.json')}")
        # run_command(
        #     f"mv {os.path.join(QWEN3_32B_EAGLE3_WEIGHTS_PATH, 'config.json')} {os.path.join(QWEN3_32B_EAGLE3_WEIGHTS_PATH, '_config.json')}")

        '''
        cls.extra_envs = {
            "HCCL_BUFFSIZE": "1024",
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
            "SGLANG_NPU_USE_MLAPO": "1",
            "SGLANG_NPU_USE_EINSUM_MM": "1",
            "SLANG_ENABLE_SPEC_V2": "1",
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
        }
        os.environ.update(cls.extra_envs)
        '''

        '''
        # Service failed to start, restoring original file name
        try:
            cls.process = popen_launch_server(
                cls.model,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--attention-backend",
                    "ascend",
                    "--disable-radix-cache",
                    "--chunked-prefill-size",
                    "-1",
                    "--max-prefill-tokens",
                    "1024",
                    "--speculative-algorithm",
                    "EAGLE3",
                    "--speculative-draft-model-path",
                    QWEN3_32B_EAGLE3_WEIGHTS_PATH,
                    "--speculative-num-steps",
                    "3",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "4",
                    "--tp-size",
                    "2",
                    "--mem-fraction-static",
                    "0.68",
                    "--disable-cuda-graph",
                    "--dtype",
                    "bfloat16",
                    "--decrypted-config-file",
                    "Qwen3-8B/config.json",
                    "--decrypted-draft-config-file",
                    "Qwen3-8B_eagle3/config.json",
                ],
                env={
                    "SLANG_ENABLE_SPEC_V2": "1",
                    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
                },
            )
        except Exception as e:
            raise RuntimeError(f"Failed to launch server: {e}") from e
        finally:
            for weights_path in [QWEN3_32B_WEIGHTS_PATH, QWEN3_32B_EAGLE3_WEIGHTS_PATH]:
                old_path = os.path.join(weights_path, '_config.json')
                new_path = os.path.join(weights_path, 'config.json')

                if os.path.exists(old_path):
                    try:
                        run_command(f"mv {old_path} {new_path}")
                    except Exception as e:
                        print(f"Warning: Failed to rename {old_path}: {e}")
                elif not os.path.exists(new_path):
                    print(f"Warning: Neither {old_path} nor {new_path} exists")
            if cls.process:
                kill_process_tree(cls.process.pid)

        '''

        try:
            # launch server with "--config" parameter
            parsed_url = urlparse(DEFAULT_URL_FOR_TEST)
            host = parsed_url.hostname
            port = str(parsed_url.port)
            command = [
                "python3",
                "-m",
                "sglang.launch_server",
                "--model-path",
                QWEN3_32B_WEIGHTS_PATH,
                "--host",
                host,
                "--port",
                port,
                "--trust-remote-code",
                "--attention-backend",
                "ascend",
                "--disable-radix-cache",
                "--chunked-prefill-size",
                "-1",
                "--max-prefill-tokens",
                "1024",
                "--speculative-algorithm",
                "EAGLE3",
                "--speculative-draft-model-path",
                QWEN3_32B_EAGLE3_WEIGHTS_PATH,
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
                "--tp-size",
                "2",
                "--mem-fraction-static",
                "0.68",
                "--disable-cuda-graph",
                "--dtype",
                "bfloat16",
                # "--decrypted-config-file",
                # "Qwen3-8B/config.json",
                # "--decrypted-draft-config-file",
                # "Qwen3-8B_eagle3/config.json",
            ]

            env = _create_clean_subprocess_env(os.environ.copy())
            cls.extra_envs = {
                "SLANG_ENABLE_SPEC_V2": "1",
                "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
            }

            os.environ.update(cls.extra_envs)
            cls.process = subprocess.Popen(command, stdout=None, stderr=None, env=env)
            _wait_for_server_health(
                cls.process, DEFAULT_URL_FOR_TEST, None, DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
            )

        except Exception as e:
            raise RuntimeError(f"Failed to launch server: {e}") from e

        finally:
            for weights_path in [QWEN3_32B_WEIGHTS_PATH, QWEN3_32B_EAGLE3_WEIGHTS_PATH]:
                old_path = os.path.join(weights_path, '_config.json')
                new_path = os.path.join(weights_path, 'config.json')

                if os.path.exists(old_path):
                    try:
                        run_command(f"mv {old_path} {new_path}")
                    except Exception as e:
                        print(f"Warning: Failed to rename {old_path}: {e}")
                elif not os.path.exists(new_path):
                    print(f"Warning: Neither {old_path} nor {new_path} exists")
            if cls.process:
                kill_process_tree(cls.process.pid)

    @classmethod
    def tearDownClass(cls):
        run_command(
            f"mv {os.path.join(QWEN3_32B_WEIGHTS_PATH, '_config.json')} {os.path.join(QWEN3_32B_WEIGHTS_PATH, 'config.json')}")
        run_command(
            f"mv {os.path.join(QWEN3_32B_EAGLE3_WEIGHTS_PATH, '_config.json')} {os.path.join(QWEN3_32B_EAGLE3_WEIGHTS_PATH, 'config.json')}")
        kill_process_tree(cls.process.pid)

    def test_decrypted_draft_config_file(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0.8,
                    "max_new_tokens": 32,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)


if __name__ == "__main__":
    unittest.main()
