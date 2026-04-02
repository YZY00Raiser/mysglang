import multiprocessing as mp
import unittest

import requests

from sglang.srt.utils import kill_process_tree
# from sglang.test.ascend.test_ascend_utils import QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)

QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH = "/home/weights/Qwen/Qwen3-VL-8B-Instruct"
MODEL = QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH

# image
IMAGE_MAN_IRONING_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/man_ironing_on_back_of_suv.png"


def popen_launch_server_wrapper(base_url, model, other_args):
    process = popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
    )
    return process


# def _send_parallel_request_task1(base_url, image_url):
#     import requests
#
#     requests.packages.urllib3.disable_warnings()
#     import ssl
#
#     ssl._create_default_https_context = ssl._create_unverified_context
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image_url", "image_url": {"url": image_url}},
#                 {"type": "text", "text": "Describe this image."},
#             ],
#         }
#     ]
#     resp = requests.post(
#         f"{base_url}/chat/completions",
#         json={"messages": messages, "temperature": 0, "max_completion_tokens": 512},
#     )
#     assert resp.status_code == 200


class TestLimitMMDatePerRequest(CustomTestCase):
    """Testcase: Configuring Multi-Modal to send different multimodal inference requests,
       each containing multiple multimodal input data.

    [Test Category] Parameter
    [Test Target] --mm-max-concurrent-calls; --mm-per-request-timeout; --enable-broadcast-mm-inputs-process; --limit-mm-data-per-request
    """

    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.base_url += "/v1"
        cls.api_key = "sk-123456"

        # limit_mm = '{"image":1, "video":1}'
        other_args = [
            "--mem-fraction-static",
            "0.5",
            "--enable-multimodal",
            "--mm-max-concurrent-calls",
            "1",
            "--mm-per-request-timeout",
            "1",
            "--enable-broadcast-mm-inputs-process",
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
            "--tp-size",
            "4",
            "--disable-cuda-graph",
            # "--limit-mm-data-per-request",
            # limit_mm,
            "--encoder-only",
            "--enable-prefix-mm-cache",
        ]
        # cls.process = popen_launch_server_wrapper(
        #     DEFAULT_URL_FOR_TEST, MODEL, other_args
        # )
        cls.process = popen_launch_server(
            MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_enable_prefix_mm(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": IMAGE_MAN_IRONING_URL},
                    },
                    {
                        "type": "text",
                        "text": "Describe this image in a sentence.",
                    },
                ],
            },
        ]
        response = requests.post(
            self.base_url + "/chat/completions",
            json={
                "messages": messages,
                "temperature": 0,
                "max_completion_tokens": 1024,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["meta_info"]["cached_tokens"], 0
        )

        response2 = requests.post(
            self.base_url + "/chat/completions",
            json={
                "messages": messages,
                "temperature": 0,
                "max_completion_tokens": 1024,
            },
        )
        self.assertEqual(response2.status_code, 200)
        self.assertGreater(
            response.json()["meta_info"]["cached_tokens"], 0
        )


if __name__ == "__main__":
    unittest.main()
