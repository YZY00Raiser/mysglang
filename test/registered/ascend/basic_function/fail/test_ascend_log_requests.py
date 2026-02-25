import os
import re
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


def get_lines_with_keyword(filename, keyword):
    results = []

    try:
        with open(filename, "r", encoding="utf-8") as file:
            for line_num, line in enumerate(file, 1):
                if re.match(keyword, line):
                    results.append(
                        {
                            "line_number": line_num,
                            "content": line.strip(),
                        }
                    )
        return results
    except Exception as e:
        print(f"error:{e}")
        return []


class TestLogRequests(CustomTestCase):
    def test_log_requests(self):
        other_args = (
            [
                "--log-requests",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ]
        )
        out_log_file = open("./log_requests_out_log.txt", "w+", encoding="utf-8")
        err_log_file = open("./log_requests_err_log.txt", "w+", encoding="utf-8")
        process = popen_launch_server(
            (
                "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct"
            ),
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(out_log_file, err_log_file),
        )

        try:
            response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
            self.assertEqual(response.status_code, 200)

            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertIn("Paris", response.text)
            err_log_file.seek(0)
            content = err_log_file.read()
            print(content)
            self.assertTrue(len(content) > 0)
            pattern = r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, text='The capital of France is'.*"
            self.assertIsNotNone(re.search(pattern, content))
        finally:
            kill_process_tree(process.pid)
            out_log_file.close()
            err_log_file.close()
            os.remove("./log_requests_out_log.txt")
            os.remove("./log_requests_err_log.txt")

    def test_log_requests_level(self):
        message = {
                "0": r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, video_data=None,.*",
                "1": r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, video_data=None, sampling_params=.*",
                "2": r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, text=.*",
                "3": r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, text=.*",
        }
        out_log_name = "./log_requests_level_out_log.txt"
        err_log_name = "./log_requests_level_err_log.txt"
        keyword_Finish = r".*Finish: obj=GenerateReqInput\(.*http_worker_ipc=None, text='just.*"
        keyword_start = "out={'text': '"
        keyword_end = "', 'output_ids'"
        for i in [0, 1, 2, 3]:
            other_args = (
                [
                    "--log-requests-level",
                    i,
                    "--log-requests",
                    "--attention-backend",
                    "ascend",
                    "--disable-cuda-graph",
                ]
            )
            out_log_file = open(out_log_name, "w+", encoding="utf-8")
            err_log_file = open(err_log_name, "w+", encoding="utf-8")
            process = popen_launch_server(
                (
                    "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct"
                ),
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=other_args,
                return_stdout_stderr=(out_log_file, err_log_file),
            )

            try:
                response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
                self.assertEqual(response.status_code, 200)

                response = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/generate",
                    json={
                        "text": "just return me a string with of 5000 characters",
                        "sampling_params": {"temperature": 0, "max_new_tokens": 10000},
                    },
                )
                self.assertEqual(response.status_code, 200)
                err_log_file.seek(0)
                content = err_log_file.read()
                print(i)
                print(content)
                self.assertTrue(len(content) > 0)
                self.assertIsNotNone(re.search(message[str(i)], content))
                if i >= 2:
                    lines = get_lines_with_keyword(err_log_name, keyword_Finish)
                    Finish_message = lines[0]["content"]
                    start_index = Finish_message.find(keyword_start) + len(
                        keyword_start
                    )
                    end_index = Finish_message.find(keyword_end)
                    out_text = Finish_message[start_index:end_index]
                    out_text_length = len(out_text)
                    print("out_text_length is ", out_text_length)
                    print("out text is ", out_text)
                    out_text_length_n = len(out_text.replace("\\n", " "))
                    if i == 2:
                        print(i)
                        print(out_text)
                        self.assertIn("' ... '", out_text)
                        self.assertTrue(out_text_length_n - len("' ... '") == 2048)
                        print("out_text_length_n is ", out_text_length_n)
                    else:
                        print(i)
                        print(out_text)
                        self.assertNotIn("' ... '", out_text)
                        self.assertTrue(out_text_length > 2048)

            finally:
                kill_process_tree(process.pid)
                out_log_file.close()
                err_log_file.close()
                os.remove(out_log_name)
                os.remove(err_log_name)


if __name__ == "__main__":
    unittest.main()
