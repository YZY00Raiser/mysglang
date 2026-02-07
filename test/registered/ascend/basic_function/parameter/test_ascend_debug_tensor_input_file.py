import os
import unittest
import numpy
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestDebugTensorInputFile(CustomTestCase):
    """Testcase：Verify set --debug-tensor-dump-input-file parameter, the -debug-tensor-dump-input-file is taking effect.

       [Test Category] Parameter
       [Test Target] --debug-tensor-dump-input-file
       """
    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        vector = numpy.array([1001, 1002, 1003, 1004, 1005, 1006, 1007])
        numpy.save("./input_tensor.npy", vector)
        other_args = [
            "--debug-tensor-dump-input-file",
            "./input_tensor.npy",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]
        cls.out_log_file = open("./tensor_input_out_log.txt", "w+", encoding="utf-8")
        cls.err_log_file = open("./tensor_input_err_log.txt", "w+", encoding="utf-8")
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.out_log_file.close()
        cls.err_log_file.close()
        os.remove("./tensor_input_out_log.txt")
        os.remove("./tensor_input_err_log.txt")
        os.remove("./input_tensor.npy")

    def test_tensor_input_file(self):
        print("-----------------out==0------------")
        print(self.out_log_file)
        print("-----------------err==0------------")
        print(self.err_log_file)
        self.err_log_file.seek(0)
        content = self.err_log_file.read()
        self.assertTrue(len(content) > 0)


if __name__ == "__main__":
    unittest.main()
