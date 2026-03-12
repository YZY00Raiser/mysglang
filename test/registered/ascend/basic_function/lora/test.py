class TestLoraWithSessionManagement(CustomTestCase):
    """Testcase：Verify LoRA adapter works properly with session management (multi-turn dialogue)

    [Test Category] LoRA + Session Management
    [Test Target] --enable-lora, session handling
    """
    lora_a = LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH
    lora_b = LLAMA_3_2_1B_INSTRUCT_TOOL_FAST_LORA_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--tp-size",
            "2",
            "--enable-lora",
            "--lora-path",
            f"lora_1={cls.lora_a},lora_2={cls.lora_b}",
            "--max-loaded-loras",
            "4",
            "--lora-eviction-policy",
            "fifo",
            "--lora-target-modules",
            "all",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--enable-mixed-chunk",
            "--max-total-tokens",
            "20480",
        ]
        cls.process = popen_launch_server(
            LLAMA_3_2_1B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_lora_with_session_multi_turn(self):
        """Test multi-turn dialogue with LoRA adapter using session management"""

        # 测试步骤1: 启动会话，发送第一轮对话，使用lora_b
        session_id_1 = "test-session-001"

        # 第一轮对话 - 介绍自己
        response1 = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "你好，我叫小明，很高兴认识你！",
                "sampling_params": {
                    "temperature": 0.7,
                    "max_new_tokens": 64,
                    "top_p": 0.9
                },
                "lora_path": self.lora_b,
                "session_params": {
                    "id": session_id_1,
                    "enable": True
                }
            },
        )

        # 验证返回结果
        self.assertEqual(response1.status_code, 200)
        response_text_1 = response1.json()["text"]
        self.assertIsNotNone(response_text_1)
        print(f"Session {session_id_1} - Turn 1 Response: {response_text_1}")

        # 测试步骤2: 使用相同会话ID，发送第二轮对话
        response2 = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "你还记得我叫什么名字吗？",
                "sampling_params": {
                    "temperature": 0.7,
                    "max_new_tokens": 64,
                    "top_p": 0.9
                },
                "lora_path": self.lora_b,
                "session_params": {
                    "id": session_id_1,
                    "enable": True
                }
            },
        )

        # 验证返回结果包含上下文信息（应该记得名字）
        self.assertEqual(response2.status_code, 200)
        response_text_2 = response2.json()["text"]
        self.assertIsNotNone(response_text_2)
        print(f"Session {session_id_1} - Turn 2 Response: {response_text_2}")

        # 验证模型记得用户名字（上下文保持）
        self.assertIn("小明", response_text_2.lower() or "ming" in response_text_2.lower(),
                      f"Expected response to contain user's name '小明', but got: {response_text_2}")

        # 测试步骤3: 关闭会话
        response_close = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "再见",
                "sampling_params": {
                    "temperature": 0.7,
                    "max_new_tokens": 32,
                },
                "lora_path": self.lora_b,
                "session_params": {
                    "id": session_id_1,
                    "enable": False  # 关闭会话
                }
            },
        )
        self.assertEqual(response_close.status_code, 200)

        # 测试步骤4: 使用新会话ID，发送对话，使用不同的LoRA
        session_id_2 = "test-session-002"

        response3 = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "你好，我叫小红，我们聊过吗？",
                "sampling_params": {
                    "temperature": 0.7,
                    "max_new_tokens": 64,
                    "top_p": 0.9
                },
                "lora_path": self.lora_a,  # 使用不同的LoRA
                "session_params": {
                    "id": session_id_2,
                    "enable": True
                }
            },
        )

        # 验证新会话返回结果
        self.assertEqual(response3.status_code, 200)
        response_text_3 = response3.json()["text"]
        self.assertIsNotNone(response_text_3)
        print(f"Session {session_id_2} - Turn 1 Response: {response_text_3}")

        # 验证新会话对立于旧会话（不应该记得旧会话的信息）
        self.assertNotIn("小明", response_text_3.lower(),
                         f"New session should not remember old session's user '小明', but got: {response_text_3}")

    def test_concurrent_sessions_with_different_loras(self):
        """Test multiple concurrent sessions with different LoRA adapters"""

        # 创建两个不同会话，使用不同的LoRA
        sessions = [
            {"id": "session-A", "lora": self.lora_a, "user": "张三"},
            {"id": "session-B", "lora": self.lora_b, "user": "李四"}
        ]

        responses = {}

        # 第一轮对话 - 两个会话同时进行
        for session in sessions:
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": f"你好，我叫{session['user']}",
                    "sampling_params": {
                        "temperature": 0.7,
                        "max_new_tokens": 64,
                    },
                    "lora_path": session["lora"],
                    "session_params": {
                        "id": session["id"],
                        "enable": True
                    }
                },
            )
            self.assertEqual(response.status_code, 200)
            responses[session["id"]] = response.json()["text"]
            print(f"Session {session['id']} - Turn 1: {responses[session['id']]}")

        # 第二轮对话 - 两个会话分别询问名字
        for session in sessions:
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "我叫什么名字？",
                    "sampling_params": {
                        "temperature": 0.7,
                        "max_new_tokens": 64,
                    },
                    "lora_path": session["lora"],
                    "session_params": {
                        "id": session["id"],
                        "enable": True
                    }
                },
            )
            self.assertEqual(response.status_code, 200)
            response_text = response.json()["text"]

            # 验证每个会话都记得自己的用户名字
            self.assertIn(session["user"], response_text,
                          f"Session {session['id']} should remember user {session['user']}, but got: {response_text}")
            print(f"Session {session['id']} - Turn 2: {response_text}")

    def test_session_reset(self):
        """Test session reset functionality"""
        session_id = "test-session-reset"

        # 第一轮对话 - 建立上下文
        response1 = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "我的宠物是一只猫，叫咪咪",
                "sampling_params": {
                    "temperature": 0.7,
                    "max_new_tokens": 64,
                },
                "lora_path": self.lora_b,
                "session_params": {
                    "id": session_id,
                    "enable": True
                }
            },
        )
        self.assertEqual(response1.status_code, 200)

        # 第二轮对话 - 验证上下文
        response2 = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "我的宠物叫什么名字？",
                "sampling_params": {
                    "temperature": 0.7,
                    "max_new_tokens": 32,
                },
                "lora_path": self.lora_b,
                "session_params": {
                    "id": session_id,
                    "enable": True
                }
            },
        )
        self.assertEqual(response2.status_code, 200)
        response_text_2 = response2.json()["text"]
        self.assertIn("咪咪", response_text_2,
                      f"Session should remember pet name '咪咪', but got: {response_text_2}")

        # 重置会话（关闭后重新开启）
        response_reset = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "重置会话",
                "sampling_params": {
                    "temperature": 0.7,
                    "max_new_tokens": 32,
                },
                "lora_path": self.lora_b,
                "session_params": {
                    "id": session_id,
                    "enable": False  # 关闭会话
                }
            },
        )
        self.assertEqual(response_reset.status_code, 200)

        # 使用相同ID开启新会话
        response3 = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "我的宠物叫什么名字？",
                "sampling_params": {
                    "temperature": 0.7,
                    "max_new_tokens": 32,
                },
                "lora_path": self.lora_b,
                "session_params": {
                    "id": session_id,
                    "enable": True  # 重新开启
                }
            },
        )
        self.assertEqual(response3.status_code, 200)
        response_text_3 = response3.json()["text"]

        # 验证新会话不记得之前的上下文
        self.assertNotIn("咪咪", response_text_3,
                         f"New session should not remember old context, but got: {response_text_3}")
