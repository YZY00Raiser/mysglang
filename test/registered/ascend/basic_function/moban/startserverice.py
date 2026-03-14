'''
        process = popen_launch_server(
                LLAMA_3_2_1B_WEIGHTS_PATH,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=other_args,
            )

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "lora_path": "lora_a",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)

        if process:
            kill_process_tree(process.pid)
        '''
