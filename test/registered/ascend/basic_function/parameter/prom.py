def cache_report_openai(self, message):
    response = self.run_openai(message)
    print(
        f"openai first request cached_tokens: {int(response.usage.prompt_tokens_details.cached_tokens)}"
    )
    first_cached_tokens = int(response.usage.prompt_tokens_details.cached_tokens)
    # assert int(response.usage.cached_tokens) == 0
    assert first_cached_tokens <= self.min_cached
    response = self.run_openai(message)
    cached_tokens = int(response.usage.prompt_tokens_details.cached_tokens)
    print(f"openai second request cached_tokens: {cached_tokens}")
    assert cached_tokens > 0
    assert cached_tokens == int(response.usage.prompt_tokens) - 1
    return first_cached_tokens



