"""Tests for LLMConfig.to_openai_client_kwargs() extra_body routing."""

from sgr_agent_core.agent_definition import LLMConfig


class TestToOpenaiClientKwargs:
    def test_declared_fields_are_top_level(self):
        cfg = LLMConfig(api_key="k", base_url="u", model="glm-5.2", max_tokens=1234, temperature=0.7)
        kwargs = cfg.to_openai_client_kwargs()
        assert kwargs["model"] == "glm-5.2"
        assert kwargs["max_tokens"] == 1234
        assert kwargs["temperature"] == 0.7

    def test_secrets_and_transport_excluded(self):
        cfg = LLMConfig(api_key="secret", base_url="https://example", proxy="socks5://127.0.0.1:1080")
        kwargs = cfg.to_openai_client_kwargs()
        assert "api_key" not in kwargs
        assert "base_url" not in kwargs
        assert "proxy" not in kwargs

    def test_no_extras_means_no_extra_body(self):
        cfg = LLMConfig(api_key="k", model="gpt-4o-mini")
        kwargs = cfg.to_openai_client_kwargs()
        assert "extra_body" not in kwargs

    def test_extra_fields_routed_into_extra_body(self):
        cfg = LLMConfig(
            api_key="k",
            model="glm-5.2",
            enable_thinking=True,
            reasoning_effort="high",
        )
        kwargs = cfg.to_openai_client_kwargs()
        # extras must NOT leak as top-level kwargs (the SDK would reject them)
        assert "enable_thinking" not in kwargs
        assert "reasoning_effort" not in kwargs
        assert kwargs["extra_body"] == {"enable_thinking": True, "reasoning_effort": "high"}
        # declared fields still top-level
        assert kwargs["model"] == "glm-5.2"

    def test_standard_sdk_param_in_extra_body_still_accepted_by_sdk_signature(self):
        """A standard param like top_p, when configured as extra, goes to extra_body."""
        import inspect

        from openai.resources.chat.completions import AsyncCompletions

        cfg = LLMConfig(api_key="k", model="m", top_p=0.9, seed=42)
        kwargs = cfg.to_openai_client_kwargs()
        assert kwargs["extra_body"] == {"top_p": 0.9, "seed": 42}
        # The SDK stream() accepts extra_body as a parameter
        assert "extra_body" in inspect.signature(AsyncCompletions.stream).parameters
