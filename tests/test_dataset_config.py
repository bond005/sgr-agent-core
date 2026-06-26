"""Tests for dataset recording configuration (DatasetRecordingConfig, role)."""

from sgr_agent_core.agent_definition import AgentConfig, DatasetRecordingConfig, LLMConfig, PromptsConfig


class TestDatasetRecordingConfig:
    """Tests for DatasetRecordingConfig model and its wiring into AgentConfig."""

    def test_dataset_section_defaults(self):
        """AgentConfig exposes a dataset section with sensible defaults."""
        cfg = AgentConfig(
            llm=LLMConfig(api_key="k"),
            prompts=PromptsConfig(system_prompt_str="p", initial_user_request_str="p", clarification_response_str="p"),
        )
        assert isinstance(cfg.dataset, DatasetRecordingConfig)
        assert cfg.dataset.enabled is False
        assert cfg.dataset.output_dir == "dataset"
        assert cfg.dataset.modes == ["raw", "trajectory"]
        assert cfg.dataset.include_reasoning is True
        assert cfg.dataset.cot_source == "sgr_reasoning"
        assert cfg.dataset.teacher_model is None

    def test_dataset_section_custom(self):
        """Dataset section accepts custom values incl. extra fields (extra=allow)."""
        cfg = AgentConfig(
            llm=LLMConfig(api_key="k"),
            prompts=PromptsConfig(system_prompt_str="p", initial_user_request_str="p", clarification_response_str="p"),
            dataset=DatasetRecordingConfig(
                enabled=True,
                output_dir="/tmp/ds",
                modes=["trajectory"],
                include_reasoning=False,
                cot_source="reasoning_content",
                teacher_model="glm-5.2",
            ),
        )
        assert cfg.dataset.enabled is True
        assert cfg.dataset.output_dir == "/tmp/ds"
        assert cfg.dataset.modes == ["trajectory"]
        assert cfg.dataset.include_reasoning is False
        assert cfg.dataset.cot_source == "reasoning_content"
        assert cfg.dataset.teacher_model == "glm-5.2"

    def test_role_field_default_none(self):
        """role defaults to None on AgentConfig."""
        cfg = AgentConfig(
            llm=LLMConfig(api_key="k"),
            prompts=PromptsConfig(system_prompt_str="p", initial_user_request_str="p", clarification_response_str="p"),
        )
        assert cfg.role is None

    def test_role_field_custom(self):
        """role can be set on AgentConfig."""
        cfg = AgentConfig(
            llm=LLMConfig(api_key="k"),
            prompts=PromptsConfig(system_prompt_str="p", initial_user_request_str="p", clarification_response_str="p"),
            role="pediatric_advisor",
        )
        assert cfg.role == "pediatric_advisor"

    def test_dataset_extra_fields_allowed(self):
        """DatasetRecordingConfig keeps unknown fields (extra=allow)."""
        cfg = DatasetRecordingConfig(enabled=True, my_custom_flag=42)  # type: ignore[call-arg]
        assert getattr(cfg, "my_custom_flag") == 42
