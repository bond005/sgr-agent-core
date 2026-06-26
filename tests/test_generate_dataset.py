"""Tests for the dataset-generation driver (orchestration, mocked agents)."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import yaml

from scripts.generate_dataset import (
    _inject_env_secrets,
    _search_key_present,
    discover_languages,
    discover_scenarios,
    expand_seeds,
    load_seeds,
    run_generation,
    run_one,
)
from sgr_agent_core.agent_config import GlobalConfig
from sgr_agent_core.models import AgentStatesEnum


def _write_seeds(root: Path, lang: str, scenario: str, seeds: list[str]) -> None:
    d = root / lang
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{scenario}.yaml").write_text(yaml.dump({"scenario": scenario, "seeds": seeds}), encoding="utf-8")


class TestLoadSeeds:
    def test_load_seeds_parses_yaml(self, tmp_path):
        _write_seeds(tmp_path, "en", "coder", ["Write a function", "Explain recursion", "  "])
        seeds = load_seeds(tmp_path, "coder", "en")
        assert seeds == ["Write a function", "Explain recursion"]

    def test_load_seeds_missing_file(self, tmp_path):
        assert load_seeds(tmp_path, "nope", "en") == []


class TestDiscovery:
    def test_discover_languages_finds_subdirs(self, tmp_path):
        _write_seeds(tmp_path, "en", "coder", ["a"])
        _write_seeds(tmp_path, "ru", "coder", ["a"])
        _write_seeds(tmp_path, "empty", "x", [])  # has a yaml -> counts as language
        assert discover_languages(tmp_path) == ["empty", "en", "ru"]

    def test_discover_languages_empty_dir_ignored(self, tmp_path):
        (tmp_path / "fr").mkdir()  # no yaml files
        _write_seeds(tmp_path, "en", "coder", ["a"])
        assert discover_languages(tmp_path) == ["en"]

    def test_discover_scenarios_union_across_languages(self, tmp_path):
        _write_seeds(tmp_path, "en", "coder", ["a"])
        _write_seeds(tmp_path, "en", "math", ["a"])
        _write_seeds(tmp_path, "ru", "coder", ["a"])
        assert discover_scenarios(tmp_path, ["en", "ru"]) == ["coder", "math"]


def _make_config_with_sentinels(tmp_path: Path, output_dir: str) -> Path:
    """Build a config that uses ${VAR} sentinels for api_key and tool kwargs."""
    cfg = {
        "llm": {
            "api_key": "${ZAI_API_KEY}",
            "base_url": "https://api.z.ai/api/paas/v4/",
            "model": "glm-5.2",
        },
        "execution": {"max_iterations": 3, "logs_dir": ""},
        "dataset": {"enabled": True, "output_dir": output_dir, "modes": ["trajectory"]},
        "mcp": {"mcpServers": {}},
        "agents": {
            "coder": {
                "base_class": "sgr_agent_core.agents.tool_calling_agent.ToolCallingAgent",
                "role": "coder",
                "llm": {"api_key": "${ZAI_API_KEY}"},
                "prompts": {"system_prompt_str": "You are a coder."},
                "tools": ["final_answer_tool"],
            }
        },
        "tools": {
            "web_search_tool": {
                "tool_class": "sgr_agent_core.tools.web_search_tool.WebSearchTool",
                "api_key": "${TAVILY_API_KEY}",
                "max_results": 5,
            }
        },
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    return path


class TestInjectEnvSecrets:
    def test_expands_global_llm_api_key(self, tmp_path, monkeypatch):
        GlobalConfig._instance = None
        GlobalConfig._initialized = False
        monkeypatch.setenv("ZAI_API_KEY", "real-zai-key-123")
        config = GlobalConfig.from_yaml(_make_config_with_sentinels(tmp_path, str(tmp_path / "ds")))
        assert config.llm.api_key == "${ZAI_API_KEY}"  # before inject
        _inject_env_secrets(config)
        assert config.llm.api_key == "real-zai-key-123"

    def test_expands_per_agent_llm_api_key(self, tmp_path, monkeypatch):
        GlobalConfig._instance = None
        GlobalConfig._initialized = False
        monkeypatch.setenv("ZAI_API_KEY", "agent-zai-key")
        config = GlobalConfig.from_yaml(_make_config_with_sentinels(tmp_path, str(tmp_path / "ds")))
        agent_def = config.agents["coder"]
        assert "${" in agent_def.llm.api_key  # before inject
        _inject_env_secrets(config)
        assert agent_def.llm.api_key == "agent-zai-key"

    def test_expands_tool_kwargs(self, tmp_path, monkeypatch):
        GlobalConfig._instance = None
        GlobalConfig._initialized = False
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-real-key")
        config = GlobalConfig.from_yaml(_make_config_with_sentinels(tmp_path, str(tmp_path / "ds")))
        tool_def = config.tools["web_search_tool"]
        assert "${" in tool_def.api_key  # before inject
        _inject_env_secrets(config)
        assert tool_def.api_key == "tvly-real-key"

    def test_unset_var_stays_as_sentinel(self, tmp_path, monkeypatch):
        GlobalConfig._instance = None
        GlobalConfig._initialized = False
        monkeypatch.delenv("ZAI_API_KEY", raising=False)
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        config = GlobalConfig.from_yaml(_make_config_with_sentinels(tmp_path, str(tmp_path / "ds")))
        _inject_env_secrets(config)
        assert config.llm.api_key == "${ZAI_API_KEY}"
        assert config.agents["coder"].llm.api_key == "${ZAI_API_KEY}"
        assert config.tools["web_search_tool"].api_key == "${TAVILY_API_KEY}"


class TestSearchKeyPresent:
    def test_present_when_expanded(self, tmp_path, monkeypatch):
        GlobalConfig._instance = None
        GlobalConfig._initialized = False
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-real")
        config = GlobalConfig.from_yaml(_make_config_with_sentinels(tmp_path, str(tmp_path / "ds")))
        _inject_env_secrets(config)
        assert _search_key_present(config) is True

    def test_absent_when_sentinel(self, tmp_path, monkeypatch):
        GlobalConfig._instance = None
        GlobalConfig._initialized = False
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        config = GlobalConfig.from_yaml(_make_config_with_sentinels(tmp_path, str(tmp_path / "ds")))
        _inject_env_secrets(config)
        assert _search_key_present(config) is False

    def test_absent_when_no_tool(self, tmp_path):
        GlobalConfig._instance = None
        GlobalConfig._initialized = False
        cfg = {
            "llm": {"api_key": "k", "base_url": "u", "model": "m"},
            "execution": {"max_iterations": 1, "logs_dir": ""},
            "dataset": {"enabled": False, "output_dir": str(tmp_path), "modes": []},
            "mcp": {"mcpServers": {}},
            "agents": {},
            "tools": {},
        }
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(cfg), encoding="utf-8")
        config = GlobalConfig.from_yaml(str(path))
        assert _search_key_present(config) is False


class TestExpandSeeds:
    @pytest.mark.asyncio
    async def test_expand_seeds_parses_lines(self):
        client = Mock()
        content = "1. First task\nSecond task\n- Third task\n"
        completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
        client.chat = SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=completion)))
        out = await expand_seeds(["seed"], 3, client, "glm-5.2")
        assert out == ["First task", "Second task", "Third task"]

    @pytest.mark.asyncio
    async def test_expand_seeds_zero_returns_empty(self):
        client = Mock()
        out = await expand_seeds(["seed"], 0, client, "glm-5.2")
        assert out == []

    @pytest.mark.asyncio
    async def test_expand_seeds_handles_api_error(self):
        client = Mock()
        client.chat = SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=RuntimeError("boom"))))
        out = await expand_seeds(["seed"], 2, client, "glm-5.2")
        assert out == []


class TestRunOne:
    @pytest.mark.asyncio
    async def test_run_one_success_sets_language(self, monkeypatch):
        fake_agent = SimpleNamespace(
            execute=AsyncMock(return_value="done"),
            _context=SimpleNamespace(state=AgentStatesEnum.COMPLETED),
        )
        monkeypatch.setattr("scripts.generate_dataset.AgentFactory.create", AsyncMock(return_value=fake_agent))
        monkeypatch.setattr(
            "scripts.generate_dataset.GlobalConfig",
            lambda: SimpleNamespace(agents={"coder": SimpleNamespace()}),
        )
        result = await run_one("coder", "write a function", asyncio.Semaphore(2), "ru")
        assert result == {"scenario": "coder", "language": "ru", "ok": True, "state": "completed"}
        assert fake_agent.language == "ru"  # driver must tag the agent

    @pytest.mark.asyncio
    async def test_run_one_failure_state_reports_not_ok(self, monkeypatch):
        # BaseAgent._execute swallows exceptions and sets FAILED; run_one must
        # judge success by state == COMPLETED, not by absence of exception.
        fake_agent = SimpleNamespace(
            execute=AsyncMock(return_value=None),
            _context=SimpleNamespace(state=AgentStatesEnum.FAILED),
        )
        monkeypatch.setattr("scripts.generate_dataset.AgentFactory.create", AsyncMock(return_value=fake_agent))
        monkeypatch.setattr(
            "scripts.generate_dataset.GlobalConfig",
            lambda: SimpleNamespace(agents={"coder": SimpleNamespace()}),
        )
        result = await run_one("coder", "write a function", asyncio.Semaphore(2), "en")
        assert result["ok"] is False
        assert result["state"] == "failed"
        assert result["language"] == "en"

    @pytest.mark.asyncio
    async def test_run_one_creation_failure(self, monkeypatch):
        monkeypatch.setattr("scripts.generate_dataset.AgentFactory.create", AsyncMock(side_effect=RuntimeError("nope")))
        monkeypatch.setattr(
            "scripts.generate_dataset.GlobalConfig",
            lambda: SimpleNamespace(agents={"coder": SimpleNamespace()}),
        )
        result = await run_one("coder", "write a function", asyncio.Semaphore(2), "en")
        assert result["ok"] is False
        assert result["language"] == "en"
        assert "nope" in result["error"]


def _minimal_config(tmp_path: Path, output_dir: str) -> Path:
    cfg = {
        "llm": {"api_key": "test-key", "base_url": "https://api.z.ai/api/paas/v4/", "model": "glm-5.2"},
        "execution": {"max_iterations": 3, "logs_dir": ""},
        "dataset": {"enabled": True, "output_dir": output_dir, "modes": ["trajectory"]},
        "mcp": {"mcpServers": {}},
        "agents": {
            "coder": {
                "base_class": "sgr_agent_core.agents.tool_calling_agent.ToolCallingAgent",
                "role": "coder",
                "prompts": {"system_prompt_str": "You are a coder."},
                "tools": ["final_answer_tool"],
            }
        },
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    return path


class TestRunGeneration:
    @pytest.mark.asyncio
    async def test_run_generation_summary_per_language(self, tmp_path, monkeypatch):
        from sgr_agent_core.agent_config import GlobalConfig

        GlobalConfig._instance = None
        GlobalConfig._initialized = False

        output_dir = tmp_path / "ds"
        config_path = _minimal_config(tmp_path, str(output_dir))

        # Hermetic seeds dir with two languages for the 'coder' scenario.
        seeds_dir = tmp_path / "seeds"
        _write_seeds(seeds_dir, "en", "coder", ["task A", "task B"])
        _write_seeds(seeds_dir, "ru", "coder", ["задача А"])

        fake_agent = SimpleNamespace(
            execute=AsyncMock(return_value="done"),
            _context=SimpleNamespace(state=AgentStatesEnum.COMPLETED),
        )
        monkeypatch.setattr("scripts.generate_dataset.AgentFactory.create", AsyncMock(return_value=fake_agent))
        monkeypatch.setattr("scripts.generate_dataset.AsyncOpenAI", Mock(return_value=Mock()))

        summary = await run_generation(
            str(config_path),
            ["coder"],
            languages=["en", "ru"],
            limit=0,
            concurrency=2,
            self_instruct=0,
            seeds_dir=seeds_dir,
        )

        assert summary["total"] == 3  # 2 en + 1 ru
        assert summary["per_scenario"][("coder", "en")]["ok"] == 2
        assert summary["per_scenario"][("coder", "ru")]["ok"] == 1
