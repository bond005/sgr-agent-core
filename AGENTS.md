# AGENTS.md

Compact guide for OpenCode sessions working in this repo. Read before editing.

## Project

**SGR Agent Core** — Python framework (>=3.10) for Schema-Guided Reasoning agents. Published to PyPI as `sgr-agent-core`. Provides a library + an OpenAI-compatible FastAPI server + a Vue frontend.

## Repo layout (what is real vs. noise)

- `sgr_agent_core/` — the framework. This is what you edit.
- `sgr-agent-frontend/` — separate Vue 3 + TypeScript + Vite app with its **own** toolchain (`npm`, eslint, prettier). Do not mix its conventions with the Python side.
- `examples/` — example agents and configs. The tracked research agents live in `examples/sgr_deep_research/`.
- `tests/` — pytest suite (flat, no subpackages).
- **Ignore these — untracked local artifacts, not source:** root `sgr_deep_research/`, `sgr_deep_research.egg-info/`, `uv.lock`. The project uses pip/venv, not uv.
- `config.yaml`, `agents.yaml`, `.venv/` are gitignored. Edit the `*.example` templates; copy to the real name to run.

## Commands

Always activate the venv first: `source .venv/bin/activate`

```bash
pytest                                            # run tests (e2e auto-excluded)
pytest tests/test_foo.py::TestClass::test_bar -v  # single test
pytest -m e2e                                     # run e2e tests (excluded by default)
pytest --cov=sgr_agent_core --cov-report=term-missing   # with coverage
make format                                       # lint+format = pre-commit run --all-files
make wheel                                        # build wheel; `make` builds sdist
sgr -c config.yaml                                # run API server (or: python -m sgr_agent_core.server --config-file config.yaml)
sgrsh                                             # interactive CLI client
sgr-dataset <recorded_dir> -o <out>               # convert recorded trajectories to a sharegpt SFT dataset
```

CI runs lint (`make format` on Python 3.13) then installs from sdist and runs `coverage run -m pytest`. Lint must pass before publish.

## Required workflow (enforced by team rules)

For **any** new feature or bug fix, follow this order strictly — do not skip steps:

1. Plan → 2. write tests, confirm they **fail (red)** → 3. implement → 4. confirm new tests **pass (green)** → 5. run **full** test suite → 6. update docs → 7. run linter → 8. report.

Implement **one class at a time, bottom-up** through the architecture layers (base classes → config/registry → factory/services → agents → tools → server).

## Conventions that differ from defaults

- **Comments and docstrings: strictly English.** This does not apply to user-facing responses.
- **User-facing responses: reply in the language the user is writing in** (Russian → Russian, Chinese → Chinese, English → English, etc.), unless the user explicitly asks for a different language.
- Type hints mandatory. Use `T | None` (not `Optional[T]`), `dict[str, Any]` (not `Dict[...]`).
- Google-style docstrings. Line length 120. Ruff for lint+format.
- Async-first. Use `httpx`, never `requests`. No `print` in service code — use `logging`.
- Error handling: guard clauses / early returns; specific exceptions, not bare `Exception`.

## Architecture essentials (not obvious from filenames)

- **Registry pattern with auto-registration:** any subclass of `BaseAgent` / `BaseTool` is auto-registered in `AgentRegistry` / `ToolRegistry` via `__init_subclass__` mixins. Agents resolve by class name or `name`; tools by `tool_name`.
- **Agent execution cycle** (in `BaseAgent._execution_step()`): three phases per iteration — Reasoning → Select Action → Action — until a finish state.
- **Three agent types** differ in how they reason/select tools:
  - `SGRAgent` — structured output (`response_format`)
  - `ToolCallingAgent` — native function calling, no explicit reasoning
  - `SGRToolCallingAgent` — hybrid (SGR reasoning + function-calling tool selection); best default
- Tools are **Pydantic models** (`BaseTool` subclass) implementing `async __call__(context, config, **kwargs) -> str`.
- Config hierarchy: `GlobalConfig` (singleton) → `AgentDefinition` → `AgentConfig`. `extra="allow"` everywhere.
- **Search settings (`tavily_api_key`, `max_results`, ...) are per-tool under `tools:`, NOT in `AgentConfig`.**
- **Dataset recording (distillation):** enable via the `dataset:` config section (on `AgentConfig`, overridable per agent). All LLM calls funnel through `BaseAgent._llm_call(phase, **openai_kwargs)` (the single capture point — keep new agents using it). Two JSONL granularities are written: `llm_calls.jsonl` (raw request/response per call) and `trajectories.jsonl` (one sharegpt-style record per agent run). A shared `DatasetRecorder` (module singleton in `services/dataset_recorder.py`) is created lazily; the `role` field tags records by agent role. Convert with `sgr-dataset`. Teacher = any OpenAI-compatible model (GLM-5.2, GPT-4o); set `enable_thinking: false` under `llm:` for reliable `SGRAgent` structured output.

## Configuration

- Two YAML files: `config.yaml` (global llm/execution/prompts/mcp/tools) and an optional `agents.yaml` (agent definitions, loaded via `GlobalConfig.definitions_from_yaml`).
- Env var override prefix is **`SGR__`** with double-underscore nesting, e.g. `SGR__LLM__API_KEY`, `SGR__EXECUTION__MAX_STEPS` (see `.env.example`).
- `base_class` in agent definitions accepts a dotted import string (`sgr_agent_core.agents.sgr_agent.SGRAgent`) or a registry name.
- **Provider-specific LLM params:** any field under `llm:` beyond the declared ones (`model`, `max_tokens`, `temperature`, ...) is kept via `extra="allow"` and routed to the OpenAI SDK through `extra_body` by `LLMConfig.to_openai_client_kwargs()`. Use this for options like GLM's `enable_thinking`/`reasoning_effort` or Qwen's `chat_template_kwargs`. Do NOT pass them as top-level keys — the SDK rejects unknown kwargs.

## Testing notes

- `asyncio_mode = "auto"` (pytest-asyncio). `@pytest.mark.asyncio` is still used in the suite.
- Markers: `unit`, `integration`, `slow`, `e2e`. **e2e is excluded by default** (`-m "not e2e"` in `pytest.ini`).
- Use the `create_test_agent()` helper and fixtures (`mock_openai_client`, `test_llm_config`, ...) in `tests/conftest.py`. Always mock `AsyncOpenAI` and external APIs (Tavily, MCP).

## Deeper reference

Detailed, authoritative rules live in `.cursor/rules/*.mdc` (architecture, core-modules, code-style, testing, workflow, implementation-order, python-fastapi). Consult them when extending the framework.
