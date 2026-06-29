"""Batch driver that generates a distillation dataset by running scenario agents.

Loads ``GlobalConfig`` from a config file, reads curated seed prompts per
scenario, optionally expands them via self-instruct (the teacher generates more
diverse prompts), then fans out agent runs with bounded concurrency. Each run
records a trajectory (Level B) and raw LLM calls (Level A) via the shared
:class:`DatasetRecorder` singleton.

Languages are auto-discovered from subdirectories of ``--seeds-dir``
(``seeds/<lang>/<scenario>.yaml``); by default ``en`` and ``ru``. A trajectory
is generated per (scenario, language, seed), tagged with ``language``.

Usage::

    python scripts/generate_dataset.py --config scripts/dataset_gen.yaml \
        --scenario all --languages auto --limit 10 --concurrency 4

Run ``--scenario list`` to print available scenarios and languages.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncOpenAI

# Location of this script (default config / seeds paths). NOTE: we intentionally
# do NOT add SCRIPTS_DIR to sys.path here: GlobalConfig.from_yaml() already
# inserts the config directory (which contains scripts/) so the custom tool
# import string resolves during config load, and running this file directly puts
# scripts/ on sys.path[0] automatically. Mutating sys.path at import time would
# pollute the test session.
SCRIPTS_DIR = Path(__file__).resolve().parent

from sgr_agent_core.agent_config import GlobalConfig  # noqa: E402
from sgr_agent_core.agent_factory import AgentFactory  # noqa: E402
from sgr_agent_core.models import AgentStatesEnum  # noqa: E402
from sgr_agent_core.services.dataset_recorder import reset_recorder  # noqa: E402

logger = logging.getLogger("generate_dataset")

# Default fallback scenario order when discovery finds nothing. The actual
# scenario set is discovered from seeds/<lang>/*.yaml at runtime.
DEFAULT_SCENARIOS = [
    "factual_qa",
    "deep_research",
    "summarizer",
    "creative_writer",
    "data_analyst",
    "coder",
    "math_reasoner",
    "instruction_following",
    "rewriter",
]

# Scenarios that require a web-search API key (Tavily). They are skipped if the
# key is missing (or the placeholder is still present).
SEARCH_SCENARIOS = {"factual_qa", "deep_research"}


def discover_languages(seeds_root: Path) -> list[str]:
    """Return the sorted list of language subdirectories containing seed YAMLs."""
    if not seeds_root.exists():
        return []
    return sorted(d.name for d in seeds_root.iterdir() if d.is_dir() and any(d.glob("*.yaml")))


def discover_scenarios(seeds_root: Path, languages: list[str]) -> list[str]:
    """Return the union of scenario stems found across the given languages."""
    stems: set[str] = set()
    for lang in languages:
        lang_dir = seeds_root / lang
        if lang_dir.is_dir():
            stems.update(f.stem for f in lang_dir.glob("*.yaml"))
    return sorted(stems)


def load_seeds(seeds_root: Path, scenario: str, lang: str) -> list[str]:
    """Load the curated seed prompts for (scenario, lang) from its YAML file."""
    path = seeds_root / lang / f"{scenario}.yaml"
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    seeds = data.get("seeds", [])
    return [str(s).strip() for s in seeds if str(s).strip()]


async def expand_seeds(seeds: list[str], n: int, client: AsyncOpenAI, model: str) -> list[str]:
    """Use the teacher LLM to generate ``n`` diverse new prompts in the same style.

    Returns the generated prompts (one per line), best-effort: on any error or
    non-positive ``n`` an empty list is returned.
    """
    if n <= 0 or not seeds:
        return []
    examples = "\n".join(f"- {s}" for s in seeds[:6])
    meta_prompt = (
        "You generate diverse, high-quality task prompts for an AI assistant with a specific role.\n"
        f"Role/scenario: infer it from the examples below.\n"
        f"Examples:\n{examples}\n\n"
        f"Generate {n} NEW, diverse, specific task prompts in the same domain and style. "
        "Each must be self-contained and answerable. Output exactly one prompt per line, "
        "with no numbering, no bullet markers, and no surrounding quotes.\n"
        "Prompts:"
    )
    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": meta_prompt}],
            temperature=0.9,
            max_tokens=2000,
        )
    except Exception as e:  # noqa: BLE001 - self-instruct is best-effort
        logger.warning(f"self-instruct generation failed: {e}")
        return []
    text = completion.choices[0].message.content or ""
    generated = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        # strip a leading bullet or "N. " / "N) " numbering
        line = re.sub(r"^[-*]\s+", "", line)
        num = re.match(r"^\d+[.)]\s+(.*)$", line)
        if num:
            line = num.group(1)
        if line and len(line) > 5 and line not in generated:
            generated.append(line)
        if len(generated) >= n:
            break
    return generated[:n]


def _expand_value(value: Any) -> Any:
    """Expand ``${VAR}`` (and ``$VAR``) env references in a string, in place safe."""
    return os.path.expandvars(value) if isinstance(value, str) else value


def _inject_env_secrets(config: GlobalConfig) -> None:
    """Expand ``${VAR}`` env sentinels in the loaded config, in memory only.

    Secrets (LLM api_key, tool api_key/tavily_api_key, ...) are referenced as
    ``${VAR}`` in the config template and resolved from the environment here, so
    no real key is ever stored on disk. The global ``llm``, every agent's ``llm``,
    the global tool kwargs, and every agent's per-tool kwargs are expanded.
    """
    from sgr_agent_core.agent_definition import LLMConfig

    # Global LLM config
    for name in list(LLMConfig.model_fields):
        setattr(config.llm, name, _expand_value(getattr(config.llm, name)))
    # Per-agent LLM configs (each agent holds its own merged copy)
    for agent_def in config.agents.values():
        for name in list(LLMConfig.model_fields):
            setattr(agent_def.llm, name, _expand_value(getattr(agent_def.llm, name)))
    # Global tool kwargs (extra="allow" fields)
    for tool_def in config.tools.values():
        for key in list(tool_def.__pydantic_extra__ or {}):
            setattr(tool_def, key, _expand_value(getattr(tool_def, key)))
    # Per-agent tool kwargs. agent_level_tools_validator copies the global tool
    # kwargs (including unresolved ${VAR} sentinels) into each agent's
    # ToolDefinition at config-load time; those copies are NOT reached by the
    # global loop above and must be expanded here, otherwise the literal sentinel
    # is sent to the provider (e.g. "${TAVILY_API_KEY}" -> Tavily 401).
    for agent_def in config.agents.values():
        for tool_def in agent_def.tools:
            for key in list(tool_def.__pydantic_extra__ or {}):
                setattr(tool_def, key, _expand_value(getattr(tool_def, key)))


def _apply_thinking_override(config: GlobalConfig, enabled: bool) -> None:
    """Force the teacher ``enable_thinking`` flag on the global llm and on
    every per-agent ``llm``.

    ``enable_thinking`` is a provider option kept under ``llm:`` via
    ``extra="allow"`` and routed to the SDK through ``extra_body``. It is an
    inference-time toggle (like Qwen's thinking/instruct switch): it only
    changes whether the model emits a ``reasoning_content`` trace, and does
    not affect the agent cycle. Because the current distillation CoT comes
    from SGR reasoning (``cot_source="sgr_reasoning"``), thinking is off by
    default; callers opt in via ``--enable-thinking`` to also capture raw
    ``reasoning_content`` for later ``cot_source`` experiments.

    Each ``AgentDefinition`` holds its own merged copy of the llm config
    (same shape as the secrets handled by ``_inject_env_secrets``), so the
    override must be written to every per-agent copy too.
    """
    setattr(config.llm, "enable_thinking", enabled)
    for agent_def in config.agents.values():
        setattr(agent_def.llm, "enable_thinking", enabled)


def _search_key_present(config: GlobalConfig) -> bool:
    """Return True if the Tavily API key is configured (resolved, not the sentinel)."""
    ws = config.tools.get("web_search_tool")
    key = (ws.tool_kwargs().get("api_key") if ws else "") or ""
    # After _inject_env_secrets: a resolved key no longer contains "${"; an unset
    # env leaves the literal "${TAVILY_API_KEY}" sentinel, treated as missing.
    return bool(key) and "${" not in key


async def run_one(scenario: str, prompt: str, semaphore: asyncio.Semaphore, language: str) -> dict[str, Any]:
    """Create and execute one agent for a single prompt; return a status dict."""
    async with semaphore:
        agent_def = GlobalConfig().agents[scenario]
        task_messages = [{"role": "user", "content": prompt}]
        try:
            agent = await AgentFactory.create(agent_def, task_messages)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{scenario}/{language}] agent creation failed: {e}")
            return {"scenario": scenario, "language": language, "ok": False, "error": str(e)}
        agent.language = language
        try:
            await agent.execute()
            state = agent._context.state
            # BaseAgent._execute swallows exceptions and sets FAILED/ERROR/CANCELLED,
            # so success must be judged by state, not by absence of an exception.
            ok = state == AgentStatesEnum.COMPLETED
            return {"scenario": scenario, "language": language, "ok": ok, "state": str(state.value)}
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{scenario}/{language}] agent execution failed: {e}")
            return {"scenario": scenario, "language": language, "ok": False, "error": str(e)}


async def run_generation(
    config_path: str,
    scenarios: list[str],
    *,
    languages: list[str],
    limit: int,
    concurrency: int,
    self_instruct: int,
    seeds_dir: Path,
    enable_thinking: bool = False,
) -> dict[str, Any]:
    """Run the full generation pipeline and return a per-(scenario, language) summary."""
    reset_recorder()
    config = GlobalConfig.from_yaml(config_path)
    _inject_env_secrets(config)
    _apply_thinking_override(config, enable_thinking)
    if enable_thinking:
        logger.info("Teacher enable_thinking=ON (capturing raw reasoning_content)")
    else:
        logger.info("Teacher enable_thinking=OFF (default)")

    if not config.llm.api_key or "${" in config.llm.api_key:
        logger.error("LLM api_key is not resolved. Export the teacher API key, e.g.: export ZAI_API_KEY=...")
        return {"total": 0, "per_scenario": {}, "trajectories": 0}

    has_search = _search_key_present(config)
    client = AsyncOpenAI(base_url=config.llm.base_url, api_key=config.llm.api_key)

    semaphore = asyncio.Semaphore(concurrency)
    tasks: list[asyncio.Task] = []

    for scenario in scenarios:
        if scenario not in config.agents:
            logger.warning(f"scenario '{scenario}' has no agent definition; skipping")
            continue
        if scenario in SEARCH_SCENARIOS and not has_search:
            logger.warning(f"scenario '{scenario}' needs a web-search key; skipping (set TAVILY_API_KEY)")
            continue
        for lang in languages:
            seeds = load_seeds(seeds_dir, scenario, lang)
            if not seeds:
                continue
            if self_instruct > 0:
                expanded = await expand_seeds(seeds, self_instruct, client, config.llm.model)
                seeds = seeds + expanded
                logger.info(f"[{scenario}/{lang}] {len(expanded)} self-instructed prompts added")
            if limit > 0:
                seeds = seeds[:limit]
            logger.info(f"[{scenario}/{lang}] queued {len(seeds)} prompts")
            for prompt in seeds:
                tasks.append(asyncio.create_task(run_one(scenario, prompt, semaphore, lang)))

    if not tasks:
        logger.warning("Nothing to generate (no scenarios/seeds matched).")
        return {"total": 0, "per_scenario": {}, "trajectories": 0}

    results = await asyncio.gather(*tasks)

    per_key: dict[tuple[str, str], dict[str, int]] = {}
    for r in results:
        key = (r["scenario"], r["language"])
        bucket = per_key.setdefault(key, {"ok": 0, "fail": 0})
        bucket["ok" if r["ok"] else "fail"] += 1

    traj_count = _count_lines(Path(config.dataset.output_dir) / "trajectories.jsonl")
    return {"total": len(results), "per_scenario": per_key, "trajectories": traj_count}


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a distillation dataset from scenario agents.")
    parser.add_argument(
        "--config",
        default=str(SCRIPTS_DIR / "dataset_gen.yaml.example"),
        help="Path to dataset_gen config (secrets come from env, not this file)",
    )
    parser.add_argument(
        "--scenario",
        default="all",
        help="Scenario name, 'all', or 'list' to print available scenarios",
    )
    parser.add_argument(
        "--languages",
        default="auto",
        help="Comma-separated language codes (e.g. 'en,ru') or 'auto' to discover from --seeds-dir",
    )
    parser.add_argument("--limit", type=int, default=10, help="Max prompts per scenario and language (0 = no limit)")
    parser.add_argument("--concurrency", type=int, default=4, help="Max concurrent agent runs")
    parser.add_argument(
        "--self-instruct", type=int, default=0, help="Extra prompts generated by the teacher per scenario"
    )
    parser.add_argument(
        "--seeds-dir",
        default=str(SCRIPTS_DIR / "seeds"),
        help="Root seeds directory containing <lang>/<scenario>.yaml files",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable teacher thinking mode (enable_thinking=true) to capture raw "
        "reasoning_content. Default: disabled (the CoT comes from SGR reasoning).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    seeds_dir = Path(args.seeds_dir)
    languages = (
        discover_languages(seeds_dir)
        if args.languages == "auto"
        else [lng.strip() for lng in args.languages.split(",") if lng.strip()]
    )
    if not languages:
        logger.error("No languages found. Pass --languages en,ru or create seeds/<lang>/ directories.")
        return 1

    discovered = discover_scenarios(seeds_dir, languages) or DEFAULT_SCENARIOS

    if args.scenario == "list":
        print(f"Languages: {', '.join(languages)}")
        print("Scenarios: " + ", ".join(discovered))
        return 0

    scenarios = discovered if args.scenario == "all" else [s.strip() for s in args.scenario.split(",")]

    summary = asyncio.run(
        run_generation(
            args.config,
            scenarios,
            languages=languages,
            limit=args.limit,
            concurrency=args.concurrency,
            self_instruct=args.self_instruct,
            seeds_dir=seeds_dir,
            enable_thinking=args.enable_thinking,
        )
    )

    logger.info("=== Generation summary ===")
    logger.info(f"Languages: {', '.join(languages)}")
    logger.info(f"Total runs: {summary['total']} | Trajectories written: {summary['trajectories']}")
    for (scenario, lang), counts in sorted(summary["per_scenario"].items()):
        logger.info(f"  {scenario} [{lang}]: ok={counts['ok']} fail={counts['fail']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
