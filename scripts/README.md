# Dataset generation (distillation)

End-to-end pipeline that synthesizes a distillation dataset (JSONL, sharegpt
format) from a teacher LLM (**GLM-5.2**) running as SGR agents across diverse
role scenarios.

## Layout

```
scripts/
  generate_dataset.sh            # bash entry point (Ubuntu/Linux)
  generate_dataset.py            # Python driver (concurrency, self-instruct, languages)
  dataset_gen.yaml.example       # config: teacher LLM + dataset + 9 scenario agents
  seeds/<lang>/<scenario>.yaml   # curated seed prompts, one subdir per language (en, ru)
  tools/pandas_analysis_tool.py  # custom safe tabular-analysis tool (data_analyst)
  data/*.csv                     # sample datasets for the data_analyst scenario
```

## Languages (bilingual by default)

Languages are **auto-discovered** from subdirectories of `seeds/`. By default
there are two — `en` and `ru` — so every scenario is generated in both English
and Russian (a parallel corpus: the same task in two languages). Each trajectory
is tagged with a `language` field.

- To add a language: create `seeds/<lang>/` with `<scenario>.yaml` files — the
  driver picks it up automatically (e.g. `seeds/de/` → German added).
- Override at the CLI: `--languages en` or `--languages en,ru,de`.
- Every scenario's system prompt instructs the agent to answer in the user's
  language, so the response language matches the seed language.

## Scenarios (role -> behavior)

| Scenario | Tools | Needs web |
|----------|-------|-----------|
| `factual_qa` | search, extract | yes |
| `deep_research` | plan, search, extract, report | yes (multihop) |
| `summarizer` | extract, report | no (URLs in prompt) |
| `creative_writer` | final answer | no |
| `data_analyst` | PandasAnalysisTool | no |
| `coder` | final answer | no |
| `math_reasoner` | final answer | no |
| `instruction_following` | final answer | no |
| `rewriter` | final answer | no |

## Prerequisites

1. Project venv at `.venv/` (see repo root `AGENTS.md`).
2. `pandas` installed (for `data_analyst`): `uv pip install --python .venv/bin/python pandas`
3. API keys exported (never stored in config files — the driver expands `${VAR}` sentinels in memory):
   ```bash
   export ZAI_API_KEY=...        # required (GLM-5.2 via Z.ai)
   export TAVILY_API_KEY=...     # required for search scenarios only
   ```

## Usage

```bash
# All scenarios, ~10 prompts each, all discovered languages (en, ru by default)
./scripts/generate_dataset.sh

# Specific scenarios + single language + more prompts + teacher expansion
./scripts/generate_dataset.sh --scenario coder,math_reasoner --languages en --limit 50 --self-instruct 20

# List available scenarios and languages
./scripts/generate_dataset.sh --scenario list

# Skip the final sharegpt conversion
./scripts/generate_dataset.sh --no-export
```

The driver forwards any args to `generate_dataset.py` (see `--help`):
`--config`, `--scenario`, `--languages`, `--limit`, `--concurrency`,
`--self-instruct`, `--seeds-dir`, `-v`.

## Output

- `scripts/dataset/llm_calls.jsonl` — raw request/response per LLM call (Level A).
- `scripts/dataset/trajectories.jsonl` — one sharegpt-style record per agent run (Level B), tagged with `role` and `language`.
- `scripts/dataset_sharegpt/train_<role>.jsonl` / `val_<role>.jsonl` — SFT-ready splits (one per role), produced by `sgr-dataset ... --split-by-role`.

The sharegpt format (`{"messages": [...], "tools": [...]}` with roles
`system`/`user`/`assistant`/`tool` and `tool_calls`) is directly consumable by
**MS-SWIFT** and **LLaMA-Factory** for fine-tuning Qwen2.5.

## Convert manually

```bash
python -m sgr_agent_core.cli.dataset_export scripts/dataset -o out --final-answer-as text --val-ratio 0.05
python -m sgr_agent_core.cli.dataset_export scripts/dataset -o out --role coder creative_writer   # filter by role
python -m sgr_agent_core.cli.dataset_export scripts/dataset -o out --language ru                  # filter by language
python -m sgr_agent_core.cli.dataset_export scripts/dataset -o out --split-by-role                # per-role files
```

## How it works

- The config template (`dataset_gen.yaml.example`) uses `${VAR}` sentinels
  (e.g. `${ZAI_API_KEY}`) instead of real keys. The driver expands these in
  memory via `_inject_env_secrets()` — no real key is ever written to disk.
- The driver auto-discovers languages from `seeds/<lang>/` subdirectories and
  scenarios from the YAML files inside. It loads seeds per (scenario, language),
  optionally expands them via the teacher (`--self-instruct`), and fans out agent
  runs with a bounded semaphore. Each agent is tagged with its `language`, and
  every run records via the shared `DatasetRecorder` singleton
  (see `AGENTS.md` -> "Dataset recording").
- The streaming generator queue is unbounded, so agents complete without anyone
  consuming the SSE stream.
