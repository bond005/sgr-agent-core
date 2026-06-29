#!/usr/bin/env bash
#
# generate_dataset.sh - Generate a distillation dataset (JSONL) from scenario agents.
#
# Prerequisites:
#   - Ubuntu 24.04 (or any modern Linux with bash)
#   - The .venv at the repo root (created by the project setup)
#   - Exported env vars:
#       ZAI_API_KEY      - GLM-5.2 (Z.ai) API key        (required)
#       TAVILY_API_KEY   - Tavily API key                 (required for search scenarios)
#
# Usage:
#   ./scripts/generate_dataset.sh                       # all scenarios, ~10 prompts each
#   ./scripts/generate_dataset.sh --scenario coder,creative_writer --limit 5
#   ./scripts/generate_dataset.sh --scenario list
#   ./scripts/generate_dataset.sh --enable-thinking     # capture raw reasoning_content
#
# Any extra args after the fixed ones are forwarded to generate_dataset.py.
# Pass --no-export to skip the final sgr-dataset conversion step.
#
# Teacher thinking (enable_thinking) is OFF by default: the distillation CoT is
# taken from SGR reasoning (cot_source=sgr_reasoning), which the agent produces
# regardless of the model's thinking mode. Pass --enable-thinking to switch the
# teacher to thinking mode and also record raw reasoning_content in llm_calls.jsonl
# (useful if you later export with cot_source=reasoning_content).

set -euo pipefail

# ---- locate repo root (this script lives in scripts/) -----------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

VENV="${REPO_ROOT}/.venv"
PYTHON="${VENV}/bin/python"
CONFIG="${SCRIPT_DIR}/dataset_gen.yaml.example"
SEEDS_DIR="${SCRIPT_DIR}/seeds"
DATASET_DIR="${SCRIPT_DIR}/dataset"
SHAREGPT_DIR="${SCRIPT_DIR}/dataset_sharegpt"

log() { echo "[generate_dataset] $*" >&2; }
die() { echo "[generate_dataset] ERROR: $*" >&2; exit 1; }

# ---- checks ----------------------------------------------------------------
if [[ ! -x "${PYTHON}" ]]; then
    die "Python interpreter not found at ${PYTHON}. Create the venv first (python -m venv .venv)."
fi
# Ensure pytest/pandas etc. are importable (best-effort, non-fatal if offline).
"${PYTHON}" -c "import yaml, openai" 2>/dev/null || die "Missing core deps in venv; run: pip install -e ."

if [[ -z "${ZAI_API_KEY:-}" ]]; then
    die "ZAI_API_KEY is not set. Export it first:  export ZAI_API_KEY=..."
fi
if [[ -z "${TAVILY_API_KEY:-}" ]]; then
    log "WARNING: TAVILY_API_KEY is not set; search scenarios (factual_qa, deep_research) will be skipped."
fi

# ---- run the driver --------------------------------------------------------
# The driver expands ${VAR} sentinels from env in memory; no disk substitution.
DEFAULT_ARGS=(--config "${CONFIG}" --scenario all --limit 10 --concurrency 4)

# Forward user args if any were passed; otherwise use defaults.
if [[ $# -gt 0 ]]; then
    log "Running driver with user args: $*"
    "${PYTHON}" "${SCRIPT_DIR}/generate_dataset.py" "$@"
else
    log "Running driver with default args: ${DEFAULT_ARGS[*]}"
    "${PYTHON}" "${SCRIPT_DIR}/generate_dataset.py" "${DEFAULT_ARGS[@]}"
fi

# ---- convert to sharegpt SFT format ---------------------------------------
if [[ " $* " == *" --no-export "* ]]; then
    log "Skipping export (--no-export)."
    exit 0
fi

log "Converting ${DATASET_DIR} -> ${SHAREGPT_DIR} (sharegpt, split by role)..."
"${PYTHON}" -m sgr_agent_core.cli.dataset_export \
    "${DATASET_DIR}" \
    -o "${SHAREGPT_DIR}" \
    --final-answer-as text \
    --val-ratio 0.05 \
    --split-by-role

# ---- summary ---------------------------------------------------------------
log "=== Done ==="
if command -v jq >/dev/null 2>&1 && [[ -f "${DATASET_DIR}/trajectories.jsonl" ]]; then
    log "Trajectories by role:"
    jq -r '.role' "${DATASET_DIR}/trajectories.jsonl" 2>/dev/null | sort | uniq -c | sort -rn >&2 || true
fi
log "Trajectories : ${DATASET_DIR}/trajectories.jsonl"
log "Raw calls    : ${DATASET_DIR}/llm_calls.jsonl"
log "Sharegpt out : ${SHAREGPT_DIR}/ (train_<role>.jsonl, val_<role>.jsonl)"
