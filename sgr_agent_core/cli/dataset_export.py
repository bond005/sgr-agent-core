"""Export recorded dataset JSONL into SFT-ready formats.

Reads ``trajectories.jsonl`` produced by :class:`~sgr_agent_core.services.dataset_recorder.DatasetRecorder`
and emits a sharegpt-style JSONL consumable by MS-SWIFT / LLaMA-Factory for
fine-tuning models such as Qwen2.5.

Each output record is::

    {"messages": [{"role": "system"|"user"|"assistant"|"tool", ...}, ...],
     "tools":   [{"type": "function", "function": {...}}, ...]}

Optional transformations:
    - ``final_answer_as="text"`` collapses the trailing ``finalanswertool`` tool
      round-trip into a single plain-text assistant answer.
    - ``cot_source`` rebuilds the chain-of-thought from ``reasoning_content`` of
      the matching raw ``llm_calls`` records (requires the calls file).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

FINISHING_TOOLS_DEFAULT = ("finalanswertool",)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of records (empty if the file is absent)."""
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def collapse_final_answer(
    messages: list[dict[str, Any]],
    finishing_tools: Iterable[str] = FINISHING_TOOLS_DEFAULT,
) -> list[dict[str, Any]]:
    """Collapse a trailing finishing-tool round-trip into a text assistant turn.

    If the last assistant message calls a finishing tool and is followed by its
    ``tool`` result, both are replaced by one assistant message whose content is
    the final answer (read from the tool result, falling back to the tool call
    ``answer`` argument).
    """
    if not messages:
        return messages
    finishing = {f.lower() for f in finishing_tools}
    result = list(messages)
    # The trailing pair is [assistant(tool_calls), tool(result)].
    if len(result) >= 2 and result[-2].get("role") == "assistant" and result[-1].get("role") == "tool":
        tool_calls = result[-2].get("tool_calls") or []
        if len(tool_calls) == 1 and tool_calls[0].get("function", {}).get("name", "").lower() in finishing:
            answer = result[-1].get("content") or ""
            if not answer:
                try:
                    args = json.loads(tool_calls[0]["function"].get("arguments") or "{}")
                    answer = args.get("answer", "")
                except (TypeError, ValueError):
                    answer = ""
            result = result[:-2] + [{"role": "assistant", "content": answer}]
    return result


def build_reasoning_index(calls: list[dict[str, Any]]) -> dict[tuple[str, int], str]:
    """Index raw reasoning-content by (agent_id, iteration) from raw call records."""
    index: dict[tuple[str, int], str] = {}
    for call in calls:
        if call.get("phase") != "reasoning":
            continue
        rc = (call.get("response") or {}).get("reasoning_content")
        if rc:
            index[(call.get("agent_id"), int(call.get("iteration", -1)))] = rc
    return index


def substitute_cot_from_raw(
    trajectories: list[dict[str, Any]],
    calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replace SGR reasoning text blocks with the teacher's reasoning_content.

    Each assistant message whose content starts with the SGR CoT marker
    ("Reasoning steps:") is replaced by the matching raw ``reasoning_content``
    when one is available for the same agent.
    """
    index = build_reasoning_index(calls)
    if not index:
        return trajectories
    for traj in trajectories:
        agent_id = traj.get("agent_id")
        for message in traj.get("messages", []):
            content = message.get("content")
            is_cot = isinstance(content, str) and content.startswith("Reasoning steps:")
            if message.get("role") == "assistant" and is_cot:
                # Pick the most recent matching reasoning_content for this agent.
                matches = [rc for (aid, _it), rc in index.items() if aid == agent_id]
                if matches:
                    message["content"] = matches[-1]
    return trajectories


def to_sharegpt(
    trajectories: list[dict[str, Any]],
    *,
    final_answer_as: str = "tool",
    finishing_tools: Iterable[str] = FINISHING_TOOLS_DEFAULT,
) -> list[dict[str, Any]]:
    """Convert trajectory records into sharegpt SFT records.

    Args:
        trajectories: Trajectory records (Level B).
        final_answer_as: ``"tool"`` keeps the finishing-tool round-trip faithful;
            ``"text"`` collapses it into a plain assistant answer.
        finishing_tools: Tool names treated as finishing tools (lower-cased).

    Returns:
        A list of ``{"messages": [...], "tools": [...]}`` records.
    """
    out: list[dict[str, Any]] = []
    for traj in trajectories:
        messages = traj.get("messages", [])
        if final_answer_as == "text":
            messages = collapse_final_answer(messages, finishing_tools)
        record = {"messages": messages}
        if traj.get("tools"):
            record["tools"] = traj["tools"]
        out.append(record)
    return out


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def split_train_val(
    records: list[dict[str, Any]], val_ratio: float, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Shuffle and split records into (train, val) by ``val_ratio``."""
    if val_ratio <= 0:
        return records, []
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) > 1 else 0
    return shuffled[n_val:], shuffled[:n_val]


def convert(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    final_answer_as: str = "tool",
    cot_source: str = "sgr_reasoning",
    val_ratio: float = 0.0,
    seed: int = 42,
    finishing_tools: Iterable[str] = FINISHING_TOOLS_DEFAULT,
    roles: Iterable[str] | None = None,
    languages: Iterable[str] | None = None,
    split_by_role: bool = False,
) -> dict[str, Any]:
    """Convert recorded JSONL into a sharegpt dataset.

    Args:
        input_dir: Directory containing ``trajectories.jsonl`` (and optionally
            ``llm_calls.jsonl`` when ``cot_source="reasoning_content"``).
        output_dir: Directory where ``train.jsonl`` (and ``val.jsonl``) are written.
        final_answer_as: See :func:`to_sharegpt`.
        cot_source: ``"reasoning_content"`` substitutes the teacher's reasoning
            content from raw calls; otherwise SGR reasoning text is kept.
        val_ratio: Fraction held out for validation (0 disables the split).
        seed: RNG seed for the split.
        finishing_tools: See :func:`to_sharegpt`.
        roles: Optional allow-list of role tags; trajectories of other roles are
            ignored. When ``None``, all roles are included.
        languages: Optional allow-list of language tags; trajectories in other
            languages are ignored. When ``None``, all languages are included.
        split_by_role: When True, write one ``train_<role>.jsonl`` /
            ``val_<role>.jsonl`` pair per role (returns per-role counts) instead of
            a single ``train.jsonl``/``val.jsonl`` pair.

    Returns:
        ``{"train": int, "val": int}``, or ``{role: {"train": int, "val": int}}``
        when ``split_by_role`` is True.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    trajectories = [r for r in _read_jsonl(input_dir / "trajectories.jsonl") if r.get("record_type") == "trajectory"]
    if not trajectories:
        logger.warning(f"No trajectory records found in {input_dir / 'trajectories.jsonl'}")
    if cot_source == "reasoning_content":
        calls = _read_jsonl(input_dir / "llm_calls.jsonl")
        trajectories = substitute_cot_from_raw(trajectories, calls)

    role_set = {r for r in roles} if roles is not None else None
    if role_set is not None:
        trajectories = [t for t in trajectories if t.get("role") in role_set]

    lang_set = {lng for lng in languages} if languages is not None else None
    if lang_set is not None:
        trajectories = [t for t in trajectories if t.get("language") in lang_set]

    if split_by_role:
        return _convert_split_by_role(trajectories, output_dir, final_answer_as, val_ratio, seed, finishing_tools)

    train, val = _convert_and_split(trajectories, final_answer_as, val_ratio, seed, finishing_tools)
    write_jsonl(train, output_dir / "train.jsonl")
    if val:
        write_jsonl(val, output_dir / "val.jsonl")
    return {"train": len(train), "val": len(val)}


def _convert_and_split(
    trajectories: list[dict[str, Any]],
    final_answer_as: str,
    val_ratio: float,
    seed: int,
    finishing_tools: Iterable[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = to_sharegpt(trajectories, final_answer_as=final_answer_as, finishing_tools=finishing_tools)
    return split_train_val(records, val_ratio, seed)


def _convert_split_by_role(
    trajectories: list[dict[str, Any]],
    output_dir: Path,
    final_answer_as: str,
    val_ratio: float,
    seed: int,
    finishing_tools: Iterable[str],
) -> dict[str, dict[str, int]]:
    by_role: dict[str, list[dict[str, Any]]] = {}
    for traj in trajectories:
        by_role.setdefault(traj.get("role") or "unknown", []).append(traj)
    counts: dict[str, dict[str, int]] = {}
    for role, group in sorted(by_role.items()):
        train, val = _convert_and_split(group, final_answer_as, val_ratio, seed, finishing_tools)
        write_jsonl(train, output_dir / f"train_{role}.jsonl")
        if val:
            write_jsonl(val, output_dir / f"val_{role}.jsonl")
        counts[role] = {"train": len(train), "val": len(val)}
    return counts


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``sgr-dataset convert``."""
    parser = argparse.ArgumentParser(
        prog="sgr-dataset",
        description="Convert recorded SGR agent trajectories into a sharegpt SFT dataset.",
    )
    parser.add_argument("input", help="Directory containing trajectories.jsonl (and llm_calls.jsonl)")
    parser.add_argument("-o", "--output", default="dataset_sharegpt", help="Output directory for train/val JSONL")
    parser.add_argument(
        "--final-answer-as",
        choices=["tool", "text"],
        default="tool",
        help="Keep the finishing tool call ('tool') or collapse it into a text answer ('text')",
    )
    parser.add_argument(
        "--cot-source",
        choices=["sgr_reasoning", "reasoning_content", "merged"],
        default="sgr_reasoning",
        help="Source of the chain-of-thought (reasoning_content reads raw llm_calls.jsonl)",
    )
    parser.add_argument("--val-ratio", type=float, default=0.0, help="Validation split fraction (0 disables)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for the train/val split")
    parser.add_argument(
        "--finishing-tools",
        nargs="*",
        default=list(FINISHING_TOOLS_DEFAULT),
        help="Tool names treated as finishing tools for --final-answer-as text",
    )
    parser.add_argument(
        "--role",
        nargs="*",
        default=None,
        help="Only include trajectories with these role tags (space-separated)",
    )
    parser.add_argument(
        "--language",
        nargs="*",
        default=None,
        help="Only include trajectories in these languages (space-separated, e.g. --language en ru)",
    )
    parser.add_argument(
        "--split-by-role",
        action="store_true",
        help="Write one train_<role>.jsonl / val_<role>.jsonl pair per role",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.cot_source == "merged":
        args.cot_source = "reasoning_content"  # merged currently behaves like reasoning_content substitution
    counts = convert(
        args.input,
        args.output,
        final_answer_as=args.final_answer_as,
        cot_source=args.cot_source,
        val_ratio=args.val_ratio,
        seed=args.seed,
        finishing_tools=args.finishing_tools,
        roles=args.role,
        languages=args.language,
        split_by_role=args.split_by_role,
    )
    logger.info(f"Wrote {counts} to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
