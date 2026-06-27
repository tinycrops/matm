#!/usr/bin/env python3
"""Extract a Claude JSONL conversation into MATM-style trajectory artifacts.

The output deliberately treats Claude logs as action-observation traces:
assistant text and tool calls are actions; tool results and user feedback are
observations. Hidden thinking/signature fields are ignored.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


TEXT_LIMIT = 1400
VALUE_TEXT_LIMIT = 2600


def clean_text(text: str, limit: int = TEXT_LIMIT) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 18].rstrip() + " ...[truncated]"


def content_text(content: Any, limit: int = TEXT_LIMIT) -> str:
    if isinstance(content, str):
        return clean_text(content, limit)
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                pieces.append(item.get("text", ""))
            elif item.get("type") == "tool_result":
                pieces.append(item.get("content", ""))
        return clean_text("\n".join(pieces), limit)
    return ""


def iter_content_items(record: dict[str, Any]) -> list[dict[str, Any]]:
    content = record.get("message", {}).get("content")
    if isinstance(content, list):
        return [item for item in content if isinstance(item, dict)]
    return []


def summarize_tool_input(tool: dict[str, Any]) -> tuple[str, str, str]:
    name = tool.get("name", "unknown")
    payload = tool.get("input") or {}
    description = payload.get("description") or ""
    command = payload.get("command") or payload.get("cmd") or ""
    if isinstance(command, dict):
        command = json.dumps(command, sort_keys=True)
    if not command:
        compact: dict[str, Any] = {}
        for key, value in payload.items():
            if key in {"content", "old_string", "new_string"}:
                compact[key] = f"<{len(str(value))} chars>"
            else:
                compact[key] = value
        command = json.dumps(compact, sort_keys=True)
    return str(name), clean_text(str(description), 300), clean_text(str(command), 900)


@dataclass
class Step:
    step_index: int
    timestamp: str
    event_type: str
    role: str
    model: str | None
    cwd: str
    uuid: str
    parent_uuid: str | None
    action: str
    observation: str
    tool_use_id: str | None = None
    tool_name: str | None = None
    outcome: str | None = None


@dataclass
class ActionObservation:
    ao_index: int
    source_step_index: int
    timestamp: str
    model: str | None
    cwd: str
    action: str
    observation: str
    outcome: str
    tool_use_id: str | None
    tool_name: str | None
    recent_context: str


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def extract_steps(records: list[dict[str, Any]]) -> list[Step]:
    pending_tools: dict[str, int] = {}
    steps: list[Step] = []

    def add_step(record: dict[str, Any], event_type: str, role: str, action: str, observation: str,
                 tool_use_id: str | None = None, tool_name: str | None = None, outcome: str | None = None) -> None:
        message = record.get("message") or {}
        steps.append(
            Step(
                step_index=len(steps),
                timestamp=record.get("timestamp", ""),
                event_type=event_type,
                role=role,
                model=message.get("model"),
                cwd=record.get("cwd", ""),
                uuid=record.get("uuid", ""),
                parent_uuid=record.get("parentUuid"),
                action=action,
                observation=observation,
                tool_use_id=tool_use_id,
                tool_name=tool_name,
                outcome=outcome,
            )
        )

    for record in records:
        rtype = record.get("type")
        message = record.get("message", {})
        role = message.get("role", rtype or "")

        if rtype == "assistant":
            for item in iter_content_items(record):
                item_type = item.get("type")
                if item_type == "thinking":
                    continue
                if item_type == "text":
                    add_step(record, "assistant_text", "assistant", content_text([item]), "")
                elif item_type == "tool_use":
                    name, description, command = summarize_tool_input(item)
                    action = f"{name}: {description}".strip()
                    observation = command
                    tool_id = item.get("id")
                    add_step(record, "tool_call", "assistant", action, observation, tool_id, name)
                    if tool_id:
                        pending_tools[tool_id] = len(steps) - 1

        elif rtype == "user":
            items = iter_content_items(record)
            if items and any(item.get("type") == "tool_result" for item in items):
                for item in items:
                    if item.get("type") != "tool_result":
                        continue
                    tool_id = item.get("tool_use_id")
                    text = clean_text(str(item.get("content", "")), 1800)
                    is_error = bool(item.get("is_error"))
                    outcome = "error" if is_error else "ok"
                    add_step(record, "tool_result", "environment", "", text, tool_id, outcome=outcome)
            else:
                text = content_text(message.get("content"), 1800)
                if not text:
                    continue
                if "<local-command-caveat>" in text or "<local-command-stdout>" in text:
                    event = "session_control"
                elif "<command-name>" in text:
                    event = "user_command"
                else:
                    event = "user_feedback"
                add_step(record, event, "user", text, "")

    return steps


def step_brief(step: Step) -> str:
    if step.event_type == "tool_call":
        return f"[{step.step_index}] CALL {step.tool_name}: {step.action} :: {step.observation}"
    if step.event_type == "tool_result":
        return f"[{step.step_index}] RESULT {step.outcome}: {step.observation}"
    return f"[{step.step_index}] {step.role.upper()} {step.event_type}: {step.action or step.observation}"


def make_chunks(steps: list[Step], session_id: str, window: int) -> list[dict[str, Any]]:
    chunks = []
    for i in range(len(steps)):
        key_steps = steps[max(0, i - window + 1) : i + 1]
        value_steps = steps[i : min(len(steps), i + window)]
        current = steps[i]
        key = "\n".join(step_brief(s) for s in key_steps)
        value = "\n".join(step_brief(s) for s in value_steps)
        chunks.append(
            {
                "chunk_id": f"{session_id}:chunk:{i:04d}",
                "session_id": session_id,
                "source_step_index": i,
                "window": window,
                "key": clean_text(key, VALUE_TEXT_LIMIT),
                "value": clean_text(value, VALUE_TEXT_LIMIT),
                "metadata": {
                    "timestamp": current.timestamp,
                    "cwd": current.cwd,
                    "event_type": current.event_type,
                    "role": current.role,
                    "model": current.model,
                    "tool_name": current.tool_name,
                    "tool_use_id": current.tool_use_id,
                },
            }
        )
    return chunks


def make_action_observations(steps: list[Step]) -> list[ActionObservation]:
    by_tool_result: dict[str, Step] = {}
    for step in steps:
        if step.event_type == "tool_result" and step.tool_use_id:
            by_tool_result[step.tool_use_id] = step

    pairs: list[ActionObservation] = []
    context_window: list[Step] = []
    for step in steps:
        if step.event_type in {"assistant_text", "user_feedback", "user_command"}:
            context_window.append(step)
            context_window = context_window[-4:]
        if step.event_type != "tool_call":
            continue
        result = by_tool_result.get(step.tool_use_id or "")
        observation = result.observation if result else ""
        outcome = (result.outcome if result else None) or "missing_result"
        context = "\n".join(step_brief(s) for s in context_window)
        pairs.append(
            ActionObservation(
                ao_index=len(pairs),
                source_step_index=step.step_index,
                timestamp=step.timestamp,
                model=step.model,
                cwd=step.cwd,
                action=step_brief(step),
                observation=observation,
                outcome=outcome,
                tool_use_id=step.tool_use_id,
                tool_name=step.tool_name,
                recent_context=clean_text(context, 1400),
            )
        )
    return pairs


def make_ao_chunks(pairs: list[ActionObservation], session_id: str, window: int) -> list[dict[str, Any]]:
    chunks = []
    for i in range(len(pairs)):
        key_pairs = pairs[max(0, i - window + 1) : i + 1]
        value_pairs = pairs[i : min(len(pairs), i + window)]
        key = "\n".join(
            f"[{p.ao_index}] context={p.recent_context} ACTION={p.action} OUTCOME={p.outcome}"
            for p in key_pairs
        )
        value = "\n".join(
            f"[{p.ao_index}] ACTION={p.action}\nOBS={p.observation}\nOUTCOME={p.outcome}"
            for p in value_pairs
        )
        chunks.append(
            {
                "chunk_id": f"{session_id}:aochunk:{i:04d}",
                "session_id": session_id,
                "source_ao_index": i,
                "window": window,
                "key": clean_text(key, VALUE_TEXT_LIMIT),
                "value": clean_text(value, VALUE_TEXT_LIMIT),
                "metadata": {
                    "timestamp": pairs[i].timestamp,
                    "model": pairs[i].model,
                    "cwd": pairs[i].cwd,
                    "tool_name": pairs[i].tool_name,
                    "outcome": pairs[i].outcome,
                },
            }
        )
    return chunks


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary(
    path: Path,
    source: Path,
    session_id: str,
    steps: list[Step],
    chunks: list[dict[str, Any]],
    pairs: list[ActionObservation],
    ao_chunks: list[dict[str, Any]],
) -> None:
    counts = Counter(step.event_type for step in steps)
    tools = Counter(step.tool_name for step in steps if step.tool_name)
    models = Counter(step.model for step in steps if step.model)
    user_feedback = [s for s in steps if s.event_type == "user_feedback"]
    assistant_text = [s for s in steps if s.event_type == "assistant_text"]
    tool_calls = [s for s in steps if s.event_type == "tool_call"]

    phases = [
        ("orientation", "Inspect existing repos, API docs, keys/libraries, and model-call patterns."),
        ("proof_of_concept", "Install local instructor editable and prove structured ASCII generation with gpt-5.4-mini."),
        ("observable_experiment", "Generate freeform/structured/refined ASCII, score with gemini-embedding-2, and build a gallery."),
        ("user_pivot", "Incorporate user feedback that one-prompt-per-page labeling and note space matter."),
        ("abstract_labeler", "Switch to abstract self-directed prompts and build keyboard-first label.html."),
    ]

    lines: list[str] = []
    lines.append("# MATM Pilot Dataset: Claude ASCII-Instructor Session")
    lines.append("")
    lines.append(f"- Source log: `{source}`")
    lines.append(f"- Session ID: `{session_id}`")
    lines.append(f"- Extracted steps: {len(steps)}")
    lines.append(f"- MATM chunks: {len(chunks)}")
    lines.append(f"- Action-observation pairs: {len(pairs)}")
    lines.append(f"- Action-observation chunks: {len(ao_chunks)}")
    lines.append(f"- Chunking: state-conditioned rolling window, `l=5`; key is recent interaction history, value is the next five normalized steps.")
    lines.append("")
    lines.append("## Event Counts")
    lines.append("")
    for key, value in sorted(counts.items()):
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("## Tool Calls")
    lines.append("")
    for key, value in tools.most_common():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    if models:
        lines.append("## Assistant Models")
        lines.append("")
        for key, value in models.most_common():
            lines.append(f"- `{key}`: {value}")
        lines.append("")
    lines.append("## Reconstructed Task Phases")
    lines.append("")
    for name, desc in phases:
        lines.append(f"- `{name}`: {desc}")
    lines.append("")
    lines.append("## Notable User Feedback")
    lines.append("")
    for step in user_feedback[:8]:
        lines.append(f"- Step {step.step_index}: {step.action}")
    lines.append("")
    lines.append("## Representative Assistant Decisions")
    lines.append("")
    for step in assistant_text[:10]:
        lines.append(f"- Step {step.step_index}: {step.action}")
    lines.append("")
    lines.append("## Representative Tool Trajectory")
    lines.append("")
    for step in tool_calls[:12]:
        lines.append(f"- Step {step.step_index}: {step.action} -> `{step.observation[:180]}`")
    lines.append("")
    lines.append("## Cleaner Action-Observation Samples")
    lines.append("")
    for pair in pairs[:10]:
        lines.append(f"- AO {pair.ao_index}: `{pair.action[:160]}` => `{pair.outcome}` / `{pair.observation[:180]}`")
    lines.append("")
    lines.append("## MATM Interpretation")
    lines.append("")
    lines.append("This pilot turns one Claude conversation into agent-generated procedural memory. A consumer agent could retrieve chunks by current state, for example: setting up `instructor` with the OpenAI Responses API, scoring generated ASCII with `gemini-embedding-2`, or redesigning an annotation UI after user feedback. The next layer would add embeddings and marginal-utility labels from downstream reuse attempts, as in the MATM paper's LTRT setup.")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()

    records = load_records(args.source)
    session_id = next((r.get("sessionId") for r in records if r.get("sessionId")), args.source.stem)
    args.outdir.mkdir(parents=True, exist_ok=True)

    steps = extract_steps(records)
    chunks = make_chunks(steps, session_id, args.window)
    pairs = make_action_observations(steps)
    ao_chunks = make_ao_chunks(pairs, session_id, args.window)
    metadata = {
        "session_id": session_id,
        "source": str(args.source),
        "records": len(records),
        "steps": len(steps),
        "chunks": len(chunks),
        "action_observations": len(pairs),
        "action_observation_chunks": len(ao_chunks),
        "window": args.window,
        "event_counts": dict(Counter(step.event_type for step in steps)),
        "tool_counts": dict(Counter(step.tool_name for step in steps if step.tool_name)),
        "model_counts": dict(Counter(step.model for step in steps if step.model)),
    }

    write_jsonl(args.outdir / "steps.jsonl", [asdict(step) for step in steps])
    write_jsonl(args.outdir / "chunks_l5.jsonl", chunks)
    write_jsonl(args.outdir / "action_observations.jsonl", [asdict(pair) for pair in pairs])
    write_jsonl(args.outdir / "ao_chunks_l5.jsonl", ao_chunks)
    (args.outdir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    write_summary(args.outdir / "README.md", args.source, session_id, steps, chunks, pairs, ao_chunks)


if __name__ == "__main__":
    main()
