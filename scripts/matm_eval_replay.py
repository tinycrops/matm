#!/usr/bin/env python3
"""Replay-style checks that local MATM traces retrieve useful operator memory."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))

import matm_apply  # noqa: E402


DEFAULT_OUT = REPO_ROOT / "local_traces" / "applied"


@dataclass(frozen=True)
class ReplayTask:
    name: str
    query: str
    expected_sessions: tuple[str, ...]
    expected_terms: tuple[str, ...]
    expected_artifacts: tuple[str, ...] = ()


TASKS = (
    ReplayTask(
        name="containerized_vibethinker",
        query="deploy VibeThinker on the vibecluster with sudo-free containers and verify endpoints",
        expected_sessions=("42d53d4f-1922-4e19-934b-5f20cb9bcdab",),
        expected_terms=("llama.cpp", "8080", "VibeThinker", "sudo-free"),
        expected_artifacts=("deploy-vibethinker.sh", "vibethinker-containerized-inference.md"),
    ),
    ReplayTask(
        name="two_token_vibethinker",
        query="debug VibeThinker only outputting two tokens in ollama or a container",
        expected_sessions=("42d53d4f-1922-4e19-934b-5f20cb9bcdab",),
        expected_terms=("ChatML", "template", "VibeThinker"),
        expected_artifacts=("Modelfile.vibethinker", "vibethinker-is-the-workload.md"),
    ),
    ReplayTask(
        name="full_reasoning_eval",
        query="run distributed VibeThinker plus qwen coder eval without truncating reasoning",
        expected_sessions=("94967c56-e4f5-4267-804b-16cb3cd819ec",),
        expected_terms=("thinking", "trace", "caller", "truncated"),
    ),
)


def pack_path(out_dir: Path, task: ReplayTask) -> Path:
    return out_dir / f"{task.name}.md"


def evaluate_task(task: ReplayTask, out_dir: Path, top_k: int) -> dict[str, Any]:
    output = matm_apply.render_pack(
        query=task.query,
        index=matm_apply.DEFAULT_INDEX,
        trace_root=matm_apply.DEFAULT_TRACE_ROOT,
        top_k=top_k,
    )
    path = pack_path(out_dir, task)
    path.write_text(output + "\n")
    lower = output.lower()
    session_hits = [session for session in task.expected_sessions if session.lower() in lower]
    term_hits = [term for term in task.expected_terms if term.lower() in lower]
    artifact_hits = [artifact for artifact in task.expected_artifacts if artifact.lower() in lower]
    required = len(task.expected_sessions) + len(task.expected_terms) + len(task.expected_artifacts)
    hits = len(session_hits) + len(term_hits) + len(artifact_hits)
    return {
        "name": task.name,
        "query": task.query,
        "pack": str(path),
        "passed": hits == required,
        "hits": hits,
        "required": required,
        "session_hits": session_hits,
        "term_hits": term_hits,
        "artifact_hits": artifact_hits,
        "missing_sessions": [s for s in task.expected_sessions if s not in session_hits],
        "missing_terms": [t for t in task.expected_terms if t not in term_hits],
        "missing_artifacts": [a for a in task.expected_artifacts if a not in artifact_hits],
    }


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# MATM Replay Smoke Report",
        "",
        f"- Built: `{report['built_at']}`",
        f"- Tasks: {report['tasks']}",
        f"- Passed: {report['passed']}",
        "",
        "## Results",
        "",
    ]
    for result in report["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        lines.append(f"### {result['name']} - {status}")
        lines.append("")
        lines.append(f"- Query: `{result['query']}`")
        lines.append(f"- Pack: `{result['pack']}`")
        lines.append(f"- Hits: {result['hits']} / {result['required']}")
        if result["missing_sessions"]:
            lines.append(f"- Missing sessions: `{', '.join(result['missing_sessions'])}`")
        if result["missing_terms"]:
            lines.append(f"- Missing terms: `{', '.join(result['missing_terms'])}`")
        if result["missing_artifacts"]:
            lines.append(f"- Missing artifacts: `{', '.join(result['missing_artifacts'])}`")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canned MATM replay smoke checks.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("-k", "--top-k", type=int, default=8)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = [evaluate_task(task, args.out_dir, args.top_k) for task in TASKS]
    report = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "tasks": len(results),
        "passed": sum(1 for result in results if result["passed"]),
        "results": results,
    }
    json_path = args.out_dir / "replay_report.json"
    md_path = args.out_dir / "replay_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_markdown_report(md_path, report)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    if report["passed"] != report["tasks"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
