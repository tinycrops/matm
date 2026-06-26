#!/usr/bin/env python3
"""Build a task-specific MATM memory pack from local Claude trace chunks."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import normalize


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_ROOT = REPO_ROOT / "local_traces"
DEFAULT_INDEX = DEFAULT_TRACE_ROOT / "index"
DEFAULT_APPLIED = DEFAULT_TRACE_ROOT / "applied"

TRAP_PATTERNS = re.compile(
    r"\b("
    r"bug|blocker|gotcha|wrong|failed|failure|premise|must|needs?|required|"
    r"root-cause|root cause|trap|false pass|substring|template|cuda|pascal|"
    r"context|truncat|sudo|nvidia-container"
    r")\b",
    re.IGNORECASE,
)

GENERIC_QUERY_TERMS = {
    "model",
    "models",
    "trace",
    "traces",
    "agent",
    "task",
    "vibethinker",
    "vibethinker-3b",
}


@dataclass
class SearchResult:
    rank: int
    score: float
    row: dict[str, Any]

    @property
    def session_id(self) -> str:
        return str(self.row.get("session_id") or "")

    @property
    def chunk_id(self) -> str:
        return str(self.row.get("chunk_id") or "")

    @property
    def source_trace(self) -> str:
        source_file = str(self.row.get("_source_file") or "")
        marker = "/extracted/"
        if marker in source_file:
            return source_file.split(marker, 1)[1].split("/", 1)[0]
        return ""

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = self.row.get("metadata")
        return metadata if isinstance(metadata, dict) else {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def compact(text: str, limit: int = 520) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 18].rstrip() + " ...[truncated]"


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug[:72] or "matm-apply"


def embed_query(index: Path, query: str) -> np.ndarray:
    vectorizer = joblib.load(index / "tfidf.joblib")
    sparse = vectorizer.transform([query])
    svd_path = index / "svd.joblib"
    if svd_path.exists():
        dense = joblib.load(svd_path).transform(sparse)
        return normalize(dense, norm="l2", axis=1).astype(np.float32)[0]
    return normalize(sparse, norm="l2", axis=1).astype(np.float32).toarray()[0]


def search(index: Path, query: str, top_k: int) -> list[SearchResult]:
    rows = load_jsonl(index / "chunks.jsonl")
    if not rows:
        raise SystemExit(f"No chunks found at {index / 'chunks.jsonl'}")
    vectors = np.load(index / "vectors.npy")
    qvec = embed_query(index, query)
    scores = vectors @ qvec
    order = np.argsort(-scores)[:top_k]
    return [
        SearchResult(rank=rank, score=float(scores[row_idx]), row=rows[int(row_idx)])
        for rank, row_idx in enumerate(order, 1)
    ]


def observation_matches(query: str, observation: dict[str, Any], sessions: set[str], traces: set[str]) -> bool:
    if observation.get("source_session_id") in sessions:
        return True
    if observation.get("source_trace") in traces:
        return True
    haystack = " ".join(
        str(observation.get(field, ""))
        for field in ("summary", "runtime", "model_ref", "observation_type")
    ).lower()
    query_terms = {
        term
        for term in re.findall(r"[a-z0-9_.:-]{4,}", query.lower())
        if term not in GENERIC_QUERY_TERMS
    }
    return sum(1 for term in query_terms if term in haystack) >= 2


def relevant_observations(trace_root: Path, query: str, results: list[SearchResult]) -> list[dict[str, Any]]:
    sessions = {result.session_id for result in results if result.session_id}
    traces = {result.source_trace for result in results if result.source_trace}
    rows = load_jsonl(trace_root / "index" / "agent_observations.jsonl")
    return [row for row in rows if observation_matches(query, row, sessions, traces)]


def relevant_artifacts(trace_root: Path, observations: list[dict[str, Any]], results: list[SearchResult]) -> list[Path]:
    traces = {result.source_trace for result in results if result.source_trace}
    traces.update(str(row.get("source_trace")) for row in observations if row.get("source_trace"))
    artifact_files: list[Path] = []
    for trace in sorted(traces):
        artifact_dir = trace_root / "artifacts" / trace
        if artifact_dir.exists():
            artifact_files.extend(sorted(path for path in artifact_dir.rglob("*") if path.is_file()))
    return artifact_files


def extract_traps(results: list[SearchResult], observations: list[dict[str, Any]], limit: int = 8) -> list[str]:
    candidates: list[str] = []
    for observation in observations:
        summary = str(observation.get("summary") or "")
        if summary:
            candidates.append(summary)
    for result in results:
        text = str(result.row.get("key") or "")
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
            if "ACTION=" in sentence or "CALL " in sentence or "RESULT " in sentence:
                continue
            if "ASSISTANT assistant_text:" in sentence:
                sentence = sentence.split("ASSISTANT assistant_text:", 1)[1]
            if "USER user_feedback:" in sentence:
                sentence = sentence.split("USER user_feedback:", 1)[1]
            if TRAP_PATTERNS.search(sentence):
                candidates.append(sentence)
    seen: set[str] = set()
    traps: list[str] = []
    for candidate in candidates:
        item = compact(candidate, 260)
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        traps.append(item)
        if len(traps) >= limit:
            break
    return traps


def render_pack(query: str, index: Path, trace_root: Path, top_k: int) -> str:
    results = search(index, query, top_k)
    observations = relevant_observations(trace_root, query, results)
    artifacts = relevant_artifacts(trace_root, observations, results)
    traps = extract_traps(results, observations)
    sessions = sorted({result.session_id for result in results if result.session_id})
    traces = sorted({result.source_trace for result in results if result.source_trace})

    lines: list[str] = []
    lines.append("# MATM Operator Memory Pack")
    lines.append("")
    lines.append(f"- Query: `{query}`")
    lines.append(f"- Built: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append(f"- Index: `{index}`")
    lines.append(f"- Retrieved chunks: {len(results)}")
    if sessions:
        lines.append(f"- Source sessions: {', '.join(f'`{session}`' for session in sessions)}")
    if traces:
        lines.append(f"- Source traces: {', '.join(f'`{trace}`' for trace in traces)}")

    if observations:
        lines.append("")
        lines.append("## Operator Shortcuts")
        lines.append("")
        for row in observations:
            parts = [str(row.get("summary") or "").strip()]
            endpoints = row.get("endpoints")
            if isinstance(endpoints, list) and endpoints:
                parts.append("Endpoints: " + ", ".join(f"`{endpoint}`" for endpoint in endpoints))
            runtime = row.get("runtime")
            if runtime:
                parts.append(f"Runtime: `{runtime}`.")
            model_ref = row.get("model_ref")
            if model_ref:
                parts.append(f"Model reference: `{model_ref}`.")
            lines.append(f"- {compact(' '.join(parts), 620)}")

    if artifacts:
        lines.append("")
        lines.append("## Durable Artifacts")
        lines.append("")
        for path in artifacts:
            rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
            lines.append(f"- `{rel}`")

    if traps:
        lines.append("")
        lines.append("## Known Traps")
        lines.append("")
        for trap in traps:
            lines.append(f"- {trap}")

    lines.append("")
    lines.append("## Retrieved Trace Chunks")
    lines.append("")
    for result in results:
        metadata = result.metadata
        tool = metadata.get("tool_name") or "unknown"
        outcome = metadata.get("outcome") or "unknown"
        model = metadata.get("model") or "unknown"
        lines.append(
            f"### {result.rank}. `{result.chunk_id}` "
            f"(score={result.score:.4f}, tool={tool}, outcome={outcome}, model={model})"
        )
        lines.append("")
        lines.append("Key:")
        lines.append("")
        lines.append(textwrap.indent(compact(str(result.row.get("key") or ""), 900), "> "))
        lines.append("")
        lines.append("Procedure:")
        lines.append("")
        lines.append(textwrap.indent(compact(str(result.row.get("value") or ""), 1400), "> "))
        lines.append("")

    lines.append("## Suggested Replay Rubric")
    lines.append("")
    lines.append("- Did the next agent reuse a retrieved artifact or command instead of rediscovering it?")
    lines.append("- Did it avoid at least one known trap from this pack?")
    lines.append("- Did it verify the actual endpoint/model/workload rather than a substitute?")
    lines.append("- Did the final result include a working smoke test or concrete failure evidence?")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a MATM memory pack for a live task.")
    parser.add_argument("query", nargs="+", help="Task/state query to retrieve memories for.")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("-k", "--top-k", type=int, default=6)
    parser.add_argument("--out", type=Path, help="Write the memory pack to this file.")
    parser.add_argument("--save", action="store_true", help="Save under local_traces/applied with a timestamped filename.")
    args = parser.parse_args()

    query = " ".join(args.query)
    output = render_pack(query=query, index=args.index, trace_root=args.trace_root, top_k=args.top_k)
    out_path = args.out
    if args.save and out_path is None:
        DEFAULT_APPLIED.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = DEFAULT_APPLIED / f"{stamp}-{slugify(query)}.md"
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output + "\n")
        print(out_path)
    else:
        print(output)


if __name__ == "__main__":
    main()
