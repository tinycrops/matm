#!/usr/bin/env python3
"""Build a local MATM retrieval index over Claude action-observation chunks.

The first production layer for MATM is intentionally simple and reproducible:
collect extracted `ao_chunks_l5.jsonl` rows, embed each chunk key as a dense
normalized vector using TF-IDF + SVD, and store row-aligned vectors plus
metadata. The model-free backend avoids making pilot indexing depend on a
network download while keeping the index format swappable later.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


DEFAULT_ROOT = Path("/home/ath/matm_claude_traces")
DEFAULT_INDEX = DEFAULT_ROOT / "index"


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            row["_source_file"] = str(path)
            rows.append(row)
    return rows


def discover_chunk_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.glob("*/ao_chunks_l5.jsonl")
        if path.is_file() and path.parent.name != "index"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def safe_text(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    return value if isinstance(value, str) else ""


def build_vectors(texts: list[str], requested_dim: int) -> tuple[np.ndarray, TfidfVectorizer, TruncatedSVD | None]:
    vectorizer = TfidfVectorizer(
        analyzer="word",
        lowercase=True,
        max_df=0.95,
        max_features=50000,
        min_df=1,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    sparse = vectorizer.fit_transform(texts)
    if sparse.shape[0] < 3 or sparse.shape[1] < 3:
        vectors = normalize(sparse, norm="l2", axis=1).astype(np.float32).toarray()
        return vectors, vectorizer, None

    dim = min(requested_dim, sparse.shape[0] - 1, sparse.shape[1] - 1)
    dim = max(2, dim)
    svd = TruncatedSVD(n_components=dim, random_state=13)
    dense = svd.fit_transform(sparse)
    vectors = normalize(dense, norm="l2", axis=1).astype(np.float32)
    return vectors, vectorizer, svd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a MATM pilot index over Claude AO chunks.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Trace root containing */ao_chunks_l5.jsonl.")
    parser.add_argument("--out", type=Path, default=DEFAULT_INDEX, help="Index output directory.")
    parser.add_argument("--dim", type=int, default=128, help="Requested dense SVD dimension.")
    args = parser.parse_args()

    chunk_files = discover_chunk_files(args.root)
    if not chunk_files:
        raise SystemExit(f"No ao_chunks_l5.jsonl files found under {args.root}")

    rows: list[dict[str, Any]] = []
    for path in chunk_files:
        rows.extend(iter_jsonl(path))
    if not rows:
        raise SystemExit("Chunk files were found, but they contained no rows.")

    texts = [safe_text(row, "key") for row in rows]
    vectors, vectorizer, svd = build_vectors(texts, args.dim)

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "chunks.jsonl", rows)
    np.save(args.out / "vectors.npy", vectors)
    joblib.dump(vectorizer, args.out / "tfidf.joblib")
    if svd is not None:
        joblib.dump(svd, args.out / "svd.joblib")
    else:
        svd_path = args.out / "svd.joblib"
        if svd_path.exists():
            svd_path.unlink()

    sessions = sorted({str(row.get("session_id", "")) for row in rows if row.get("session_id")})
    tool_counts: dict[str, int] = {}
    outcome_counts: dict[str, int] = {}
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        tool_name = str(metadata.get("tool_name") or "unknown")
        outcome = str(metadata.get("outcome") or "unknown")
        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

    manifest = {
        "backend": "tfidf_svd_dense",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "chunk_files": [str(path) for path in chunk_files],
        "chunks": len(rows),
        "dim": int(vectors.shape[1]),
        "index_dir": str(args.out),
        "outcome_counts": outcome_counts,
        "root": str(args.root),
        "sessions": sessions,
        "tool_counts": tool_counts,
    }
    (args.out / "metadata.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.out / "README.md").write_text(
        "# Claude MATM Pilot Index\n\n"
        f"- Backend: `{manifest['backend']}`\n"
        f"- Chunks: {manifest['chunks']}\n"
        f"- Dense dimensions: {manifest['dim']}\n"
        f"- Sessions: {len(sessions)}\n"
        f"- Built at: `{manifest['built_at']}`\n\n"
        "Search with:\n\n"
        "```bash\n"
        f"python3 /home/ath/.codex/skills/claude-matm-extraction/scripts/matm_search.py --index {args.out} \"your query\"\n"
        "```\n",
    )
    print(f"Indexed {len(rows)} chunks from {len(chunk_files)} files into {args.out}")
    print(f"Vector shape: {vectors.shape[0]} x {vectors.shape[1]}")


if __name__ == "__main__":
    main()
