#!/usr/bin/env python3
"""Search a local Claude MATM pilot index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import normalize


DEFAULT_INDEX = Path("/home/ath/matm_claude_traces/index")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 18].rstrip() + " ...[truncated]"


def embed_query(index: Path, query: str) -> np.ndarray:
    vectorizer = joblib.load(index / "tfidf.joblib")
    sparse = vectorizer.transform([query])
    svd_path = index / "svd.joblib"
    if svd_path.exists():
        svd = joblib.load(svd_path)
        dense = svd.transform(sparse)
        return normalize(dense, norm="l2", axis=1).astype(np.float32)[0]
    return normalize(sparse, norm="l2", axis=1).astype(np.float32).toarray()[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search a Claude MATM pilot index.")
    parser.add_argument("query", nargs="+", help="Task/state query text.")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("-k", "--top-k", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Emit JSONL results.")
    parser.add_argument("--show-value", action="store_true", help="Print retrieved procedural value text.")
    args = parser.parse_args()

    query = " ".join(args.query)
    vectors = np.load(args.index / "vectors.npy")
    rows = load_jsonl(args.index / "chunks.jsonl")
    qvec = embed_query(args.index, query)
    scores = vectors @ qvec
    order = np.argsort(-scores)[: args.top_k]

    for rank, row_idx in enumerate(order, 1):
        row = rows[int(row_idx)]
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        result = {
            "rank": rank,
            "score": float(scores[row_idx]),
            "chunk_id": row.get("chunk_id"),
            "session_id": row.get("session_id"),
            "source_ao_index": row.get("source_ao_index"),
            "source_file": row.get("_source_file"),
            "tool_name": metadata.get("tool_name"),
            "outcome": metadata.get("outcome"),
            "key": row.get("key", ""),
            "value": row.get("value", ""),
        }
        if args.json:
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            continue
        print(f"[{rank}] score={result['score']:.4f} {result['chunk_id']} tool={result['tool_name']} outcome={result['outcome']}")
        print(f"    key: {truncate(str(result['key']), 420)}")
        if args.show_value:
            print(f"    value: {truncate(str(result['value']), 900)}")
        print()


if __name__ == "__main__":
    main()
