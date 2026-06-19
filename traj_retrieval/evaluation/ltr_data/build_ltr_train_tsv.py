#!/usr/bin/env python3
"""Export chunk-aligned enriched disagreement JSON to LTR training TSVs.

Reads the JSON produced by ``build_chunk_aligned_enriched_jsonl.py`` (or the
equivalent enriched disagreement files) and writes:

- ``ltr/data_out/<environment>/ltr_train.tsv`` — training features
- ``ltr/data_out/<environment>/qid_map.tsv`` — qid-to-query mapping

The TSV schema matches what ``python -m ltr.train`` and the retrieval runtime
expect: ``qid``, ``label``, ``docid``, then numeric feature columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ltr.runtime_features import (  # noqa: E402
    load_consumer_model_features,
    project_features_to_vector,
    sanitize_agent_type,
)

LTR_DATA_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "ltr" / "data_out"

SCALAR_FEATURE_COLUMNS: Tuple[str, ...] = (
    "context_only_jaccard",
    "context_only_overlap_bigram",
    "context_only_overlap_tfidf",
    "context_only_query_overlap",
    "context_only_similarity",
    "embedding_similarity",
    "goal_only_jaccard",
    "goal_only_overlap_bigram",
    "goal_only_overlap_tfidf",
    "goal_only_query_overlap",
    "goal_only_similarity",
    "progress_alignment_abs_diff",
    "query_length",
    "query_overlap_ratio",
    "retrieved_text_length",
    "retriever_score",
    "state_only_jaccard",
    "state_only_overlap_bigram",
    "state_only_overlap_tfidf",
    "state_only_query_overlap",
    "state_only_similarity",
    "step_index",
    "success_flag",
    "task_match",
    "task_variation_match",
    "text_overlap_bigram",
    "text_overlap_jaccard",
    "text_overlap_tfidf",
    "total_steps",
    "trajectory_length",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="Chunk-aligned or enriched disagreement JSON file(s).",
    )
    parser.add_argument(
        "--environment",
        choices=["alfworld", "webarena"],
        default=None,
        help="Benchmark environment. Inferred from input paths when omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: ltr/data_out/<environment>/).",
    )
    parser.add_argument(
        "--reference-tsv",
        type=Path,
        default=None,
        help=(
            "Optional existing ltr_train.tsv whose feature column order should "
            "be reused (useful when retraining against shipped model schemas)."
        ),
    )
    return parser.parse_args()


def infer_environment(path: Path) -> str:
    path_parts = set(path.parts)
    for env in ("alfworld", "webarena"):
        if env in path_parts:
            return env
    raise ValueError(f"Cannot infer environment from path: {path}")


def iter_training_entries(
    data: Dict[str, Any],
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    for query_key, value in data.items():
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, dict):
                    yield query_key, entry
        elif isinstance(value, dict):
            yield query_key, value


def entry_docid(entry: Dict[str, Any]) -> Optional[str]:
    run_key = entry.get("run_key")
    if run_key:
        return str(run_key)

    metadata = entry.get("metadata") or {}
    parts = [
        metadata.get("run_id"),
        metadata.get("task_name"),
        metadata.get("variation"),
        metadata.get("rank_retrieve"),
    ]
    if any(part is not None for part in parts):
        return "|||".join("" if part is None else str(part) for part in parts)
    return None


def load_reference_feature_columns(reference_tsv: Path) -> List[str]:
    with reference_tsv.open("r", encoding="utf-8") as file_obj:
        reader = csv.reader(file_obj, delimiter="\t")
        header = next(reader, None)
    if not header:
        raise ValueError(f"Empty reference TSV: {reference_tsv}")
    ignore = {"qid", "label", "docid", "folder_nativeid"}
    return [column for column in header if column not in ignore]


def build_source_feature_columns() -> List[str]:
    feature_columns = load_consumer_model_features().get("feature_columns", [])
    return [f"source_{name}" for name in feature_columns]


def collect_agent_type_columns(
    entries: Sequence[Tuple[str, Dict[str, Any]]],
    reference_columns: Optional[Sequence[str]] = None,
) -> List[str]:
    agent_types = set()
    for _, entry in entries:
        features = entry.get("features") or {}
        agent_type = features.get("agent_type")
        if agent_type is not None:
            agent_types.add(sanitize_agent_type(agent_type))

    if reference_columns:
        for column in reference_columns:
            if column.startswith("agent_type__"):
                agent_types.add(column.split("agent_type__", 1)[1])

    return [f"agent_type__{name}" for name in sorted(agent_types)]


def build_feature_columns(
    entries: Sequence[Tuple[str, Dict[str, Any]]],
    reference_tsv: Optional[Path],
) -> List[str]:
    if reference_tsv is not None:
        return load_reference_feature_columns(reference_tsv)

    source_columns = build_source_feature_columns()
    agent_columns = collect_agent_type_columns(entries)
    return list(SCALAR_FEATURE_COLUMNS) + source_columns + agent_columns


def assign_qids(entries: Sequence[Tuple[str, Dict[str, Any]]]) -> Dict[str, int]:
    query_keys = sorted({query_key for query_key, _ in entries})
    return {query_key: idx + 1 for idx, query_key in enumerate(query_keys)}


def build_rows(
    entries: Sequence[Tuple[str, Dict[str, Any]]],
    feature_columns: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[int, str], Dict[str, int]]:
    qid_by_query = assign_qids(entries)
    qid_to_query = {qid: query for query, qid in qid_by_query.items()}
    rows: List[Dict[str, Any]] = []
    skipped = 0

    for query_key, entry in entries:
        if entry.get("enrichment_failed"):
            skipped += 1
            continue

        features = entry.get("features")
        if not isinstance(features, dict):
            skipped += 1
            continue

        docid = entry_docid(entry)
        if not docid:
            skipped += 1
            continue

        vector, _ = project_features_to_vector(features, list(feature_columns))
        row: Dict[str, Any] = {
            "qid": qid_by_query[query_key],
            "label": float(entry.get("score", 0)),
            "docid": docid,
        }
        for column, value in zip(feature_columns, vector):
            row[column] = value
        rows.append(row)

    stats = {
        "entries_total": len(entries),
        "rows_written": len(rows),
        "rows_skipped": skipped,
        "query_groups": len(qid_by_query),
    }
    return rows, qid_to_query, stats


def write_tsv(path: Path, rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["qid", "label", "docid", *columns]
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=header, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in header})


def write_qid_map(path: Path, qid_to_query: Dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj, delimiter="\t")
        writer.writerow(["qid", "query"])
        for qid in sorted(qid_to_query):
            writer.writerow([qid, qid_to_query[qid]])


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {path}")
    return data


def main() -> int:
    args = parse_args()
    inputs = [path.resolve() for path in args.inputs]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")

    environment = args.environment or infer_environment(inputs[0])
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / environment)
    output_tsv = output_dir / "ltr_train.tsv"
    output_qid_map = output_dir / "qid_map.tsv"

    all_entries: List[Tuple[str, Dict[str, Any]]] = []
    for path in inputs:
        data = load_json(path)
        all_entries.extend(list(iter_training_entries(data)))

    if not all_entries:
        raise ValueError("No training entries found in the provided input files.")

    feature_columns = build_feature_columns(all_entries, args.reference_tsv)
    rows, qid_to_query, stats = build_rows(all_entries, feature_columns)

    if not rows:
        raise ValueError(
            "No rows exported. Ensure inputs contain enriched entries with "
            "`features`, `score`, and `run_key` (or reconstructable metadata)."
        )

    write_tsv(output_tsv, rows, feature_columns)
    write_qid_map(output_qid_map, qid_to_query)

    print(f"Environment     : {environment}")
    print(f"Input files     : {len(inputs)}")
    print(f"Query groups    : {stats['query_groups']}")
    print(f"Rows written    : {stats['rows_written']}")
    print(f"Rows skipped    : {stats['rows_skipped']}")
    print(f"Feature columns : {len(feature_columns)}")
    print(f"Wrote TSV       : {output_tsv}")
    print(f"Wrote qid map   : {output_qid_map}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
