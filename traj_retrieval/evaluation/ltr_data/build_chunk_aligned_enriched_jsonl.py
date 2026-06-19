import os
#!/usr/bin/env python3
"""
Build chunk-aligned enriched JSON files from disagreement datasets.

Behavior:
- Does NOT trust `rank_retrieve`.
- Uses `metadata.thought_id` + `retrieved_chunk` to find LanceDB row.
- Preserves original JSON structure and adds/overwrites `retrieved_doc` only.
- For ambiguous matches, falls back to the first candidate (never empty because ambiguous).
"""

import argparse
import json
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm


ENV_TO_LANCEDB = {
    "alfworld": os.path.join(
        os.environ.get("MATM_DATA_ROOT", "environments"),
        "train_only_lancedb/alfworld/lancedb_indices",
    ),
    "webarena": os.path.join(
        os.environ.get("MATM_DATA_ROOT", "environments"),
        "train_only_lancedb/webarena/lancedb_indices",
    ),
}


DOC_FIELDS = (
    "key_raw_goal",
    "key_raw_state",
    "key_raw_context",
    "key_raw_progress",
)


_WORKER_ENV: Optional[str] = None
_WORKER_SOURCE_FILE: Optional[str] = None
_WORKER_THOUGHT_CACHE: Optional["ThoughtIdIndexCache"] = None
_WORKER_ADD_ALIGNMENT_DEBUG: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create chunk-aligned enriched JSON by resolving retrieved_doc "
            "from LanceDB via thought_id + retrieved_chunk."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="Input disagreement or enriched JSON files.",
    )

    # Backward-compatible alias
    parser.add_argument(
        "--output-jsonl",
        dest="output_json",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output-json",
        dest="output_json",
        type=Path,
        default=None,
        help=(
            "Output JSON path override. Only valid when exactly one input is provided. "
            "With multiple inputs, each input will generate its own sibling output file."
        ),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help=(
            "Summary JSON path override. Only valid when exactly one input is provided. "
            "Default is <output-json>.summary.json for each input."
        ),
    )
    parser.add_argument(
        "--max-thought-rows",
        type=int,
        default=5000,
        help="Maximum LanceDB rows fetched for each thought_id.",
    )
    parser.add_argument(
        "--limit-per-file",
        type=int,
        default=None,
        help="Optional debug limit of entries per input file.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes per input file (default: 1).",
    )
    parser.add_argument(
        "--add-alignment-debug",
        action="store_true",
        help="Also write `retrieved_doc_alignment` debug info into each entry.",
    )
    return parser.parse_args()


def infer_environment(path: Path) -> str:
    path_parts = set(path.parts)
    for env in ENV_TO_LANCEDB:
        if env in path_parts:
            return env
    raise ValueError(f"Cannot infer environment from path: {path}")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_chunk_signature(chunk: Any) -> Optional[Tuple[Tuple[str, str], ...]]:
    if not isinstance(chunk, list):
        return None
    signature: List[Tuple[str, str]] = []
    for step in chunk:
        if not isinstance(step, dict):
            return None
        signature.append(
            (
                normalize_text(step.get("action", "")),
                normalize_text(step.get("observation", "")),
            )
        )
    return tuple(signature)


def parse_guidance(guidance_value: Any) -> List[Dict[str, Any]]:
    if isinstance(guidance_value, str):
        try:
            parsed = json.loads(guidance_value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    if isinstance(guidance_value, list):
        return guidance_value
    return []


def extract_doc_from_row(row: Dict[str, Any]) -> Dict[str, str]:
    return {field: normalize_text(row.get(field, "")) for field in DOC_FIELDS}


def summarize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "thought_id": normalize_text(row.get("thought_id", "")),
        "task_name": normalize_text(row.get("task_name", "")),
        "variation_idx": normalize_text(row.get("variation_idx", "")),
        "success": bool(row.get("success", False)),
    }


def iter_entries(data: Dict[str, Any]) -> Iterable[Tuple[str, int, Dict[str, Any]]]:
    for query_group_key, value in data.items():
        if isinstance(value, list):
            for idx, entry in enumerate(value):
                if isinstance(entry, dict):
                    yield query_group_key, idx, entry
        elif isinstance(value, dict):
            yield query_group_key, 0, value


class ThoughtIdIndexCache:
    def __init__(self, table: Any, max_thought_rows: int):
        self.table = table
        self.max_thought_rows = max_thought_rows
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _load_thought_id(self, thought_id: str) -> Dict[str, Any]:
        if thought_id in self._cache:
            return self._cache[thought_id]

        escaped_thought_id = thought_id.replace("'", "''")
        rows = (
            self.table.search()
            .where(f"thought_id = '{escaped_thought_id}'")
            .limit(self.max_thought_rows)
            .to_list()
        )

        by_sig: Dict[Tuple[Tuple[str, str], ...], List[Dict[str, Any]]] = defaultdict(
            list
        )
        invalid_guidance_rows = 0
        for row in rows:
            guidance = parse_guidance(row.get("guidance", []))
            sig = normalize_chunk_signature(guidance)
            if sig is None:
                invalid_guidance_rows += 1
                continue
            by_sig[sig].append(row)

        packed = {
            "rows_total": len(rows),
            "invalid_guidance_rows": invalid_guidance_rows,
            "by_signature": by_sig,
        }
        self._cache[thought_id] = packed
        return packed

    def find_candidates(
        self, thought_id: str, chunk_sig: Tuple[Tuple[str, str], ...]
    ) -> Dict[str, Any]:
        packed = self._load_thought_id(thought_id)
        candidates = packed["by_signature"].get(chunk_sig, [])
        return {
            "rows_total": packed["rows_total"],
            "invalid_guidance_rows": packed["invalid_guidance_rows"],
            "candidates": candidates,
        }


def disambiguate_candidates(
    candidates: Sequence[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], str]:
    if len(candidates) == 1:
        return candidates[0], "unique"
    if not candidates:
        return None, "no_candidate"

    filtered = list(candidates)

    preferred_task = normalize_text(metadata.get("retrieved_task_name", ""))
    if preferred_task:
        task_filtered = [
            row
            for row in filtered
            if normalize_text(row.get("task_name", "")) == preferred_task
        ]
        if task_filtered:
            filtered = task_filtered

    preferred_variation = normalize_text(metadata.get("retrieved_variation", ""))
    if preferred_variation:
        variation_filtered = [
            row
            for row in filtered
            if normalize_text(row.get("variation_idx", "")) == preferred_variation
        ]
        if variation_filtered:
            filtered = variation_filtered

    if len(filtered) == 1:
        return filtered[0], "filtered_by_metadata"

    doc_signatures = {
        json.dumps(extract_doc_from_row(r), sort_keys=True) for r in filtered
    }
    if len(doc_signatures) == 1:
        return filtered[0], "multi_rows_same_doc"

    # User-required policy: for ambiguous cases, pick the first.
    return filtered[0], "ambiguous_fallback_first"


def build_enriched_entry_for_item(
    entry: Dict[str, Any],
    thought_cache: ThoughtIdIndexCache,
    add_alignment_debug: bool,
) -> Tuple[Dict[str, Any], str, str]:
    metadata = entry.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    thought_id = normalize_text(metadata.get("thought_id", ""))
    retrieved_chunk = entry.get("retrieved_chunk", [])
    chunk_sig = normalize_chunk_signature(retrieved_chunk)

    alignment: Dict[str, Any] = {
        "method": "thought_id+retrieved_chunk_guidance_exact",
        "status": "",
        "thought_id": thought_id,
        "rows_with_same_thought_id": 0,
        "invalid_guidance_rows": 0,
        "chunk_match_candidates": 0,
        "selection_mode": None,
        "selected_row": None,
    }

    matched_doc: Optional[Dict[str, str]] = None

    if not thought_id:
        alignment["status"] = "missing_thought_id"
    elif chunk_sig is None:
        alignment["status"] = "invalid_retrieved_chunk"
    else:
        lookup = thought_cache.find_candidates(thought_id, chunk_sig)
        candidates = lookup["candidates"]
        alignment["rows_with_same_thought_id"] = lookup["rows_total"]
        alignment["invalid_guidance_rows"] = lookup["invalid_guidance_rows"]
        alignment["chunk_match_candidates"] = len(candidates)

        selected_row, selection_mode = disambiguate_candidates(candidates, metadata)
        alignment["selection_mode"] = selection_mode

        if selected_row is None:
            if selection_mode == "no_candidate":
                alignment["status"] = "chunk_not_found_under_thought_id"
            else:
                alignment["status"] = "ambiguous_chunk_match"
        else:
            matched_doc = extract_doc_from_row(selected_row)
            alignment["status"] = "matched"
            alignment["selected_row"] = summarize_row(selected_row)

    enriched_entry = dict(entry)
    enriched_entry["retrieved_doc"] = matched_doc
    if add_alignment_debug:
        enriched_entry["retrieved_doc_alignment"] = alignment

    return enriched_entry, alignment["status"], str(alignment.get("selection_mode", ""))


def init_worker_for_file(
    env: str,
    source_file: str,
    max_thought_rows: int,
    add_alignment_debug: bool,
) -> None:
    global _WORKER_ENV, _WORKER_SOURCE_FILE, _WORKER_THOUGHT_CACHE, _WORKER_ADD_ALIGNMENT_DEBUG

    try:
        import lancedb  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "lancedb is required to run this script. Activate the correct env first."
        ) from exc

    db = lancedb.connect(ENV_TO_LANCEDB[env])
    table = db.open_table(env)
    _WORKER_ENV = env
    _WORKER_SOURCE_FILE = source_file
    _WORKER_THOUGHT_CACHE = ThoughtIdIndexCache(
        table=table, max_thought_rows=max_thought_rows
    )
    _WORKER_ADD_ALIGNMENT_DEBUG = add_alignment_debug


def process_entry_in_worker(
    task: Tuple[str, int, Dict[str, Any]]
) -> Tuple[str, int, Dict[str, Any], str, str]:
    query_group_key, in_group_idx, entry = task

    if _WORKER_THOUGHT_CACHE is None:
        raise RuntimeError("Worker is not initialized")

    enriched_entry, status, selection_mode = build_enriched_entry_for_item(
        entry=entry,
        thought_cache=_WORKER_THOUGHT_CACHE,
        add_alignment_debug=_WORKER_ADD_ALIGNMENT_DEBUG,
    )
    return query_group_key, in_group_idx, enriched_entry, status, selection_mode


def set_entry_value(
    data: Dict[str, Any],
    query_group_key: str,
    in_group_idx: int,
    enriched_entry: Dict[str, Any],
) -> None:
    value = data.get(query_group_key)
    if isinstance(value, list):
        value[in_group_idx] = enriched_entry
    elif isinstance(value, dict):
        data[query_group_key] = enriched_entry
    else:
        data[query_group_key] = enriched_entry


def process_file(
    input_path: Path,
    env: str,
    thought_cache: Optional[ThoughtIdIndexCache],
    max_entries: Optional[int],
    num_workers: int,
    max_thought_rows: int,
    add_alignment_debug: bool,
) -> Tuple[Dict[str, Any], Counter]:
    data = json.loads(input_path.read_text(encoding="utf-8"))

    stats: Counter = Counter()
    stats["query_groups"] = len(data)

    tasks: List[Tuple[str, int, Dict[str, Any]]] = []
    for entry_idx, (query_group_key, in_group_idx, entry) in enumerate(
        iter_entries(data)
    ):
        if max_entries is not None and entry_idx >= max_entries:
            break
        tasks.append((query_group_key, in_group_idx, entry))

    stats["entries_total"] = len(tasks)
    unique_thought_ids = len(
        {
            normalize_text((entry.get("metadata") or {}).get("thought_id", ""))
            for _, _, entry in tasks
            if isinstance(entry, dict)
        }
    )
    print(
        f"  entries={len(tasks)}, unique_thought_id={unique_thought_ids}, workers={num_workers}",
        flush=True,
    )
    if num_workers > 1:
        print(
            "  note: multiprocessing can be slower here because large records are "
            "serialized across processes and worker caches are not shared.",
            flush=True,
        )

    if num_workers <= 1:
        if thought_cache is None:
            raise ValueError("thought_cache is required for single-worker mode")
        iterator = tqdm(
            tasks,
            total=len(tasks),
            desc=f"{env}:{input_path.name}",
            unit="entry",
            dynamic_ncols=True,
        )
        for query_group_key, in_group_idx, entry in iterator:
            enriched_entry, status, selection_mode = build_enriched_entry_for_item(
                entry=entry,
                thought_cache=thought_cache,
                add_alignment_debug=add_alignment_debug,
            )
            set_entry_value(data, query_group_key, in_group_idx, enriched_entry)
            if status:
                stats[status] += 1
            if selection_mode == "ambiguous_fallback_first":
                stats["ambiguous_fallback_first"] += 1
    else:
        with Pool(
            processes=num_workers,
            initializer=init_worker_for_file,
            initargs=(env, str(input_path), max_thought_rows, add_alignment_debug),
        ) as pool:
            iterator = tqdm(
                pool.imap_unordered(process_entry_in_worker, tasks, chunksize=50),
                total=len(tasks),
                desc=f"{env}:{input_path.name}",
                unit="entry",
                dynamic_ncols=True,
            )
            for (
                query_group_key,
                in_group_idx,
                enriched_entry,
                status,
                selection_mode,
            ) in iterator:
                set_entry_value(data, query_group_key, in_group_idx, enriched_entry)
                if status:
                    stats[status] += 1
                if selection_mode == "ambiguous_fallback_first":
                    stats["ambiguous_fallback_first"] += 1

    return data, stats


def resolve_output_paths_for_input(
    input_path: Path,
    total_inputs: int,
    output_json_override: Optional[Path],
    summary_json_override: Optional[Path],
) -> Tuple[Path, Path]:
    if total_inputs > 1 and output_json_override is not None:
        raise ValueError(
            "--output-json override only supports a single input. "
            "For multiple inputs, omit it and per-input outputs will be generated."
        )
    if total_inputs > 1 and summary_json_override is not None:
        raise ValueError(
            "--summary-json override only supports a single input. "
            "For multiple inputs, omit it and per-input summaries will be generated."
        )

    if output_json_override is not None:
        out_json = output_json_override
    else:
        out_json = input_path.with_name(f"{input_path.stem}_chunk_aligned.json")

    if summary_json_override is not None:
        out_summary = summary_json_override
    else:
        out_summary = Path(str(out_json) + ".summary.json")

    return out_json, out_summary


def main() -> None:
    args = parse_args()

    try:
        import lancedb  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "lancedb is required to run this script. Activate the correct env first."
        ) from exc

    inputs = [p.resolve() for p in args.inputs]
    for p in inputs:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    env_to_inputs: Dict[str, List[Path]] = defaultdict(list)
    for input_path in inputs:
        env = infer_environment(input_path)
        env_to_inputs[env].append(input_path)

    env_resources: Dict[str, Dict[str, Any]] = {}
    if args.num_workers <= 1:
        for env in env_to_inputs:
            db = lancedb.connect(ENV_TO_LANCEDB[env])
            table = db.open_table(env)
            env_resources[env] = {
                "cache": ThoughtIdIndexCache(
                    table=table, max_thought_rows=args.max_thought_rows
                ),
            }

    file_summaries: Dict[str, Dict[str, Any]] = {}
    overall_stats: Counter = Counter()

    for env, input_files in env_to_inputs.items():
        thought_cache = None
        if args.num_workers <= 1:
            thought_cache = env_resources[env]["cache"]

        for input_path in input_files:
            print(f"Processing {input_path} (env={env}, workers={args.num_workers})")
            enriched_data, stats = process_file(
                input_path=input_path,
                env=env,
                thought_cache=thought_cache,
                max_entries=args.limit_per_file,
                num_workers=args.num_workers,
                max_thought_rows=args.max_thought_rows,
                add_alignment_debug=args.add_alignment_debug,
            )

            output_json, summary_json = resolve_output_paths_for_input(
                input_path=input_path,
                total_inputs=len(inputs),
                output_json_override=args.output_json,
                summary_json_override=args.summary_json,
            )
            output_json.parent.mkdir(parents=True, exist_ok=True)
            summary_json.parent.mkdir(parents=True, exist_ok=True)

            output_json.write_text(
                json.dumps(enriched_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            overall_stats.update(stats)

            summary_item = dict(stats)
            total = summary_item.get("entries_total", 0)
            matched = summary_item.get("matched", 0)
            summary_item["match_rate"] = (matched / total) if total else 0.0
            summary_item["environment"] = env
            summary_item["input"] = str(input_path)
            summary_item["output_json"] = str(output_json)
            summary_item["summary_json"] = str(summary_json)
            file_summaries[str(input_path)] = summary_item
            summary_json.write_text(
                json.dumps(summary_item, indent=2), encoding="utf-8"
            )

    overall = dict(overall_stats)
    overall_total = overall.get("entries_total", 0)
    overall_matched = overall.get("matched", 0)
    overall["match_rate"] = (overall_matched / overall_total) if overall_total else 0.0

    print("\nDone.")
    print("Per-input outputs:")
    for path, item in file_summaries.items():
        print(f"  - input: {path}")
        print(f"    output_json: {item['output_json']}")
        print(f"    summary_json: {item['summary_json']}")
    print(f"Entries total: {overall_total}")
    print(f"Matched: {overall_matched}")
    print(f"Match rate: {overall['match_rate']:.4f}")


if __name__ == "__main__":
    main()
