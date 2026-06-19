#!/usr/bin/env python3
"""
Verify there is no overlap between WebArena train-only index source and eval sets.

Default usage:
    python traj_retrieval/preprocess/webarena/verify_no_leakage_train_only.py
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple


DEFAULT_INDEX_MANIFEST = (
    "traj_retrieval/preprocess/new_splits/webarena/train/index_source_train_augmented.json"
)
DEFAULT_EVALUATION_SET = "traj_retrieval/preprocess/new_splits/webarena/test.json"
DEFAULT_OUTPUT_REPORT = (
    "traj_retrieval/preprocess/new_splits/webarena/leakage_report.json"
)


def _load_json_list(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list")
    return data


def _pair_set(entries: List[Dict]) -> Set[Tuple[str, str]]:
    pairs: Set[Tuple[str, str]] = set()
    for item in entries:
        task_name = item.get("task_name")
        variation_id = item.get("variation_id")
        if task_name is None or variation_id is None:
            continue
        pairs.add((str(task_name), str(variation_id)))
    return pairs


def _task_id_set(entries: List[Dict]) -> Set[str]:
    values: Set[str] = set()
    for item in entries:
        task_id = item.get("task_id")
        if task_id is not None:
            values.add(str(task_id))
    return values


def _sample_pair_overlap(values: Set[Tuple[str, str]], limit: int = 10) -> List[Dict]:
    out = []
    for task_name, variation_id in sorted(values)[:limit]:
        out.append({"task_name": task_name, "variation_id": variation_id})
    return out


def _sample_values(values: Set[str], limit: int = 10) -> List[str]:
    return sorted(values)[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify WebArena train-only leakage.")
    parser.add_argument(
        "--index-manifest",
        default=DEFAULT_INDEX_MANIFEST,
        help="Path to train-only index source manifest JSON list.",
    )
    parser.add_argument(
        "--evaluation-set",
        action="append",
        default=[],
        help="Path to evaluation set JSON list. Can be repeated.",
    )
    parser.add_argument(
        "--output-report",
        default=DEFAULT_OUTPUT_REPORT,
        help="Path to write leakage report JSON.",
    )
    args = parser.parse_args()

    eval_paths = args.evaluation_set or [DEFAULT_EVALUATION_SET]
    index_manifest_path = Path(args.index_manifest)
    report_path = Path(args.output_report)

    if not index_manifest_path.exists():
        print(f"ERROR: Index manifest not found: {index_manifest_path}")
        return 1

    try:
        index_entries = _load_json_list(index_manifest_path)
    except Exception as exc:
        print(f"ERROR: Failed loading index manifest: {exc}")
        return 1

    index_pairs = _pair_set(index_entries)
    index_task_ids = _task_id_set(index_entries)

    report = {
        "created_at": datetime.now().isoformat(),
        "index_manifest": str(index_manifest_path),
        "index_entries": len(index_entries),
        "index_unique_pairs": len(index_pairs),
        "index_unique_task_ids": len(index_task_ids),
        "evaluations": [],
        "has_any_overlap": False,
    }

    for eval_path_str in eval_paths:
        eval_path = Path(eval_path_str)
        if not eval_path.exists():
            report["evaluations"].append(
                {"evaluation_set": str(eval_path), "error": "file_not_found"}
            )
            report["has_any_overlap"] = True
            continue

        try:
            eval_entries = _load_json_list(eval_path)
        except Exception as exc:
            report["evaluations"].append(
                {"evaluation_set": str(eval_path), "error": f"invalid_json_list: {exc}"}
            )
            report["has_any_overlap"] = True
            continue

        eval_pairs = _pair_set(eval_entries)
        eval_task_ids = _task_id_set(eval_entries)

        overlap_pairs = index_pairs.intersection(eval_pairs)
        overlap_task_ids = index_task_ids.intersection(eval_task_ids)
        has_overlap = bool(overlap_pairs or overlap_task_ids)
        report["has_any_overlap"] = report["has_any_overlap"] or has_overlap

        report["evaluations"].append(
            {
                "evaluation_set": str(eval_path),
                "evaluation_entries": len(eval_entries),
                "has_overlap": has_overlap,
                "pair_check": {
                    "index_unique_pairs": len(index_pairs),
                    "eval_unique_pairs": len(eval_pairs),
                    "overlap_count": len(overlap_pairs),
                    "overlap_samples": _sample_pair_overlap(overlap_pairs),
                },
                "task_id_check": {
                    "index_unique_task_ids": len(index_task_ids),
                    "eval_unique_task_ids": len(eval_task_ids),
                    "overlap_count": len(overlap_task_ids),
                    "overlap_samples": _sample_values(overlap_task_ids),
                },
            }
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"OK: Leakage report written: {report_path}")
    for item in report["evaluations"]:
        eval_name = item.get("evaluation_set", "unknown")
        if "error" in item:
            print(f"  ERROR {eval_name}: {item['error']}")
            continue
        marker = "ERROR" if item.get("has_overlap") else "OK"
        pair_overlap = item.get("pair_check", {}).get("overlap_count", 0)
        task_id_overlap = item.get("task_id_check", {}).get("overlap_count", 0)
        print(
            f"  {marker} {eval_name}: pair_overlap={pair_overlap}, "
            f"task_id_overlap={task_id_overlap}"
        )

    if report["has_any_overlap"]:
        print("ERROR: Leakage detected (or evaluation file issues).")
        return 2

    print("OK: No overlap detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
