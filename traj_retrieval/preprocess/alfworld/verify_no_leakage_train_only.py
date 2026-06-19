#!/usr/bin/env python3
"""
Verify there is no overlap between train-only index source and evaluation sets.

Default usage checks the new ALFWorld train-only manifest against official test:
    python traj_retrieval/preprocess/alfworld/verify_no_leakage_train_only.py
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set


DEFAULT_INDEX_MANIFEST = (
    "traj_retrieval/preprocess/new_splits/alfworld/train/index_source_train_all.json"
)
DEFAULT_EVALUATION_SET = "traj_retrieval/preprocess/new_splits/alfworld/official_test/complete_seen_unseen.json"
DEFAULT_OUTPUT_REPORT = (
    "traj_retrieval/preprocess/new_splits/alfworld/leakage_report.json"
)


def _normalize_path(value: str) -> str:
    if not value:
        return ""
    # Keep deterministic normalization while tolerating non-existing files.
    return os.path.normpath(value)


def _load_json_list(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list")
    return data


def _collect_values(entries: List[Dict], key: str) -> Set[str]:
    values: Set[str] = set()
    for item in entries:
        value = item.get(key)
        if value is None:
            continue
        if key == "game_file":
            value = _normalize_path(str(value))
        else:
            value = str(value)
        if value:
            values.add(value)
    return values


def _sample_overlap(
    index_entries: List[Dict],
    eval_entries: List[Dict],
    field: str,
    overlap_values: Set[str],
    limit: int = 10,
) -> List[Dict]:
    if not overlap_values:
        return []

    overlap_values = set(list(overlap_values)[:limit])
    samples: List[Dict] = []
    for eval_item in eval_entries:
        raw_value = eval_item.get(field)
        if raw_value is None:
            continue
        if field == "game_file":
            normalized = _normalize_path(str(raw_value))
        else:
            normalized = str(raw_value)
        if normalized in overlap_values:
            samples.append(
                {
                    "field": field,
                    "value": normalized,
                    "evaluation_item": eval_item,
                }
            )
        if len(samples) >= limit:
            break

    # If eval side has sparse fields, also sample from index side.
    if not samples:
        for index_item in index_entries:
            raw_value = index_item.get(field)
            if raw_value is None:
                continue
            if field == "game_file":
                normalized = _normalize_path(str(raw_value))
            else:
                normalized = str(raw_value)
            if normalized in overlap_values:
                samples.append(
                    {
                        "field": field,
                        "value": normalized,
                        "index_item": index_item,
                    }
                )
            if len(samples) >= limit:
                break
    return samples


def _check_overlap(index_entries: List[Dict], eval_entries: List[Dict]) -> Dict:
    checks: Dict[str, Dict] = {}
    for field in ["variation_id", "game_file", "task_id"]:
        index_values = _collect_values(index_entries, field)
        eval_values = _collect_values(eval_entries, field)
        overlap = index_values.intersection(eval_values)
        checks[field] = {
            "index_unique": len(index_values),
            "eval_unique": len(eval_values),
            "overlap_count": len(overlap),
            "overlap_samples": _sample_overlap(
                index_entries, eval_entries, field, overlap
            ),
        }
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify train-only index leakage.")
    parser.add_argument(
        "--index-manifest",
        default=DEFAULT_INDEX_MANIFEST,
        help="Path to train-only index source manifest JSON list.",
    )
    parser.add_argument(
        "--evaluation-set",
        action="append",
        default=[],
        help=(
            "Path to evaluation set JSON list. "
            "Can be repeated. Defaults to official complete_seen_unseen."
        ),
    )
    parser.add_argument(
        "--output-report",
        default=DEFAULT_OUTPUT_REPORT,
        help="Path to write leakage report JSON.",
    )
    args = parser.parse_args()

    eval_paths = args.evaluation_set or [DEFAULT_EVALUATION_SET]
    manifest_path = Path(args.index_manifest)
    output_report_path = Path(args.output_report)

    if not manifest_path.exists():
        print(f"ERROR: Index manifest not found: {manifest_path}")
        return 1

    try:
        index_entries = _load_json_list(manifest_path)
    except Exception as exc:
        print(f"ERROR: Failed loading index manifest: {exc}")
        return 1

    report = {
        "created_at": datetime.now().isoformat(),
        "index_manifest": str(manifest_path),
        "index_entries": len(index_entries),
        "evaluations": [],
        "has_any_overlap": False,
    }

    for eval_path_str in eval_paths:
        eval_path = Path(eval_path_str)
        if not eval_path.exists():
            report["evaluations"].append(
                {
                    "evaluation_set": str(eval_path),
                    "error": "file_not_found",
                }
            )
            report["has_any_overlap"] = True
            continue

        try:
            eval_entries = _load_json_list(eval_path)
        except Exception as exc:
            report["evaluations"].append(
                {
                    "evaluation_set": str(eval_path),
                    "error": f"invalid_json_list: {exc}",
                }
            )
            report["has_any_overlap"] = True
            continue

        checks = _check_overlap(index_entries, eval_entries)
        has_overlap = any(item["overlap_count"] > 0 for item in checks.values())
        report["has_any_overlap"] = report["has_any_overlap"] or has_overlap

        report["evaluations"].append(
            {
                "evaluation_set": str(eval_path),
                "evaluation_entries": len(eval_entries),
                "has_overlap": has_overlap,
                "checks": checks,
            }
        )

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with output_report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"OK: Leakage report written: {output_report_path}")
    for item in report["evaluations"]:
        eval_name = item.get("evaluation_set", "unknown")
        if "error" in item:
            print(f"  ERROR {eval_name}: {item['error']}")
            continue
        status = "OVERLAP" if item.get("has_overlap") else "OK"
        marker = "ERROR" if item.get("has_overlap") else "OK"
        print(f"  {marker} {eval_name}: {status}")
        checks = item.get("checks", {})
        print(
            "     overlaps -> "
            f"variation_id={checks.get('variation_id', {}).get('overlap_count', 0)}, "
            f"game_file={checks.get('game_file', {}).get('overlap_count', 0)}, "
            f"task_id={checks.get('task_id', {}).get('overlap_count', 0)}"
        )

    if report["has_any_overlap"]:
        print("ERROR: Leakage detected (or evaluation file issues).")
        return 2

    print("OK: No overlap detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
