#!/usr/bin/env python3
"""
Create a representative subset from ALFWorld official test split under a budget.

Sampling strategy (fully code-driven):
1. Primary stratification by (split_tag, task_type), where split_tag is inferred
   from game_file path (valid_seen -> seen, valid_unseen -> unseen).
2. Allocate per-stratum quotas with Hamilton / largest-remainder method.
   This preserves global proportions while hitting the exact budget.
3. Inside each stratum, allocate quota proportionally by floor_plan, then by
   num_steps. Final picks in each bucket are seeded-random.
4. Write sampled JSON + a report JSON with source/sample distributions and
   simple shift metrics (total variation distance).

Usage example:
    python traj_retrieval/preprocess/alfworld/sample_official_test_subset.py \
      --budget 60 \
      --seed 20260225
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Hashable, List, Sequence, Tuple


DEFAULT_INPUT = "traj_retrieval/preprocess/new_splits/alfworld/official_test/complete_seen_unseen.json"


def _sort_key(value: Hashable) -> str:
    if isinstance(value, tuple):
        return "::".join(str(v) for v in value)
    return str(value)


def infer_split_tag(game_file: str) -> str:
    game_file = (game_file or "").lower()
    if "/valid_seen/" in game_file:
        return "seen"
    if "/valid_unseen/" in game_file:
        return "unseen"
    if "unseen" in game_file:
        return "unseen"
    if "seen" in game_file:
        return "seen"
    return "unknown"


def allocate_quotas(
    counts: Dict[Hashable, int],
    total: int,
    min_per_group: int = 0,
) -> Dict[Hashable, int]:
    """
    Allocate integer quotas summing to `total` under per-group capacities.

    Uses largest remainder (Hamilton). When min_per_group > 0, first attempts to
    assign a minimum quota to each non-empty group if budget allows.
    """
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total}")
    if min_per_group < 0:
        raise ValueError(f"min_per_group must be >= 0, got {min_per_group}")

    keys = sorted(counts.keys(), key=_sort_key)
    capacities = {k: int(counts[k]) for k in keys}
    if any(v < 0 for v in capacities.values()):
        raise ValueError("counts must be non-negative")

    available = sum(capacities.values())
    if total > available:
        raise ValueError(f"total {total} exceeds available capacity {available}")

    quotas = {k: 0 for k in keys}

    if total == 0:
        return quotas

    eligible = [k for k in keys if capacities[k] > 0]
    if min_per_group > 0 and eligible:
        required = min_per_group * len(eligible)
        if required <= total:
            for k in eligible:
                add = min(min_per_group, capacities[k])
                quotas[k] += add
                capacities[k] -= add
            total -= sum(quotas.values())

    if total == 0:
        return quotas

    remaining_capacity = sum(capacities.values())
    if remaining_capacity == 0:
        return quotas

    raw = {k: total * capacities[k] / remaining_capacity for k in keys}
    base = {k: min(capacities[k], int(math.floor(raw[k]))) for k in keys}

    for k in keys:
        quotas[k] += base[k]
        capacities[k] -= base[k]

    leftover = total - sum(base.values())
    if leftover <= 0:
        return quotas

    ranked = sorted(
        keys,
        key=lambda k: (raw[k] - math.floor(raw[k]), counts[k], _sort_key(k)),
        reverse=True,
    )

    while leftover > 0:
        progressed = False
        for k in ranked:
            if leftover == 0:
                break
            if capacities[k] <= 0:
                continue
            quotas[k] += 1
            capacities[k] -= 1
            leftover -= 1
            progressed = True
        if not progressed:
            break

    if leftover != 0:
        raise RuntimeError("Failed to allocate all quota units")

    return quotas


def sample_entries(
    entries: Sequence[dict], quota: int, rng: random.Random
) -> List[dict]:
    if quota <= 0:
        return []
    if quota >= len(entries):
        return list(entries)
    shuffled = list(entries)
    rng.shuffle(shuffled)
    return shuffled[:quota]


def hierarchical_sample_stratum(
    entries: Sequence[dict],
    quota: int,
    rng: random.Random,
) -> List[dict]:
    """
    Sample within a stratum with proportional hierarchy:
      floor_plan -> num_steps -> random pick.
    """
    if quota <= 0:
        return []
    if quota >= len(entries):
        return list(entries)

    by_floor: Dict[str, List[dict]] = defaultdict(list)
    for row in entries:
        by_floor[str(row.get("floor_plan", "unknown"))].append(row)

    floor_counts = {k: len(v) for k, v in by_floor.items()}
    floor_quotas = allocate_quotas(floor_counts, quota, min_per_group=0)

    picked: List[dict] = []
    for floor in sorted(by_floor.keys(), key=_sort_key):
        q_floor = floor_quotas[floor]
        if q_floor <= 0:
            continue

        floor_entries = by_floor[floor]
        by_steps: Dict[int, List[dict]] = defaultdict(list)
        for row in floor_entries:
            by_steps[int(row.get("num_steps", -1))].append(row)

        step_counts = {k: len(v) for k, v in by_steps.items()}
        step_quotas = allocate_quotas(step_counts, q_floor, min_per_group=0)

        for step in sorted(by_steps.keys()):
            q_step = step_quotas[step]
            if q_step <= 0:
                continue
            picked.extend(sample_entries(by_steps[step], q_step, rng))

    if len(picked) < quota:
        remaining = [row for row in entries if row not in picked]
        picked.extend(sample_entries(remaining, quota - len(picked), rng))
    elif len(picked) > quota:
        picked = sample_entries(picked, quota, rng)

    return picked


def summarize_distribution(rows: Sequence[dict]) -> dict:
    split_counts = Counter(infer_split_tag(r.get("game_file", "")) for r in rows)
    task_counts = Counter(str(r.get("task_type", "unknown")) for r in rows)
    split_task_counts = Counter(
        (infer_split_tag(r.get("game_file", "")), str(r.get("task_type", "unknown")))
        for r in rows
    )
    steps_counts = Counter(int(r.get("num_steps", -1)) for r in rows)
    floor_counts = Counter(str(r.get("floor_plan", "unknown")) for r in rows)

    return {
        "size": len(rows),
        "split_counts": {
            str(k): split_counts[k] for k in sorted(split_counts.keys(), key=_sort_key)
        },
        "task_type_counts": {
            str(k): task_counts[k] for k in sorted(task_counts.keys(), key=_sort_key)
        },
        "split_task_counts": {
            f"{k[0]}::{k[1]}": split_task_counts[k]
            for k in sorted(split_task_counts.keys(), key=_sort_key)
        },
        "num_steps_counts": {
            str(k): steps_counts[k] for k in sorted(steps_counts.keys())
        },
        "unique_floor_plans": len(floor_counts),
    }


def total_variation_distance(
    full_counts: Counter,
    sample_counts: Counter,
) -> float:
    full_total = sum(full_counts.values())
    sample_total = sum(sample_counts.values())
    if full_total == 0 or sample_total == 0:
        return 0.0
    keys = set(full_counts.keys()) | set(sample_counts.keys())
    l1 = 0.0
    for key in keys:
        p = full_counts.get(key, 0) / full_total
        q = sample_counts.get(key, 0) / sample_total
        l1 += abs(p - q)
    return 0.5 * l1


def compute_shift_metrics(
    full_rows: Sequence[dict], sample_rows: Sequence[dict]
) -> dict:
    full_split = Counter(infer_split_tag(r.get("game_file", "")) for r in full_rows)
    full_task = Counter(str(r.get("task_type", "unknown")) for r in full_rows)
    full_split_task = Counter(
        (infer_split_tag(r.get("game_file", "")), str(r.get("task_type", "unknown")))
        for r in full_rows
    )
    full_steps = Counter(int(r.get("num_steps", -1)) for r in full_rows)

    sample_split = Counter(infer_split_tag(r.get("game_file", "")) for r in sample_rows)
    sample_task = Counter(str(r.get("task_type", "unknown")) for r in sample_rows)
    sample_split_task = Counter(
        (infer_split_tag(r.get("game_file", "")), str(r.get("task_type", "unknown")))
        for r in sample_rows
    )
    sample_steps = Counter(int(r.get("num_steps", -1)) for r in sample_rows)

    return {
        "split_tvd": total_variation_distance(full_split, sample_split),
        "task_type_tvd": total_variation_distance(full_task, sample_task),
        "split_task_tvd": total_variation_distance(full_split_task, sample_split_task),
        "num_steps_tvd": total_variation_distance(full_steps, sample_steps),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a representative ALFWorld official-test subset."
    )
    parser.add_argument("--input", type=Path, default=Path(DEFAULT_INPUT))
    parser.add_argument(
        "--budget",
        type=int,
        required=True,
        help="Number of samples to keep.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output sampled JSON path. Default: <input_dir>/sample_budget{B}_seed{S}.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Output report JSON path. Default: <output_stem>_report.json",
    )
    parser.add_argument(
        "--min-per-stratum",
        type=int,
        default=1,
        help=(
            "Minimum quota for each non-empty primary stratum when budget allows. "
            "Set to 0 to disable."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    with args.input.open("r") as f:
        rows = json.load(f)

    if not isinstance(rows, list):
        raise ValueError(f"Input must be a JSON list, got {type(rows).__name__}")

    total_rows = len(rows)
    if args.budget <= 0:
        raise ValueError("--budget must be positive")
    if args.budget > total_rows:
        raise ValueError(f"--budget {args.budget} exceeds total rows {total_rows}")

    if args.output is None:
        args.output = (
            args.input.parent / f"sample_budget{args.budget}_seed{args.seed}.json"
        )
    if args.report is None:
        args.report = args.output.with_name(f"{args.output.stem}_report.json")

    rng = random.Random(args.seed)

    strata: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for row in rows:
        key = (
            infer_split_tag(row.get("game_file", "")),
            str(row.get("task_type", "unknown")),
        )
        strata[key].append(row)

    stratum_counts = {k: len(v) for k, v in strata.items()}
    non_empty_strata = len(stratum_counts)
    min_per_stratum = args.min_per_stratum
    if min_per_stratum > 0 and args.budget < (non_empty_strata * min_per_stratum):
        min_per_stratum = 0

    stratum_quotas = allocate_quotas(
        stratum_counts,
        args.budget,
        min_per_group=min_per_stratum,
    )

    sampled_rows: List[dict] = []
    for key in sorted(strata.keys(), key=_sort_key):
        quota = stratum_quotas[key]
        if quota <= 0:
            continue
        sampled_rows.extend(hierarchical_sample_stratum(strata[key], quota, rng))

    if len(sampled_rows) != args.budget:
        raise RuntimeError(
            f"Internal error: sampled {len(sampled_rows)} rows, expected {args.budget}"
        )

    # Keep output stable and close to original order for easier diffing/debugging.
    id_set = {id(r) for r in sampled_rows}
    sampled_rows = [r for r in rows if id(r) in id_set]

    stratum_quota_json = {
        f"{k[0]}::{k[1]}": stratum_quotas[k]
        for k in sorted(stratum_quotas.keys(), key=_sort_key)
    }
    stratum_count_json = {
        f"{k[0]}::{k[1]}": stratum_counts[k]
        for k in sorted(stratum_counts.keys(), key=_sort_key)
    }

    report = {
        "input_file": str(args.input),
        "output_file": str(args.output),
        "budget": args.budget,
        "seed": args.seed,
        "strategy": {
            "primary_strata": "split_tag x task_type",
            "quota_allocation": "largest_remainder_hamilton",
            "secondary_sampling": "within_stratum floor_plan -> num_steps -> seeded_random",
            "min_per_stratum_applied": min_per_stratum,
        },
        "primary_strata": {
            "counts": stratum_count_json,
            "quotas": stratum_quota_json,
        },
        "source_summary": summarize_distribution(rows),
        "sample_summary": summarize_distribution(sampled_rows),
        "distribution_shift_tvd": compute_shift_metrics(rows, sampled_rows),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w") as f:
        json.dump(sampled_rows, f, indent=2)
    with args.report.open("w") as f:
        json.dump(report, f, indent=2)

    print("Sampling complete.")
    print(f"Input:  {args.input} ({len(rows)} rows)")
    print(f"Output: {args.output} ({len(sampled_rows)} rows)")
    print(f"Report: {args.report}")
    print("Primary strata quotas:")
    for key in sorted(stratum_quotas.keys(), key=_sort_key):
        print(f"  {key[0]}::{key[1]} -> {stratum_quotas[key]} / {stratum_counts[key]}")
    print("Distribution shift (TVD, lower is better):")
    for metric, value in report["distribution_shift_tvd"].items():
        print(f"  {metric}: {value:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
