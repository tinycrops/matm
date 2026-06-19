#!/usr/bin/env python3
"""
Create a representative subset from the ALFWorld train pool.

Sampling strategy:
1. Primary stratification by task_type.
2. Allocate per-task quotas with Hamilton / largest-remainder method.
3. Within each task, allocate proportionally by floor_plan, then by num_steps.
4. Final picks in each bucket are seeded-random.

This is intended for selecting a train-set subset for downstream LTR data
collection while preserving the original train-pool distribution as closely
as possible.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Hashable, List, Sequence, Tuple


DEFAULT_INPUT = (
    "traj_retrieval/preprocess/new_splits/alfworld/train/index_source_train_all.json"
)
DEFAULT_BUDGET = 355
DEFAULT_SEED = 20260310


def _sort_key(value: Hashable) -> str:
    if isinstance(value, tuple):
        return "::".join(str(v) for v in value)
    return str(value)


def allocate_quotas(
    counts: Dict[Hashable, int],
    total: int,
    min_per_group: int = 0,
) -> Dict[Hashable, int]:
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


def build_step_bin_edges(rows: Sequence[dict]) -> List[int]:
    steps = sorted(int(r.get("num_steps", -1)) for r in rows)
    if not steps:
        return []

    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    edges: List[int] = []
    prev = None
    for q in quantiles:
        idx = min(len(steps) - 1, max(0, round((len(steps) - 1) * q)))
        value = steps[idx]
        if prev is None or value > prev:
            edges.append(value)
            prev = value
    return edges


def step_bin_label(num_steps: int, edges: Sequence[int]) -> str:
    if not edges:
        return "all"
    lower = None
    for edge in edges:
        if num_steps <= edge:
            if lower is None:
                return f"<= {edge}"
            return f"{lower + 1}-{edge}"
        lower = edge
    return f">= {edges[-1] + 1}"


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


def hierarchical_sample_task(
    entries: Sequence[dict],
    quota: int,
    step_edges: Sequence[int],
    rng: random.Random,
) -> List[dict]:
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
        by_step_bin: Dict[str, List[dict]] = defaultdict(list)
        for row in floor_entries:
            label = step_bin_label(int(row.get("num_steps", -1)), step_edges)
            by_step_bin[label].append(row)

        step_counts = {k: len(v) for k, v in by_step_bin.items()}
        step_quotas = allocate_quotas(step_counts, q_floor, min_per_group=0)

        for label in sorted(by_step_bin.keys(), key=_sort_key):
            q_step = step_quotas[label]
            if q_step <= 0:
                continue
            picked.extend(sample_entries(by_step_bin[label], q_step, rng))

    if len(picked) < quota:
        remaining = [row for row in entries if row not in picked]
        picked.extend(sample_entries(remaining, quota - len(picked), rng))
    elif len(picked) > quota:
        picked = sample_entries(picked, quota, rng)

    return picked


def summarize_distribution(rows: Sequence[dict], step_edges: Sequence[int]) -> dict:
    task_counts = Counter(str(r.get("task_type", "unknown")) for r in rows)
    floor_counts = Counter(str(r.get("floor_plan", "unknown")) for r in rows)
    steps_counts = Counter(int(r.get("num_steps", -1)) for r in rows)
    step_bin_counts = Counter(
        step_bin_label(int(r.get("num_steps", -1)), step_edges) for r in rows
    )
    task_floor_counts = Counter(
        (str(r.get("task_type", "unknown")), str(r.get("floor_plan", "unknown")))
        for r in rows
    )
    task_step_counts = Counter(
        (str(r.get("task_type", "unknown")), int(r.get("num_steps", -1))) for r in rows
    )
    task_step_bin_counts = Counter(
        (
            str(r.get("task_type", "unknown")),
            step_bin_label(int(r.get("num_steps", -1)), step_edges),
        )
        for r in rows
    )
    steps = [int(r.get("num_steps", -1)) for r in rows]

    return {
        "size": len(rows),
        "task_type_counts": {
            str(k): task_counts[k] for k in sorted(task_counts.keys(), key=_sort_key)
        },
        "floor_plan_counts": {
            str(k): floor_counts[k] for k in sorted(floor_counts.keys(), key=_sort_key)
        },
        "num_steps_counts": {
            str(k): steps_counts[k] for k in sorted(steps_counts.keys())
        },
        "step_bin_counts": {
            str(k): step_bin_counts[k]
            for k in sorted(step_bin_counts.keys(), key=_sort_key)
        },
        "task_floor_counts": {
            f"{k[0]}::{k[1]}": task_floor_counts[k]
            for k in sorted(task_floor_counts.keys(), key=_sort_key)
        },
        "task_num_steps_counts": {
            f"{k[0]}::{k[1]}": task_step_counts[k]
            for k in sorted(task_step_counts.keys(), key=_sort_key)
        },
        "task_step_bin_counts": {
            f"{k[0]}::{k[1]}": task_step_bin_counts[k]
            for k in sorted(task_step_bin_counts.keys(), key=_sort_key)
        },
        "unique_floor_plans": len(floor_counts),
        "median_num_steps": statistics.median(steps) if steps else None,
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
    full_rows: Sequence[dict],
    sample_rows: Sequence[dict],
    step_edges: Sequence[int],
) -> dict:
    full_task = Counter(str(r.get("task_type", "unknown")) for r in full_rows)
    full_floor = Counter(str(r.get("floor_plan", "unknown")) for r in full_rows)
    full_steps = Counter(int(r.get("num_steps", -1)) for r in full_rows)
    full_step_bin = Counter(
        step_bin_label(int(r.get("num_steps", -1)), step_edges) for r in full_rows
    )
    full_task_floor = Counter(
        (str(r.get("task_type", "unknown")), str(r.get("floor_plan", "unknown")))
        for r in full_rows
    )
    full_task_step = Counter(
        (str(r.get("task_type", "unknown")), int(r.get("num_steps", -1)))
        for r in full_rows
    )
    full_task_step_bin = Counter(
        (
            str(r.get("task_type", "unknown")),
            step_bin_label(int(r.get("num_steps", -1)), step_edges),
        )
        for r in full_rows
    )

    sample_task = Counter(str(r.get("task_type", "unknown")) for r in sample_rows)
    sample_floor = Counter(str(r.get("floor_plan", "unknown")) for r in sample_rows)
    sample_steps = Counter(int(r.get("num_steps", -1)) for r in sample_rows)
    sample_step_bin = Counter(
        step_bin_label(int(r.get("num_steps", -1)), step_edges) for r in sample_rows
    )
    sample_task_floor = Counter(
        (str(r.get("task_type", "unknown")), str(r.get("floor_plan", "unknown")))
        for r in sample_rows
    )
    sample_task_step = Counter(
        (str(r.get("task_type", "unknown")), int(r.get("num_steps", -1)))
        for r in sample_rows
    )
    sample_task_step_bin = Counter(
        (
            str(r.get("task_type", "unknown")),
            step_bin_label(int(r.get("num_steps", -1)), step_edges),
        )
        for r in sample_rows
    )

    return {
        "task_type_tvd": total_variation_distance(full_task, sample_task),
        "floor_plan_tvd": total_variation_distance(full_floor, sample_floor),
        "step_bin_tvd": total_variation_distance(full_step_bin, sample_step_bin),
        "num_steps_tvd": total_variation_distance(full_steps, sample_steps),
        "task_floor_tvd": total_variation_distance(full_task_floor, sample_task_floor),
        "task_step_bin_tvd": total_variation_distance(
            full_task_step_bin, sample_task_step_bin
        ),
        "task_num_steps_tvd": total_variation_distance(
            full_task_step, sample_task_step
        ),
        "median_num_steps_gap": abs(
            (
                statistics.median(int(r.get("num_steps", -1)) for r in full_rows)
                if full_rows
                else 0
            )
            - (
                statistics.median(int(r.get("num_steps", -1)) for r in sample_rows)
                if sample_rows
                else 0
            )
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a representative ALFWorld train subset."
    )
    parser.add_argument("--input", type=Path, default=Path(DEFAULT_INPUT))
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <input_dir>/samples_ltr_datasets.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Default: <output_stem>_report.json",
    )
    parser.add_argument(
        "--min-per-task",
        type=int,
        default=1,
        help="Minimum quota for each non-empty task_type when budget allows.",
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
        args.output = args.input.parent / "samples_ltr_datasets.json"
    if args.report is None:
        args.report = args.output.with_name(f"{args.output.stem}_report.json")

    rng = random.Random(args.seed)
    step_edges = build_step_bin_edges(rows)

    by_task: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_task[str(row.get("task_type", "unknown"))].append(row)

    task_counts = {k: len(v) for k, v in by_task.items()}
    min_per_task = args.min_per_task
    if min_per_task > 0 and args.budget < len(task_counts) * min_per_task:
        min_per_task = 0

    task_quotas = allocate_quotas(task_counts, args.budget, min_per_group=min_per_task)

    sampled_rows: List[dict] = []
    for task in sorted(by_task.keys(), key=_sort_key):
        quota = task_quotas[task]
        if quota <= 0:
            continue
        sampled_rows.extend(
            hierarchical_sample_task(by_task[task], quota, step_edges, rng)
        )

    if len(sampled_rows) != args.budget:
        raise RuntimeError(
            f"Internal error: sampled {len(sampled_rows)} rows, expected {args.budget}"
        )

    id_set = {id(r) for r in sampled_rows}
    sampled_rows = [r for r in rows if id(r) in id_set]

    report = {
        "input_file": str(args.input),
        "output_file": str(args.output),
        "budget": args.budget,
        "seed": args.seed,
        "strategy": {
            "primary_strata": "task_type",
            "quota_allocation": "largest_remainder_hamilton",
            "secondary_sampling": "within_task floor_plan -> step_bin -> seeded_random",
            "step_bin_edges": list(step_edges),
            "min_per_task_applied": min_per_task,
        },
        "primary_strata": {
            "counts": {
                str(k): task_counts[k]
                for k in sorted(task_counts.keys(), key=_sort_key)
            },
            "quotas": {
                str(k): task_quotas[k]
                for k in sorted(task_quotas.keys(), key=_sort_key)
            },
        },
        "source_summary": summarize_distribution(rows, step_edges),
        "sample_summary": summarize_distribution(sampled_rows, step_edges),
        "distribution_shift_tvd": compute_shift_metrics(rows, sampled_rows, step_edges),
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
    print("Task quotas:")
    for task in sorted(task_quotas.keys(), key=_sort_key):
        print(f"  {task}: {task_quotas[task]} / {task_counts[task]}")
    print("Distribution shift (TVD, lower is better):")
    for metric, value in report["distribution_shift_tvd"].items():
        print(f"  {metric}: {value:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
