#!/usr/bin/env python3
"""
Sample a representative subset from WebArena test split (100-task no-leakage pool).

Sampling strategy:
1. Primary stratification by site_combo = tuple(sorted(sites)).
2. Allocate site_combo quotas with largest-remainder (Hamilton), with optional
   minimum-per-combo guarantee when budget allows.
3. Within each combo, sample by intent_template_id using rare-first round robin:
   lower global intent frequency gets priority; one sample per intent each round.
4. Write sampled JSON + report JSON with distribution summaries and TVD metrics.

Usage example:
    python traj_retrieval/preprocess/webarena/sample_test_subset.py \
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


DEFAULT_INPUT = "traj_retrieval/preprocess/new_splits/webarena/test.json"


def normalize_site_combo(sites: Sequence[str]) -> Tuple[str, ...]:
    return tuple(sorted(str(site) for site in sites))


def site_combo_label(combo: Tuple[str, ...]) -> str:
    return "|".join(combo)


def _sort_key(value: Hashable) -> str:
    if isinstance(value, tuple):
        return "::".join(str(v) for v in value)
    return str(value)


def allocate_quotas(
    counts: Dict[Hashable, int],
    total: int,
    min_per_group: int = 0,
) -> Dict[Hashable, int]:
    """
    Allocate integer quotas summing to `total` under per-group capacities.

    Uses largest remainder (Hamilton). min_per_group is applied only when budget
    allows and capacity exists.
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


def sample_indices_random(
    indices: Sequence[int], quota: int, rng: random.Random
) -> List[int]:
    if quota <= 0:
        return []
    if quota >= len(indices):
        return list(indices)
    shuffled = list(indices)
    rng.shuffle(shuffled)
    return shuffled[:quota]


def sample_within_combo_rare_first(
    combo_indices: Sequence[int],
    quota: int,
    rows: Sequence[dict],
    global_intent_counts: Counter,
    rng: random.Random,
) -> List[int]:
    """
    Rare-first round-robin sampling by intent within one site_combo.

    - Intents are prioritized by global frequency ascending.
    - One sample per intent per round to avoid head-intent collapse.
    """
    if quota <= 0:
        return []
    if quota >= len(combo_indices):
        return list(combo_indices)

    by_intent: Dict[int, List[int]] = defaultdict(list)
    for idx in combo_indices:
        intent = int(rows[idx].get("intent_template_id", -1))
        by_intent[intent].append(idx)

    for intent in by_intent:
        rng.shuffle(by_intent[intent])

    tie = {intent: rng.random() for intent in by_intent.keys()}
    sorted_intents = sorted(
        by_intent.keys(),
        key=lambda intent: (
            global_intent_counts[intent],
            len(by_intent[intent]),
            tie[intent],
        ),
    )

    selected: List[int] = []
    while len(selected) < quota:
        progressed = False
        for intent in sorted_intents:
            if len(selected) >= quota:
                break
            bucket = by_intent[intent]
            if not bucket:
                continue
            selected.append(bucket.pop())
            progressed = True
        if not progressed:
            break

    if len(selected) < quota:
        selected_set = set(selected)
        remaining = [idx for idx in combo_indices if idx not in selected_set]
        rng.shuffle(remaining)
        selected.extend(remaining[: quota - len(selected)])
    elif len(selected) > quota:
        selected = sample_indices_random(selected, quota, rng)

    return selected


def summarize_distribution(rows: Sequence[dict]) -> dict:
    intent_counts = Counter(int(r.get("intent_template_id", -1)) for r in rows)
    site_combo_counts = Counter(normalize_site_combo(r.get("sites", [])) for r in rows)
    site_incidence_counts = Counter()
    multi_site_tasks = 0
    for r in rows:
        sites = normalize_site_combo(r.get("sites", []))
        if len(sites) > 1:
            multi_site_tasks += 1
        for site in sites:
            site_incidence_counts[site] += 1

    return {
        "size": len(rows),
        "unique_intents": len(intent_counts),
        "intent_template_counts": {
            str(k): intent_counts[k] for k in sorted(intent_counts.keys())
        },
        "site_combo_counts": {
            site_combo_label(k): site_combo_counts[k]
            for k in sorted(site_combo_counts.keys(), key=_sort_key)
        },
        "site_incidence_counts": {
            str(site): site_incidence_counts[site]
            for site in sorted(site_incidence_counts.keys())
        },
        "multi_site_tasks": multi_site_tasks,
        "multi_site_rate": (multi_site_tasks / len(rows)) if rows else 0.0,
        "avg_sites_per_task": (
            sum(site_incidence_counts.values()) / len(rows) if rows else 0.0
        ),
    }


def total_variation_distance(full_counts: Counter, sample_counts: Counter) -> float:
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
    full_intent = Counter(int(r.get("intent_template_id", -1)) for r in full_rows)
    sample_intent = Counter(int(r.get("intent_template_id", -1)) for r in sample_rows)

    full_combo = Counter(normalize_site_combo(r.get("sites", [])) for r in full_rows)
    sample_combo = Counter(
        normalize_site_combo(r.get("sites", [])) for r in sample_rows
    )

    full_site_inc = Counter()
    sample_site_inc = Counter()
    for r in full_rows:
        for site in normalize_site_combo(r.get("sites", [])):
            full_site_inc[site] += 1
    for r in sample_rows:
        for site in normalize_site_combo(r.get("sites", [])):
            sample_site_inc[site] += 1

    return {
        "site_combo_tvd": total_variation_distance(full_combo, sample_combo),
        "site_incidence_tvd": total_variation_distance(full_site_inc, sample_site_inc),
        "intent_tvd": total_variation_distance(full_intent, sample_intent),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a representative WebArena subset from test.json (no-leakage pool)."
    )
    parser.add_argument("--input", type=Path, default=Path(DEFAULT_INPUT))
    parser.add_argument(
        "--budget", type=int, required=True, help="Number of tasks to keep."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-per-combo",
        type=int,
        default=1,
        help="Minimum samples per site_combo when budget allows. Set to 0 to disable.",
    )
    parser.add_argument(
        "--site-combo-tvd-threshold",
        type=float,
        default=0.12,
        help="Pass threshold for site_combo TVD.",
    )
    parser.add_argument(
        "--site-incidence-tvd-threshold",
        type=float,
        default=0.10,
        help="Pass threshold for site incidence TVD.",
    )
    parser.add_argument(
        "--intent-tvd-threshold",
        type=float,
        default=0.35,
        help="Pass threshold for intent_template TVD (long-tail on 100-task pool).",
    )
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

    by_combo: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        combo = normalize_site_combo(row.get("sites", []))
        by_combo[combo].append(idx)

    combo_counts = {combo: len(idxs) for combo, idxs in by_combo.items()}
    num_combos = len(combo_counts)

    min_per_combo = max(args.min_per_combo, 0)
    if num_combos > 0:
        min_per_combo = min(min_per_combo, args.budget // num_combos)

    combo_quotas = allocate_quotas(
        combo_counts, args.budget, min_per_group=min_per_combo
    )
    global_intent_counts = Counter(int(r.get("intent_template_id", -1)) for r in rows)

    selected_indices: List[int] = []
    combo_intent_quota_map: Dict[str, Dict[str, int]] = {}

    for combo in sorted(by_combo.keys(), key=_sort_key):
        quota = combo_quotas[combo]
        if quota <= 0:
            continue

        combo_indices = by_combo[combo]
        selected = sample_within_combo_rare_first(
            combo_indices=combo_indices,
            quota=quota,
            rows=rows,
            global_intent_counts=global_intent_counts,
            rng=rng,
        )
        selected_indices.extend(selected)

        # Record sampled intent breakdown in this combo for transparency.
        sampled_intent = Counter(
            int(rows[idx].get("intent_template_id", -1)) for idx in selected
        )
        combo_intent_quota_map[site_combo_label(combo)] = {
            str(intent): sampled_intent[intent]
            for intent in sorted(sampled_intent.keys())
        }

    deduped = []
    seen = set()
    for idx in selected_indices:
        if idx in seen:
            continue
        seen.add(idx)
        deduped.append(idx)
    selected_indices = deduped

    if len(selected_indices) < args.budget:
        remaining = [
            idx for idx in range(total_rows) if idx not in set(selected_indices)
        ]
        selected_indices.extend(
            sample_indices_random(remaining, args.budget - len(selected_indices), rng)
        )
    elif len(selected_indices) > args.budget:
        selected_indices = sample_indices_random(selected_indices, args.budget, rng)

    selected_indices = sorted(selected_indices)
    if len(selected_indices) != args.budget:
        raise RuntimeError(
            f"Internal error: sampled {len(selected_indices)} rows, expected {args.budget}"
        )

    sample_rows = [rows[idx] for idx in selected_indices]

    source_summary = summarize_distribution(rows)
    sample_summary = summarize_distribution(sample_rows)
    shifts = compute_shift_metrics(rows, sample_rows)

    full_combos = set(normalize_site_combo(r.get("sites", [])) for r in rows)
    sample_combos = set(normalize_site_combo(r.get("sites", [])) for r in sample_rows)
    missing_combos = sorted(site_combo_label(c) for c in (full_combos - sample_combos))
    budget_allows_full_combo_coverage = args.budget >= len(full_combos)

    full_intents = set(int(r.get("intent_template_id", -1)) for r in rows)
    sample_intents = set(int(r.get("intent_template_id", -1)) for r in sample_rows)
    intent_coverage = (len(sample_intents) / len(full_intents)) if full_intents else 0.0

    multi_site_gap = abs(
        sample_summary["multi_site_rate"] - source_summary["multi_site_rate"]
    )

    quality_checks = {
        "site_combo_coverage": {
            "covered_combos": len(sample_combos),
            "total_combos": len(full_combos),
            "budget_allows_full_coverage": budget_allows_full_combo_coverage,
            "missing_combos": missing_combos,
            "passed": (not budget_allows_full_combo_coverage)
            or (len(missing_combos) == 0),
        },
        "distribution": {
            "site_combo_tvd": shifts["site_combo_tvd"],
            "site_combo_threshold": args.site_combo_tvd_threshold,
            "site_combo_passed": shifts["site_combo_tvd"]
            <= args.site_combo_tvd_threshold,
            "site_incidence_tvd": shifts["site_incidence_tvd"],
            "site_incidence_threshold": args.site_incidence_tvd_threshold,
            "site_incidence_passed": shifts["site_incidence_tvd"]
            <= args.site_incidence_tvd_threshold,
            "intent_tvd": shifts["intent_tvd"],
            "intent_threshold": args.intent_tvd_threshold,
            "intent_passed": shifts["intent_tvd"] <= args.intent_tvd_threshold,
        },
        "intent_coverage": {
            "source_unique_intents": len(full_intents),
            "sample_unique_intents": len(sample_intents),
            "coverage_ratio": intent_coverage,
        },
        "multi_site_rate": {
            "source": source_summary["multi_site_rate"],
            "sample": sample_summary["multi_site_rate"],
            "abs_gap": multi_site_gap,
        },
    }

    report = {
        "input_file": str(args.input),
        "output_file": str(args.output),
        "budget": args.budget,
        "seed": args.seed,
        "strategy": {
            "primary_strata": "site_combo",
            "quota_allocation": "largest_remainder_hamilton",
            "min_per_combo_applied": min_per_combo,
            "secondary_sampling": "within_combo rare_first_round_robin_by_intent",
        },
        "site_combo_quotas": {
            site_combo_label(combo): combo_quotas[combo]
            for combo in sorted(combo_quotas.keys(), key=_sort_key)
        },
        "combo_sampled_intent_breakdown": {
            combo: combo_intent_quota_map[combo]
            for combo in sorted(combo_intent_quota_map.keys())
        },
        "source_summary": source_summary,
        "sample_summary": sample_summary,
        "distribution_shift_tvd": shifts,
        "quality_checks": quality_checks,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w") as f:
        json.dump(sample_rows, f, indent=2)
    with args.report.open("w") as f:
        json.dump(report, f, indent=2)

    print("Sampling complete.")
    print(f"Input:  {args.input} ({len(rows)} rows)")
    print(f"Output: {args.output} ({len(sample_rows)} rows)")
    print(f"Report: {args.report}")
    print("Site-combo quotas (sample / source):")
    for combo in sorted(combo_quotas.keys(), key=_sort_key):
        label = site_combo_label(combo)
        print(f"  {label}: {combo_quotas[combo]} / {combo_counts[combo]}")
    print("Distribution shift (TVD, lower is better):")
    for metric, value in shifts.items():
        print(f"  {metric}: {value:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
