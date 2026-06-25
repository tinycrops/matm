#!/usr/bin/env python3
"""
Stage ALFWorld public expert trajectories from the MATM HuggingFace dataset.

The canonical train manifest points at local run_gold JSON files, but the
released trace data lives in toeunkim/matm-trajectories as parquet. This script
reconstructs the expected run_gold files by joining those parquet rows to the
manifest metadata that contains game_file, floor_plan, and trajectory_file.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
from huggingface_hub import hf_hub_download


REPO_ID = "toeunkim/matm-trajectories"
PARQUET_FILENAME = "alfworld/prepopulation.parquet"
DEFAULT_MANIFEST = Path(
    "traj_retrieval/preprocess/new_splits/alfworld/train/index_source_train_all.json"
)
DEFAULT_LOCAL_DIR = Path("hf_dataset")


def _json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, float) and pd.isna(value):
        return default
    if isinstance(value, str):
        if value == "NA":
            return default
        return json.loads(value)
    return value


def _normalise_step(step: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "action": step.get("action", ""),
        "observation": step.get("observation", ""),
        "reasoning": step.get("reasoning"),
        "isCompleted": step.get("isCompleted"),
        "inventory": step.get("inventory"),
        "reward": step.get("reward"),
        "score": step.get("score"),
        "url": step.get("url"),
    }


def _with_throwaway_tail(trajectory: List[Dict[str, Any]], text_actions: List[str]) -> List[Dict[str, Any]]:
    staged = [_normalise_step(step) for step in trajectory]
    if len(text_actions) > len(staged):
        for action in text_actions[len(staged) :]:
            staged.append(
                {
                    "action": action,
                    "observation": "",
                    "reasoning": None,
                    "isCompleted": None,
                    "inventory": None,
                    "reward": None,
                    "score": None,
                    "url": None,
                }
            )
    return staged


def _dedupe_prepopulation_rows(df: pd.DataFrame, manifest_task_ids: Iterable[str]) -> pd.DataFrame:
    manifest_task_id_set = set(manifest_task_ids)
    df = df[df["task_id"].isin(manifest_task_id_set)].copy()
    df = df.sort_values(
        by=["task_id", "num_steps", "trajectory"],
        kind="mergesort",
    )
    return df.drop_duplicates(subset=["task_id"], keep="first")


def stage_hf_prepopulation(
    *,
    manifest_path: Path,
    parquet_path: Path,
    force: bool,
) -> Dict[str, int]:
    manifest = pd.read_json(manifest_path)
    required_manifest_cols = {
        "task_id",
        "task_type",
        "variation_id",
        "trajectory_file",
        "game_file",
        "floor_plan",
        "goal",
        "num_steps",
    }
    missing_manifest_cols = required_manifest_cols - set(manifest.columns)
    if missing_manifest_cols:
        raise ValueError(
            f"Manifest is missing required columns: {sorted(missing_manifest_cols)}"
        )

    prepopulation = pd.read_parquet(parquet_path)
    prepopulation = prepopulation[
        (prepopulation["environment"] == "alfworld")
        & (prepopulation["source_type"] == "public_expert")
        & (prepopulation["fold"] == "train")
        & (prepopulation["success"] == True)
    ].copy()
    prepopulation = _dedupe_prepopulation_rows(prepopulation, manifest["task_id"])

    merged = manifest.merge(
        prepopulation,
        on=["task_id", "task_type"],
        how="left",
        suffixes=("_manifest", ""),
    )
    missing_rows = merged[merged["trajectory"].isna()]
    if not missing_rows.empty:
        examples = ", ".join(missing_rows["task_id"].head(5).tolist())
        raise RuntimeError(
            f"Missing {len(missing_rows)} manifest task_ids in {parquet_path}: {examples}"
        )

    written = 0
    skipped = 0
    for _, row in merged.iterrows():
        output_path = Path(row["trajectory_file"])
        if output_path.exists() and not force:
            skipped += 1
            continue

        trajectory = _json_loads(row["trajectory"], [])
        text_actions = _json_loads(row.get("text_actions"), [])
        pddl_params = _json_loads(row.get("pddl_params"), {})
        high_level_descriptions = _json_loads(row.get("high_level_descriptions"), [])

        payload = {
            "task_id": row["task_id"],
            "task_type": row["task_type"],
            "variation_id": row["variation_id"],
            "split": "train",
            "fold": "train",
            "floor_plan": row["floor_plan"],
            "goal": row["goal_manifest"],
            "game_file": row["game_file"],
            "num_steps": int(row["num_steps_manifest"]),
            "success": True,
            "done": True,
            "source_type": "public_expert",
            "source_dataset": REPO_ID,
            "source_dataset_file": PARQUET_FILENAME,
            "text_actions": text_actions,
            "pddl_params": pddl_params,
            "high_level_descriptions": high_level_descriptions,
            "trajectory": _with_throwaway_tail(trajectory, text_actions),
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written += 1

    return {
        "manifest_rows": len(manifest),
        "prepopulation_rows": len(prepopulation),
        "written": written,
        "skipped_existing": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage ALFWorld MATM HuggingFace prepopulation traces as run_gold JSON."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--parquet", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    parquet_path = args.parquet
    if parquet_path is None:
        parquet_path = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=PARQUET_FILENAME,
                local_dir=str(args.local_dir),
            )
        )

    summary = stage_hf_prepopulation(
        manifest_path=args.manifest,
        parquet_path=parquet_path,
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
