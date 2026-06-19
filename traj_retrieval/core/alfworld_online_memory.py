#!/usr/bin/env python3
"""
Online memory writer for ALFWorld runtime trajectories.

This appends completed ALFWorld episode trajectories into an existing LanceDB
table using the same step-level entry template as the offline index, while
preserving source-model provenance through `agent_type` and `metadata.model_name`.
"""

from __future__ import annotations

import fcntl
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import lancedb
from sentence_transformers import SentenceTransformer

from traj_retrieval.preprocess.alfworld.trajectory_entry import (
    AGENT_FRAMEWORK,
    ENVIRONMENT,
    MAX_CONTEXT_STEPS,
    MAX_GUIDANCE_STEPS,
    STORAGE_STRATEGY,
    VERSION,
)


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat()


def _parse_bool(raw: str | None, default: bool = False) -> bool:
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _sql_escape(value: str) -> str:
    return value.replace("'", "''")


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


class AlfworldOnlineMemoryWriter:
    """Append runtime ALFWorld trajectories to a LanceDB table."""

    def __init__(
        self,
        indices_dir: str,
        table_name: str = "alfworld",
        embedding_model_name: str = "intfloat/e5-base",
        run_number: int = 1,
        lock_path: Optional[str] = None,
        dedup_within_model: bool = True,
    ):
        self.indices_dir = indices_dir
        self.table_name = table_name
        self.embedding_model_name = embedding_model_name
        self.run_number = int(run_number)
        self.dedup_within_model = dedup_within_model
        self.lock_path = Path(
            lock_path or (Path(indices_dir) / ".alfworld_online_memory.lock")
        )
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.touch(exist_ok=True)

        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def _open_table(self):
        """
        Open a fresh LanceDB table handle.

        We do this per commit instead of caching the table object across commits.
        Multiple jobs append to the same table, and cached handles can become
        stale after another process writes a newer manifest version.
        """
        db = lancedb.connect(self.indices_dir)
        table = db.open_table(self.table_name)
        return db, table

    def count_rows(self) -> int:
        _, table = self._open_table()
        return table.count_rows()

    def _infer_split(self, game_file: str) -> str:
        parts = Path(game_file).parts
        for idx, part in enumerate(parts):
            if part == "json_2.1.1" and idx + 1 < len(parts):
                return parts[idx + 1]
        return "train"

    def _build_context(self, steps: List[Dict[str, Any]], current_step_idx: int) -> str:
        context_parts: List[str] = []
        start_idx = max(0, current_step_idx - MAX_CONTEXT_STEPS)
        for j in range(start_idx, current_step_idx):
            prev_step = steps[j]
            prev_obs = prev_step.get("observation", "")
            prev_action = prev_step.get("action", "")
            context_parts.append(f"observation: {prev_obs} | action: {prev_action}")
        return " ; ".join(context_parts) if context_parts else ""

    def _build_guidance(
        self, steps: List[Dict[str, Any]], current_step_idx: int
    ) -> List[Dict[str, Any]]:
        guidance_steps: List[Dict[str, Any]] = []
        end_idx = min(current_step_idx + MAX_GUIDANCE_STEPS, len(steps))
        for j in range(current_step_idx, end_idx):
            step = steps[j]
            guidance_steps.append(
                {
                    "action": step.get("action", ""),
                    "observation": step.get("observation", ""),
                    "score": _float_value(
                        step.get("score", step.get("reward", 0.0)), 0.0
                    ),
                }
            )
        return guidance_steps

    def _entry_exists(
        self, table, entry: Dict[str, Any], source_model_label: str
    ) -> bool:
        conditions = [
            f"key_raw_goal = '{_sql_escape(entry['key_raw_goal'])}'",
            f"key_raw_state = '{_sql_escape(entry['key_raw_state'])}'",
            f"key_raw_context = '{_sql_escape(entry['key_raw_context'])}'",
            f"key_raw_progress = '{_sql_escape(entry['key_raw_progress'])}'",
        ]
        if entry["key_raw_constraints"] is None:
            conditions.append("key_raw_constraints IS NULL")
        else:
            conditions.append(
                f"key_raw_constraints = '{_sql_escape(str(entry['key_raw_constraints']))}'"
            )

        if self.dedup_within_model:
            conditions.append(f"agent_type = '{_sql_escape(source_model_label)}'")

        sql = " AND ".join(conditions)
        rows = table.search().where(sql).limit(1).to_list()
        return bool(rows)

    def _build_entries(
        self,
        *,
        trajectory: Dict[str, Any],
        episode_metadata: Dict[str, Any],
        source_model_label: str,
    ) -> List[Dict[str, Any]]:
        steps = list(trajectory.get("steps", []))
        if not steps:
            return []

        task_name = (
            episode_metadata.get("task_type")
            or trajectory.get("task")
            or "unknown_task"
        )
        variation_idx = (
            episode_metadata.get("variation_id")
            or trajectory.get("variation")
            or "unknown_variation"
        )
        task_id = episode_metadata.get("task_id") or f"runtime_{uuid.uuid4()}"
        goal = trajectory.get("goal_text") or episode_metadata.get("goal") or ""
        floor_plan = episode_metadata.get("floor_plan") or "unknown_floor_plan"
        game_file = episode_metadata.get("game_file") or ""
        split = self._infer_split(game_file)
        is_successful = (
            bool(trajectory.get("done", False))
            and _float_value(trajectory.get("final_score", 0.0), 0.0) > 0.0
        )
        reasoning_present = any(bool(step.get("reasoning")) for step in steps)
        trajectory_id = f"{source_model_label}-{uuid.uuid4()}"
        timestamp = _utc_now_iso()

        entries: List[Dict[str, Any]] = []
        for idx, current_step in enumerate(steps):
            observation = current_step.get("observation", "")
            state_repr = f"observation: {observation}"
            context_repr = self._build_context(steps, idx)
            progress_repr = f"step_till_now: {idx}"
            full_key = (
                f"goal: {goal} | state: {state_repr} | "
                f"context: {context_repr} | progress: {progress_repr}"
            )

            goal_embed = self.model.encode(goal).tolist()
            state_embed = self.model.encode(state_repr).tolist()
            context_embed = (
                self.model.encode(context_repr).tolist()
                if context_repr
                else [0.0] * len(goal_embed)
            )
            key_embed = self.model.encode(full_key).tolist()

            if idx + 1 < len(steps):
                next_reward = _float_value(
                    steps[idx + 1].get("score", steps[idx + 1].get("reward", 0.0)),
                    0.0,
                )
            else:
                next_reward = _float_value(
                    trajectory.get(
                        "final_score",
                        current_step.get("score", current_step.get("reward", 0.0)),
                    ),
                    0.0,
                )

            metadata = {
                "reasoning_present": reasoning_present,
                "fold": split,
                "step_idx": idx,
                "total_steps": len(steps),
                "max_context_steps": MAX_CONTEXT_STEPS,
                "max_guidance_steps": MAX_GUIDANCE_STEPS,
                "embedding_model": self.embedding_model_name,
                "task_id": task_id,
                "floor_plan": floor_plan,
                "game_file": game_file,
                "variation_id": variation_idx,
                "model_name": source_model_label,
            }

            entry = {
                "key_raw_goal": goal,
                "key_raw_state": state_repr,
                "key_raw_context": context_repr,
                "key_raw_progress": progress_repr,
                "key_raw_constraints": None,
                "key_raw_model_name": self.embedding_model_name,
                "key_raw_run_number": self.run_number,
                "goal_only": goal_embed,
                "state_only": state_embed,
                "context_only": context_embed,
                "key_embed": key_embed,
                "guidance": json.dumps(self._build_guidance(steps, idx)),
                "thought_id": trajectory_id,
                "task_name": task_name,
                "variation_idx": variation_idx,
                "success": is_successful,
                "next_reward": next_reward,
                "task_type": task_name,
                "environment": ENVIRONMENT,
                "storage_strategy": STORAGE_STRATEGY,
                "agent_type": source_model_label,
                "agent_framework": AGENT_FRAMEWORK,
                "version": VERSION,
                "created_at": timestamp,
                "updated_at": timestamp,
                "metadata": json.dumps(metadata),
            }
            entries.append(entry)

        return entries

    def commit_runtime_trajectory(
        self,
        *,
        trajectory: Dict[str, Any],
        episode_metadata: Dict[str, Any],
        source_model_label: str,
    ) -> Dict[str, Any]:
        entries = self._build_entries(
            trajectory=trajectory,
            episode_metadata=episode_metadata,
            source_model_label=source_model_label,
        )
        if not entries:
            return {
                "total_entries": 0,
                "new_entries": 0,
                "skipped_duplicates": 0,
            }

        with self.lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                _, table = self._open_table()
                new_entries: List[Dict[str, Any]] = []
                duplicate_count = 0
                for entry in entries:
                    if self._entry_exists(table, entry, source_model_label):
                        duplicate_count += 1
                    else:
                        new_entries.append(entry)
                if new_entries:
                    table.add(new_entries)
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

        return {
            "total_entries": len(entries),
            "new_entries": len(new_entries),
            "skipped_duplicates": duplicate_count,
        }

    def commit_runtime_trajectories_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Commit multiple trajectories as one serialized batch.

        Each item must contain:
          - trajectory
          - episode_metadata
          - source_model_label
        """
        all_entries: List[Dict[str, Any]] = []
        for item in items:
            entries = self._build_entries(
                trajectory=item["trajectory"],
                episode_metadata=item["episode_metadata"],
                source_model_label=item["source_model_label"],
            )
            all_entries.extend(entries)

        if not all_entries:
            return {
                "total_entries": 0,
                "new_entries": 0,
                "skipped_duplicates": 0,
                "batch_items": len(items),
            }

        def signature(entry: Dict[str, Any]) -> tuple:
            return (
                entry["key_raw_goal"],
                entry["key_raw_state"],
                entry["key_raw_context"],
                entry["key_raw_progress"],
                entry["key_raw_constraints"],
                entry["agent_type"] if self.dedup_within_model else None,
            )

        with self.lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                _, table = self._open_table()
                new_entries: List[Dict[str, Any]] = []
                duplicate_count = 0
                pending_signatures = set()

                for entry in all_entries:
                    sig = signature(entry)
                    if sig in pending_signatures:
                        duplicate_count += 1
                        continue
                    if self._entry_exists(table, entry, entry["agent_type"]):
                        duplicate_count += 1
                        continue
                    pending_signatures.add(sig)
                    new_entries.append(entry)

                if new_entries:
                    table.add(new_entries)
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

        return {
            "total_entries": len(all_entries),
            "new_entries": len(new_entries),
            "skipped_duplicates": duplicate_count,
            "batch_items": len(items),
        }


def online_memory_enabled_from_env() -> bool:
    return _parse_bool(os.environ.get("ONLINE_MEMORY_ENABLED"), False)
