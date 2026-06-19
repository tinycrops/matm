#!/usr/bin/env python3
"""
Online memory writer for WebArena runtime trajectories.

This appends completed WebArena episode trajectories into an existing LanceDB
table using the same step-level entry template as the offline index while
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

from traj_retrieval.preprocess.webarena.trajectory_entry import TrajectoryEntry


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat()


def _parse_bool(raw: str | None, default: bool = False) -> bool:
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _sql_escape(value: str) -> str:
    return value.replace("'", "''")


class WebArenaOnlineMemoryWriter:
    """Append runtime WebArena trajectories to a LanceDB table."""

    def __init__(
        self,
        indices_dir: str,
        table_name: str = "webarena",
        embedding_model_name: str = "intfloat/e5-base",
        lock_path: Optional[str] = None,
        dedup_within_model: bool = True,
    ):
        self.indices_dir = indices_dir
        self.table_name = table_name
        self.embedding_model_name = embedding_model_name
        self.dedup_within_model = dedup_within_model
        self.lock_path = Path(
            lock_path or (Path(indices_dir) / ".webarena_online_memory.lock")
        )
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.touch(exist_ok=True)

        self._model: Optional[SentenceTransformer] = None
        self._entry_builder: Optional[TrajectoryEntry] = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    @property
    def entry_builder(self) -> TrajectoryEntry:
        if self._entry_builder is None:
            self._entry_builder = TrajectoryEntry(self.model, self.embedding_model_name)
        return self._entry_builder

    def _open_table(self):
        db = lancedb.connect(self.indices_dir)
        table = db.open_table(self.table_name)
        return db, table

    def count_rows(self) -> int:
        _, table = self._open_table()
        return table.count_rows()

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
            episode_metadata.get("task_name")
            or trajectory.get("task")
            or "unknown_task"
        )
        variation_id = (
            episode_metadata.get("variation_id")
            or trajectory.get("variation")
            or "task_id_0"
        )
        config_task_id = episode_metadata.get("task_id")
        objective = trajectory.get("goal_text") or episode_metadata.get("intent") or ""
        sites = episode_metadata.get("sites") or []
        intent_template_id = episode_metadata.get("intent_template_id")
        trajectory_id = f"{source_model_label}-{uuid.uuid4()}"

        normalized_steps: List[Dict[str, Any]] = []
        final_score = trajectory.get("final_score", 0)
        episode_done = bool(trajectory.get("done", False))
        for step in steps:
            normalized_steps.append(
                {
                    "observation": step.get("observation", ""),
                    "action": step.get("action", ""),
                    "url": step.get("url", ""),
                    "success": final_score if episode_done else 0.0,
                    "done": episode_done,
                }
            )

        entries: List[Dict[str, Any]] = []
        for idx in range(len(normalized_steps)):
            entry = self.entry_builder.create_entry(
                trajectory_id=trajectory_id,
                objective=objective,
                trajectory=normalized_steps,
                current_step_idx=idx,
                agent_type=source_model_label,
                task_name=task_name,
                variation_idx=variation_id,
                split="train",
                config_task_id=config_task_id,
                sites=sites,
                intent_template=episode_metadata.get("intent"),
                intent_template_id=intent_template_id,
            )

            metadata = json.loads(entry["metadata"])
            metadata["model_name"] = source_model_label
            metadata["consumer_model"] = episode_metadata.get("consumer_model")
            metadata["consumer_split_seed"] = episode_metadata.get(
                "consumer_split_seed"
            )
            metadata["config_file"] = episode_metadata.get("config_file")
            metadata["source_split"] = episode_metadata.get("source_split")
            metadata["task_id"] = config_task_id
            entry["metadata"] = json.dumps(metadata)
            entry["created_at"] = _utc_now_iso()
            entry["updated_at"] = entry["created_at"]
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
            return {"total_entries": 0, "new_entries": 0, "skipped_duplicates": 0}

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
        self, items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        all_entries: List[Dict[str, Any]] = []
        for item in items:
            all_entries.extend(
                self._build_entries(
                    trajectory=item["trajectory"],
                    episode_metadata=item["episode_metadata"],
                    source_model_label=item["source_model_label"],
                )
            )

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
