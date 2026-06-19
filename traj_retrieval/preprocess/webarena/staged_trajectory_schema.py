#!/usr/bin/env python3
"""
Schema for staged trajectories in WebArena.

This schema ensures all trajectory staging scripts produce compatible output
that can be consumed by create_lancedb_indices.py.
"""
from typing import Dict, List, Optional

# Environment metadata
ENVIRONMENT = "webarena"
STORAGE_STRATEGY = "raw_trajectories"
AGENT_FRAMEWORK = "react"
VERSION = 1


class StagedTrajectorySchema:
    """
    Schema for staged trajectories that matches the format expected by create_lancedb_indices.py.

    This ensures compatibility with the existing indexing pipeline.
    """

    @staticmethod
    def create(
        task: str,
        model: str,
        trajectory: List[Dict],
        split: str,
        trajectory_id: str,
        source_file: str,
        agent_type: str,
        # Optional fields from original data
        id: Optional[str] = None,
        type: Optional[str] = None,
    ) -> Dict:
        """
        Create a trajectory in the standard staged format.

        Required fields:
        - task: Complete path to config file
        - model: Agent/model type
        - trajectory: List of trajectory steps
        - split: train/test/etc
        - trajectory_id: UUID for this trajectory
        - source_file: Original source file path
        - metadata_info: Agent and system configuration

        Args:
            task: Path to config file (e.g., /path/to/config_files/42.json)
            model: Model name (e.g., openai/gpt-oss-20b)
            trajectory: List of trajectory step dictionaries
            split: Split name (train/test/etc)
            trajectory_id: UUID for this trajectory
            source_file: Original source file path
            agent_type: Agent type for metadata
            id: Optional original ID field
            type: Optional type field

        Returns:
            Dictionary in the staged trajectory format
        """
        staged_data = {
            "task": task,
            "model": model,
            "trajectory": trajectory,
            "split": split,
            "trajectory_id": trajectory_id,
            "source_file": source_file,
            "metadata_info": {
                "agent_type": agent_type,
                "environment": ENVIRONMENT,
                "storage_strategy": STORAGE_STRATEGY,
                "agent_framework": AGENT_FRAMEWORK,
                "version": VERSION,
            },
        }

        # Add optional fields if provided
        if id is not None:
            staged_data["id"] = id
        if type is not None:
            staged_data["type"] = type

        return staged_data

    @staticmethod
    def create_from_existing(
        existing_data: Dict,
        trajectory_id: str,
        source_file: str,
        normalized_task_path: str,
        agent_type: str,
    ) -> Dict:
        """
        Create a staged trajectory from existing trajectory data.

        This preserves all original fields and adds/overrides staging metadata.
        Used when staging trajectories that already have most required fields.

        Args:
            existing_data: Original trajectory data dictionary
            trajectory_id: UUID for this trajectory
            source_file: Original source file path
            normalized_task_path: Complete path to config file
            agent_type: Agent type for metadata

        Returns:
            Dictionary in the staged trajectory format with original data preserved
        """
        # Start with a copy of existing data
        staged_data = existing_data.copy()

        # Override/add required fields
        staged_data["task"] = normalized_task_path
        staged_data["trajectory_id"] = trajectory_id
        staged_data["source_file"] = source_file
        staged_data["metadata_info"] = {
            "agent_type": agent_type,
            "environment": ENVIRONMENT,
            "storage_strategy": STORAGE_STRATEGY,
            "agent_framework": AGENT_FRAMEWORK,
            "version": VERSION,
        }

        return staged_data
