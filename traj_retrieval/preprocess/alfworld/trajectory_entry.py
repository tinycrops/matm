#!/usr/bin/env python3
"""
TrajectoryEntry class for creating LanceDB entries from AlfWorld trajectories.
"""

import json
import uuid
from typing import Dict, List, Optional
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print("ERROR: sentence-transformers import failed")
    print(f"Error details: {e}")
    print("\nPlease install/update dependencies with:")
    print("  pip install --upgrade transformers")
    print("  pip install sentence-transformers")
    import sys

    sys.exit(1)


# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

# Maximum context steps to include
MAX_CONTEXT_STEPS = 5

# Maximum guidance steps (next k steps)
MAX_GUIDANCE_STEPS = 5

# Agent configuration
AGENT_TYPE = "seq_2_seq"
ENVIRONMENT = "alfworld"
STORAGE_STRATEGY = "raw_trajectories"
AGENT_FRAMEWORK = "react"
VERSION = 1

# AlfWorld specific metadata
REASONING_PRESENT = False

# ============================================================================


class TrajectoryEntry:
    """Class for creating a single trajectory entry for LanceDB."""

    def __init__(self, embedding_model: SentenceTransformer, model_name: str):
        """
        Initialize the trajectory entry creator.

        Args:
            embedding_model: Sentence transformer model for creating embeddings
            model_name: Name of the embedding model used
        """
        self.model = embedding_model
        self.model_name = model_name

    @staticmethod
    def _create_variation_id(game_file: str, split: str) -> str:
        """
        Create variation_id from game_file path.

        Format: variation_id_{hash}

        Game file path format:
        .../json_2.1.1/{split}/{task_type}-{obj}-{subobj}-{recep}-{fp}/trial_{...}/game.tw-pddl

        The variation_id is created by:
        1. Extracting the path starting from json_2.1.1 onwards
        2. Hashing the entire extracted path
        3. Formatting as variation_id_{hash}

        Args:
            game_file: Path to game.tw-pddl file
            split: Data split (train/valid_seen/valid_unseen/etc) - not used in variation_id

        Returns:
            variation_id string in format: variation_id_{hash}
        """
        from pathlib import Path
        import hashlib

        path_parts = Path(game_file).parts

        # Find the index where json_2.1.1 appears
        json_idx = None
        for i, part in enumerate(path_parts):
            if part == "json_2.1.1":
                json_idx = i
                break

        if json_idx is None:
            # Fallback if json_2.1.1 not found in path
            # Use hash of full game_file path
            path_hash = hashlib.md5(game_file.encode()).hexdigest()
            return f"variation_id_{path_hash}"

        # Extract path from json_2.1.1 onwards
        path_from_json = "/".join(path_parts[json_idx:])

        # Create hash of this path
        path_hash = hashlib.md5(path_from_json.encode()).hexdigest()

        # Return formatted variation_id
        return f"variation_id_{path_hash}"

    def create_entry(
        self,
        trajectory_id: str,
        task_id: str,
        task_type: str,
        goal: str,
        floor_plan: str,
        game_file: str,
        split: str,
        trajectory: List[Dict],
        current_step_idx: int,
    ) -> Dict:
        """
        Create a single entry for the database.

        Args:
            trajectory_id: Unique identifier for this trajectory
            task_id: Task ID from the JSON file
            task_type: Type of task (e.g., "look_at_obj_in_light")
            goal: Goal description
            floor_plan: Floor plan identifier
            game_file: Path to game file
            split: Data split (train/test/dev)
            trajectory: Complete trajectory (excluding last step)
            current_step_idx: Current step index in the trajectory

        Returns:
            Dictionary entry ready for LanceDB insertion
        """
        current_step = trajectory[current_step_idx]

        # Extract current state information
        observation = current_step["observation"]
        action = current_step["action"]

        # Calculate current score (0 until the final step, then 1)
        # Remember: trajectory is already trimmed (last step removed)
        is_final_step = current_step_idx == len(trajectory) - 1
        current_score = 1.0 if is_final_step else 0.0

        # All gold trajectories are successful
        is_successful = True

        # Build state representation (no inventory in AlfWorld)
        state_repr = f"observation: {observation}"

        # Build context (last MAX_CONTEXT_STEPS state+action pairs)
        context_repr = self._build_context(trajectory, current_step_idx)

        # Build progress representation
        progress_repr = f"step_till_now: {current_step_idx}"

        # Build key_raw
        key_raw = {
            "goal": goal,
            "state": state_repr,
            "context": context_repr,
            "progress": progress_repr,
            "constraints": None,
        }

        # Create embeddings
        goal_embed = self.model.encode(goal).tolist()
        state_embed = self.model.encode(state_repr).tolist()
        context_embed = (
            self.model.encode(context_repr).tolist()
            if context_repr
            else [0.0] * len(goal_embed)
        )

        # Create full key representation for embedding
        full_key = f"goal: {goal} | state: {state_repr} | context: {context_repr} | progress: {progress_repr}"
        key_embed = self.model.encode(full_key).tolist()

        # Build guidance (next k steps, maximum MAX_GUIDANCE_STEPS)
        guidance_steps = self._build_guidance(trajectory, current_step_idx)

        # Next reward is always 1.0 for AlfWorld gold trajectories
        next_reward = 1.0

        # Get current timestamp
        timestamp = datetime.utcnow().isoformat()

        # Create variation_id from game_file path
        variation_id = self._create_variation_id(game_file, split)

        # Build metadata
        metadata = {
            "reasoning_present": REASONING_PRESENT,
            "fold": split,
            "step_idx": current_step_idx,
            "total_steps": len(trajectory),
            "max_context_steps": MAX_CONTEXT_STEPS,
            "max_guidance_steps": MAX_GUIDANCE_STEPS,
            "embedding_model": self.model_name,
            "task_id": task_id,
            "floor_plan": floor_plan,
            "game_file": game_file,
            "variation_id": variation_id,
        }

        # Construct the entry
        entry = {
            # Key components (raw)
            "key_raw_goal": key_raw["goal"],
            "key_raw_state": key_raw["state"],
            "key_raw_context": key_raw["context"],
            "key_raw_progress": key_raw["progress"],
            "key_raw_constraints": key_raw["constraints"],
            # Embedded representations (vectors)
            "goal_only": goal_embed,
            "state_only": state_embed,
            "context_only": context_embed,
            "key_embed": key_embed,
            # Value
            "guidance": json.dumps(guidance_steps),
            "thought_id": trajectory_id,
            # Task information
            "task_name": task_type,
            "variation_idx": variation_id,
            "success": is_successful,
            "next_reward": next_reward,
            "task_type": task_type,
            # System information
            "environment": ENVIRONMENT,
            "storage_strategy": STORAGE_STRATEGY,
            "agent_type": AGENT_TYPE,
            "agent_framework": AGENT_FRAMEWORK,
            "version": VERSION,
            # Timestamps
            "created_at": timestamp,
            "updated_at": timestamp,
            # Metadata (as JSON string)
            "metadata": json.dumps(metadata),
        }

        return entry

    def _build_context(self, trajectory: List[Dict], current_step_idx: int) -> str:
        """Build context from previous steps."""
        context_parts = []
        start_idx = max(0, current_step_idx - MAX_CONTEXT_STEPS)

        for j in range(start_idx, current_step_idx):
            prev_step = trajectory[j]
            prev_obs = prev_step["observation"]
            prev_action = prev_step["action"]
            # No reasoning for gold trajectories
            context_parts.append(f"observation: {prev_obs} | action: {prev_action}")

        return " ; ".join(context_parts) if context_parts else ""

    def _build_guidance(
        self, trajectory: List[Dict], current_step_idx: int
    ) -> List[Dict]:
        """Build guidance steps (next k steps)."""
        guidance_steps = []
        end_idx = min(current_step_idx + MAX_GUIDANCE_STEPS, len(trajectory))

        for j in range(current_step_idx, end_idx):
            guidance_step = trajectory[j]
            # Score is 1.0 only for the final step
            is_final = j == len(trajectory) - 1
            guidance_steps.append(
                {
                    "action": guidance_step["action"],
                    "observation": guidance_step["observation"],
                    "score": 1.0 if is_final else 0.0,
                }
            )

        return guidance_steps
