#!/usr/bin/env python3
"""
TrajectoryEntry class for creating LanceDB entries from WebArena trajectories.
"""
import json
import uuid
from typing import Dict, List, Optional
from datetime import datetime
from urllib.parse import urlparse

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

# Environment and strategy configuration
ENVIRONMENT = "webarena"
STORAGE_STRATEGY = "raw_trajectories"
AGENT_FRAMEWORK = "react"
VERSION = 1

# WebArena specific metadata
REASONING_PRESENT = False

# ============================================================================


def make_url_domain_agnostic(url: str) -> str:
    """
    Convert full URL to domain-agnostic path.

    Example:
        http://<WEBARENA_HOST>:7780/admin/admin/dashboard/
        -> /admin/admin/dashboard/

    Args:
        url: Full URL string

    Returns:
        Domain-agnostic path
    """
    if not url:
        return ""

    parsed = urlparse(url)
    # Return path + query + fragment (if any)
    result = parsed.path
    if parsed.query:
        result += f"?{parsed.query}"
    if parsed.fragment:
        result += f"#{parsed.fragment}"

    return result if result else "/"


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

    def create_entry(
        self,
        trajectory_id: str,
        objective: str,
        trajectory: List[Dict],
        current_step_idx: int,
        agent_type: str,
        task_name: str,
        variation_idx: int,
        split: str = "train",
        config_task_id: Optional[int] = None,
        sites: Optional[List[str]] = None,
        intent_template: Optional[str] = None,
        intent_template_id: Optional[int] = None,
    ) -> Dict:
        """
        Create a single entry for the database.

        Args:
            trajectory_id: Unique identifier for this trajectory
            objective: Task objective/goal description
            trajectory: Complete trajectory
            current_step_idx: Current step index in the trajectory
            agent_type: Model name (e.g., "gpt-4-turbo-preview")
            task_name: Concatenated sorted sites (e.g., "shopping_admin")
            variation_idx: Intent template ID from config
            split: Data split (train/test/dev)
            config_task_id: Task ID from config file
            sites: List of sites from config
            intent_template: Intent template from config
            intent_template_id: Intent template ID from config

        Returns:
            Dictionary entry ready for LanceDB insertion
        """
        current_step = trajectory[current_step_idx]

        # Extract current state information
        observation = current_step.get("observation", "")
        action = current_step.get("action", "")
        url = current_step.get("url", "")

        # Make URL domain-agnostic
        url_path = make_url_domain_agnostic(url)

        # Extract success information from current step
        current_success = current_step.get("success", 0.0)
        is_done = current_step.get("done", False)

        # Determine if trajectory is successful (check last step)
        last_step = trajectory[-1]
        is_successful = (
            last_step.get("done", False) and last_step.get("success", 0.0) == 1.0
        )

        # Build state representation
        # Truncate observation if too long (WebArena observations can be very long HTML)
        obs_truncated = (
            observation[:500] + "..." if len(observation) > 500 else observation
        )
        state_repr = f"url: {url_path} | observation: {obs_truncated}"

        # Build context (last MAX_CONTEXT_STEPS state+action pairs)
        context_repr = self._build_context(trajectory, current_step_idx)

        # Build progress representation
        progress_repr = f"step_till_now: {current_step_idx}/{len(trajectory)}"

        # Build key_raw
        key_raw = {
            "goal": objective,
            "state": state_repr,
            "context": context_repr,
            "progress": progress_repr,
            "constraints": None,
        }

        # Create embeddings
        goal_embed = self.model.encode(objective).tolist()
        state_embed = self.model.encode(state_repr).tolist()
        context_embed = (
            self.model.encode(context_repr).tolist()
            if context_repr
            else [0.0] * len(goal_embed)
        )

        # Create full key representation for embedding
        full_key = f"goal: {objective} | state: {state_repr} | context: {context_repr} | progress: {progress_repr}"
        key_embed = self.model.encode(full_key).tolist()

        # Build guidance (next k steps, maximum MAX_GUIDANCE_STEPS)
        guidance_steps = self._build_guidance(trajectory, current_step_idx)

        # Next reward calculation
        # For WebArena, reward is typically the success value
        if current_step_idx + 1 < len(trajectory):
            next_step = trajectory[current_step_idx + 1]
            next_reward = next_step.get("success", 0.0)
        else:
            next_reward = current_success

        # Get current timestamp
        timestamp = datetime.utcnow().isoformat()

        # Build metadata
        metadata = {
            "reasoning_present": REASONING_PRESENT,
            "fold": split,
            "step_idx": current_step_idx,
            "total_steps": len(trajectory),
            "max_context_steps": MAX_CONTEXT_STEPS,
            "max_guidance_steps": MAX_GUIDANCE_STEPS,
            "embedding_model": self.model_name,
            "config_task_id": config_task_id,
            "url_path": url_path,
            "sites": sites,
            "intent_template": intent_template,
            "intent_template_id": intent_template_id,
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
            "task_name": task_name,  # Concatenated sorted sites
            "variation_idx": variation_idx,  # Intent template ID
            "success": is_successful,
            "next_reward": next_reward,
            "task_type": task_name,  # Use task_name as task_type
            # System information
            "environment": ENVIRONMENT,
            "storage_strategy": STORAGE_STRATEGY,
            "agent_type": agent_type,  # Model name from trajectory JSON
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
            prev_obs = prev_step.get("observation", "")
            prev_action = prev_step.get("action", "")
            prev_url = prev_step.get("url", "")

            # Make URL domain-agnostic
            prev_url_path = make_url_domain_agnostic(prev_url)

            # Truncate observation for context
            obs_truncated = prev_obs[:200] + "..." if len(prev_obs) > 200 else prev_obs

            context_parts.append(
                f"url: {prev_url_path} | observation: {obs_truncated} | action: {prev_action}"
            )

        return " ; ".join(context_parts) if context_parts else ""

    def _build_guidance(
        self, trajectory: List[Dict], current_step_idx: int
    ) -> List[Dict]:
        """Build guidance steps (next k steps)."""
        guidance_steps = []
        end_idx = min(current_step_idx + MAX_GUIDANCE_STEPS, len(trajectory))

        for j in range(current_step_idx, end_idx):
            guidance_step = trajectory[j]
            url = guidance_step.get("url", "")
            url_path = make_url_domain_agnostic(url)

            guidance_steps.append(
                {
                    "action": guidance_step.get("action", ""),
                    "observation": guidance_step.get("observation", "")[
                        :500
                    ],  # Truncate
                    "url_path": url_path,
                    "score": guidance_step.get("success", 0.0),
                    "done": guidance_step.get("done", False),
                }
            )

        return guidance_steps
