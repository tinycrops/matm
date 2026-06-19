"""
Simulation Actions Loader for simulate_till_tk strategy.

This module provides functionality to load pre-recorded trajectories from no-retrieval runs
and use them for simulation in the simulate_till_tk frequency strategy.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any


class SimulationActionsLoader:
    """
    Loads and manages pre-recorded action sequences from no-retrieval runs.

    Used by the simulate_till_tk strategy to simulate actions for steps 0 to k-1,
    then perform retrieval at step k and continue with LLM from step k+1 onwards.
    """

    def __init__(self, json_file_path: str, environment_name: str):
        """
        Initialize the simulation loader.

        Args:
            json_file_path: Path to the JSON file containing no-retrieval run results
            environment_name: Name of the environment (for error messages)

        Raises:
            FileNotFoundError: If the JSON file doesn't exist
            ValueError: If the JSON file has invalid format

        Note:
            simulate_step_k is now read from each episode descriptor in the evaluation set,
            not from the config file. This allows different episodes to have different
            simulation steps.
        """
        self.json_file_path = Path(json_file_path)
        self.environment_name = environment_name
        self.data_cache: Dict[
            tuple, List[Dict]
        ] = {}  # (task_name, variation_id) -> list of {action, reasoning}

        # Validate file exists
        if not self.json_file_path.exists():
            raise FileNotFoundError(
                f"\n{'='*80}\n"
                f"❌ SIMULATION DATA NOT FOUND\n"
                f"{'='*80}\n"
                f"Strategy: simulate_till_tk\n"
                f"Environment: {environment_name}\n"
                f"Expected file: {json_file_path}\n"
                f"\n"
                f"The simulate_till_tk strategy requires pre-recorded no-retrieval runs.\n"
                f"Please ensure the simulation data file exists at the specified path.\n"
                f"{'='*80}\n"
            )

        # Load and validate data
        self._load_data()

    def _load_data(self):
        """
        Load simulation data from JSON file and build lookup cache.

        Raises:
            ValueError: If JSON format is invalid
        """
        try:
            with self.json_file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"\n{'='*80}\n"
                f"❌ INVALID SIMULATION DATA FORMAT\n"
                f"{'='*80}\n"
                f"Strategy: simulate_till_tk\n"
                f"Environment: {self.environment_name}\n"
                f"File: {self.json_file_path}\n"
                f"Error: {e}\n"
                f"\n"
                f"The JSON file could not be parsed. Please check the file format.\n"
                f"{'='*80}\n"
            )

        # Validate data is a list
        if not isinstance(data, list):
            raise ValueError(
                f"\n{'='*80}\n"
                f"❌ INVALID SIMULATION DATA STRUCTURE\n"
                f"{'='*80}\n"
                f"Strategy: simulate_till_tk\n"
                f"Environment: {self.environment_name}\n"
                f"File: {self.json_file_path}\n"
                f"\n"
                f"Expected: List of episode dictionaries\n"
                f"Got: {type(data).__name__}\n"
                f"{'='*80}\n"
            )

        # Build lookup cache: (task_name, variation_id) -> list of {action, reasoning, observation, url}
        for episode in data:
            if not isinstance(episode, dict):
                continue

            task_name = episode.get("task_name")
            # Handle both variation_id (WebArena) and variation_idx (others)
            variation_id = episode.get("variation_id") or episode.get("variation_idx")
            action_sequences = episode.get("actionSequences", [])

            if task_name is None:
                continue

            # Extract actions, reasoning, observations, and URLs from actionSequences
            steps = []
            for step in action_sequences:
                if isinstance(step, dict) and "action" in step:
                    steps.append(
                        {
                            "action": step["action"],
                            "reasoning": step.get(
                                "reasoning", ""
                            ),  # Use actual reasoning, fallback to empty string
                            "observation": step.get(
                                "observation", ""
                            ),  # Store observation for pseudo_simulate_till_closest_tk
                            "url": step.get(
                                "url", ""
                            ),  # Store URL for WebArena (empty string for other environments)
                        }
                    )

            # Store in cache
            # Use string representation for variation_id to handle both int and string
            key = (task_name, str(variation_id) if variation_id is not None else None)
            self.data_cache[key] = steps

        print(f"✅ Loaded simulation data: {self.json_file_path}")
        print(f"   - Episodes with simulation data: {len(self.data_cache)}")

    def get_steps(self, task_name: str, variation_id: Any) -> List[Dict]:
        """
        Get step sequence (action + reasoning) for a specific episode.

        Args:
            task_name: Task name (e.g., "boil", "intent_template_id_280")
            variation_id: Variation ID (can be int, str, or None)

        Returns:
            List of step dictionaries, each containing 'action' and 'reasoning' keys

        Raises:
            RuntimeError: If no simulation data exists for this episode (fail-fast)
        """
        # Normalize variation_id to string for lookup
        variation_key = str(variation_id) if variation_id is not None else None
        key = (task_name, variation_key)

        if key not in self.data_cache:
            raise RuntimeError(
                f"\n{'='*80}\n"
                f"❌ NO SIMULATION DATA FOR EPISODE\n"
                f"{'='*80}\n"
                f"Strategy: simulate_till_tk\n"
                f"Environment: {self.environment_name}\n"
                f"Task: {task_name}\n"
                f"Variation ID: {variation_id}\n"
                f"\n"
                f"The simulate_till_tk strategy requires pre-recorded actions for ALL episodes.\n"
                f"This episode is missing from the simulation data file:\n"
                f"  {self.json_file_path}\n"
                f"\n"
                f"Available episodes in simulation data: {len(self.data_cache)}\n"
                f"{'='*80}\n"
            )

        return self.data_cache[key]

    def has_data(self, task_name: str, variation_id: Any) -> bool:
        """
        Check if simulation data exists for an episode.

        Args:
            task_name: Task name
            variation_id: Variation ID (can be int, str, or None)

        Returns:
            True if simulation data exists, False otherwise
        """
        variation_key = str(variation_id) if variation_id is not None else None
        key = (task_name, variation_key)
        return key in self.data_cache

    def get_episode_count(self) -> int:
        """Get total number of episodes with simulation data."""
        return len(self.data_cache)

    def get_target_state_info(
        self, task_name: str, variation_id: Any, target_step_k: int
    ) -> dict:
        """
        Get the state info (observation + URL) at step k from simulation data.

        Used by pseudo_simulate_till_closest_tk strategy for WebArena to find the target state
        that will trigger retrieval when matched during execution.

        Args:
            task_name: Task name (e.g., "intent_template_id_280")
            variation_id: Variation ID (can be int, str, or None)
            target_step_k: The step number to retrieve state info from

        Returns:
            Dict with keys: {"observation": str, "url": str}

        Raises:
            RuntimeError: If no simulation data exists for this episode or step is invalid
        """
        # Normalize variation_id to string for lookup
        variation_key = str(variation_id) if variation_id is not None else None
        key = (task_name, variation_key)

        if key not in self.data_cache:
            raise RuntimeError(
                f"\n{'='*80}\n"
                f"❌ NO SIMULATION DATA FOR EPISODE\n"
                f"{'='*80}\n"
                f"Strategy: pseudo_simulate_till_closest_tk\n"
                f"Environment: {self.environment_name}\n"
                f"Task: {task_name}\n"
                f"Variation ID: {variation_id}\n"
                f"\n"
                f"The pseudo_simulate_till_closest_tk strategy requires pre-recorded data for ALL episodes.\n"
                f"This episode is missing from the simulation data file:\n"
                f"  {self.json_file_path}\n"
                f"\n"
                f"Available episodes in simulation data: {len(self.data_cache)}\n"
                f"{'='*80}\n"
            )

        steps = self.data_cache[key]

        # Check if target_step_k is within bounds
        if target_step_k < 0 or target_step_k >= len(steps):
            raise RuntimeError(
                f"\n{'='*80}\n"
                f"❌ INVALID TARGET STEP\n"
                f"{'='*80}\n"
                f"Strategy: pseudo_simulate_till_closest_tk\n"
                f"Environment: {self.environment_name}\n"
                f"Task: {task_name}\n"
                f"Variation ID: {variation_id}\n"
                f"Target step k: {target_step_k}\n"
                f"Available steps: {len(steps)}\n"
                f"\n"
                f"The target step k is out of bounds. Please ensure the target step\n"
                f"is within the range of available steps in the simulation data.\n"
                f"{'='*80}\n"
            )

        # Get state info at step k
        step_data = steps[target_step_k]
        observation = step_data.get("observation", "")
        url = step_data.get("url", "")  # Extract URL if available

        return {
            "observation": observation,
            "url": url,  # May be empty string for non-web environments
        }
