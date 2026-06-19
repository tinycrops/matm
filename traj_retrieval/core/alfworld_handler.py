# traj_retrieval/core/alfworld_handler.py
# ALFWorld-specific implementation of EnvironmentHandler

import os
import random
import json
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Any
from .base_handler import EnvironmentHandler, TrajectoryNode, StepResult

# Try to import alfworld (with AlfredTWEnv wrapper)
try:
    import alfworld
    import alfworld.agents.environment.alfred_tw_env as alfred_env

    ALFWORLD_AVAILABLE = True
except ImportError:
    ALFWORLD_AVAILABLE = False
    print("[WARNING] ALFWorld not available. Install with: pip install alfworld")

# Task name to task type ID mapping (used by AlfredTWEnv)
FOLDER_SLUG_TO_ID = {
    "pick_and_place_simple": 1,
    "look_at_obj_in_light": 2,
    "pick_clean_then_place_in_recep": 3,
    "pick_heat_then_place_in_recep": 4,
    "pick_cool_then_place_in_recep": 5,
    "pick_two_obj_and_place": 6,
}


def process_ob(ob: Any) -> str:
    """Clean up observation text."""
    if not isinstance(ob, str):
        ob = str(ob)
    return ob


def first_scalar(x: Any) -> Any:
    """Extract the first scalar-like value from nested lists/tuples/arrays."""
    y = x
    while isinstance(y, (list, tuple)) and len(y) > 0:
        y = y[0]
    return y


def reward_to_float(reward: Any) -> float:
    """Convert env reward (possibly nested) to a float."""
    y = first_scalar(reward)
    try:
        return float(y)
    except Exception:
        if isinstance(y, (list, tuple)):
            try:
                return float(sum(map(float, y)))
            except Exception:
                pass
        return 0.0


def done_to_bool(done_flag: Any) -> bool:
    """Convert env done (possibly nested) to bool."""
    y = first_scalar(done_flag)
    try:
        return bool(y)
    except Exception:
        return False


class AlfWorldHandler(EnvironmentHandler):
    """
    ALFWorld-specific implementation of EnvironmentHandler.
    Uses the AlfredTWEnv wrapper for cleaner commands and observations.

    Each instance creates its own independent environment, ensuring no interference
    between concurrent episodes running in different processes/threads.

    IMPORTANT: Always requires a specific game_file to be set via initialize_from_episode()
    or initialize(). Will raise clear errors if game_file is missing or invalid.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ALFWorld handler.

        Args:
            config_path: Path to base_config.yaml. Auto-detected if None.

        Raises:
            ImportError: If ALFWorld is not installed
        """
        if not ALFWORLD_AVAILABLE:
            raise ImportError(
                "ALFWorld is not installed. " "Install with: pip install alfworld"
            )

        # Auto-detect config path if not provided
        if config_path is None:
            # Try relative path first (from project root)
            default_path = Path("environments/alfworld/base_config.yaml")
            if not default_path.exists():
                # Try from this file's location
                default_path = (
                    Path(__file__).parent.parent.parent
                    / "environments"
                    / "alfworld"
                    / "base_config.yaml"
                )
            self.config_path = str(default_path)
        else:
            self.config_path = config_path

        # Instance-specific state (NO shared state across instances)
        self.env = None
        self.task_name: str = ""
        self.task_type_id: int = 0
        self.variation_idx: str = (
            "unknown"  # Format: "{split}-fp{N}-{obj}-{subobj}-{recep}"
        )
        self.game_file: str = ""  # REQUIRED - must be set before initialize()
        self.goal_text: str = ""
        self.max_steps: int = 0
        self.is_unlimited_mode: bool = False
        self._last_observation: str = ""
        self._last_info: Dict = {}
        self._step_count: int = 0
        self._split: str = "train"  # Default split
        self.episode_metadata: Dict = {}  # Store episode metadata for detailed logging

    def parse_evaluation_set(
        self, evaluation_set_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Parse ALFWorld evaluation set into episode descriptors.
        Only supports flat list format: [{task_type, variation_id, num_steps, game_file, ...}]

        IMPORTANT: Each episode MUST have a 'game_file' field that points to an existing file.
        This handler requires specific game files and will fail if they're missing.

        Args:
            evaluation_set_data: List of episode dictionaries

        Returns:
            List of episode descriptors

        Raises:
            ValueError: If evaluation set format is invalid or required fields are missing
            FileNotFoundError: If any game_file doesn't exist
        """
        episodes = []

        # Evaluation set must be a flat list
        if not isinstance(evaluation_set_data, list):
            raise ValueError(
                f"ALFWorld evaluation set must be a flat list. "
                f"Got {type(evaluation_set_data).__name__}."
            )

        for idx, item in enumerate(evaluation_set_data):
            # Required field: task_type
            task_name = item.get("task_type")
            if not task_name:
                raise ValueError(
                    f"Episode {idx}: Missing required field 'task_type' in evaluation set item"
                )

            # Required field: variation_id
            # Note: variation_id format in ALFWorld is: "{split}-fp{N}-{obj}-{subobj}-{recep}"
            # Example: "valid_seen-fp7-Plate-None-Cabinet" or "train-fp12-Potato-None-GarbageCan"
            # This matches the format stored in LanceDB, so we use it directly
            variation_id = item.get("variation_id")
            if not variation_id:
                raise ValueError(
                    f"Episode {idx}: Missing required field 'variation_id' in evaluation set item"
                )

            # Use variation_id as-is for variation_idx (matches LanceDB format)
            variation_idx = variation_id

            # Required field: num_steps
            max_steps = item.get("num_steps")
            if max_steps is None:
                raise ValueError(
                    f"Episode {idx} ({task_name} {variation_id}): Missing required field 'num_steps'"
                )

            # CRITICAL: game_file is REQUIRED
            game_file = item.get("game_file", "")
            if not game_file:
                raise ValueError(
                    f"Episode {idx} ({task_name} {variation_id}): Missing required field 'game_file'. "
                    f"ALFWorld handler requires specific game files for each episode."
                )

            # Validate game_file exists
            if not os.path.exists(game_file):
                raise FileNotFoundError(
                    f"Episode {idx} ({task_name} {variation_id}): Game file not found: '{game_file}'. "
                    f"Ensure all game files exist before running evaluation."
                )

            episode_descriptor = {
                "episode_id": f"{task_name}_v{variation_idx}",
                "task_name": task_name,
                "variation_idx": variation_idx,
                "max_steps": max_steps,
                "game_file": game_file,
                "metadata": item,
            }

            # Extract simulate_step_k if present (for simulate_till_tk strategy)
            if "simulate_step_k" in item:
                episode_descriptor["simulate_step_k"] = item["simulate_step_k"]

            episodes.append(episode_descriptor)

        print(
            f"[{self.environment_name}] ✅ Parsed {len(episodes)} episodes from evaluation set",
            flush=True,
        )
        print(
            f"[{self.environment_name}] All episodes have valid game files", flush=True
        )

        return episodes

    def initialize_from_episode(self, episode_descriptor: Dict[str, Any]) -> None:
        """Initialize from episode descriptor."""
        task_name = episode_descriptor["task_name"]
        max_steps = episode_descriptor["max_steps"]
        variation_idx = episode_descriptor["variation_idx"]
        self.game_file = episode_descriptor.get("game_file", "")

        # Store episode metadata for later access (e.g., for detailed logging)
        self.episode_metadata = episode_descriptor.get("metadata", {})

        self.initialize(task_name, max_steps, variation_idx)

    @property
    def environment_name(self) -> str:
        return "ALFWorld"

    def initialize(
        self, task_name: str, max_steps: int, variation_idx: Optional[str] = None
    ) -> None:
        """
        Initialize ALFWorld environment with a specific game file.

        IMPORTANT: self.game_file MUST be set before calling this method
        (typically via initialize_from_episode() which extracts it from episode_descriptor).

        Args:
            task_name: Task type name (e.g., "look_at_obj_in_light")
            max_steps: Maximum steps for the episode
            variation_idx: Variation identifier string (e.g., "train-fp12-Potato-None-GarbageCan")
                         Format: "{split}-fp{N}-{obj}-{subobj}-{recep}"

        Raises:
            ValueError: If task_name is invalid or game_file is not set
            FileNotFoundError: If config file or game_file doesn't exist
            RuntimeError: If environment initialization fails
        """
        self.task_name = task_name
        self.max_steps = max_steps
        self.is_unlimited_mode = max_steps >= 10000
        self.variation_idx = variation_idx if variation_idx is not None else "unknown"

        # VALIDATE: game_file must be set and exist
        if not self.game_file:
            raise ValueError(
                f"game_file must be set before calling initialize(). "
                f"Use initialize_from_episode() or set handler.game_file directly."
            )

        if not os.path.exists(self.game_file):
            raise FileNotFoundError(
                f"Game file not found: {self.game_file}. "
                f"Ensure the game file exists before initializing."
            )

        print(
            f"[{self.environment_name}] Initializing task: {task_name}, variation: {self.variation_idx}",
            flush=True,
        )
        print(f"[{self.environment_name}] Game file: {self.game_file}", flush=True)

        # Map task name to task type ID
        task_type_id = FOLDER_SLUG_TO_ID.get(task_name)
        if task_type_id is None:
            raise ValueError(
                f"Unknown task name: '{task_name}'. "
                f"Valid task names: {list(FOLDER_SLUG_TO_ID.keys())}"
            )

        self.task_type_id = task_type_id

        # Set ALFWORLD_DATA environment variable (use absolute path)
        alfworld_data_dir = os.environ.get("ALFWORLD_DATA")
        if not alfworld_data_dir:
            # Default to absolute path from project root
            alfworld_data_dir = str(Path("environments/alfworld/data").resolve())
            os.environ["ALFWORLD_DATA"] = alfworld_data_dir

        print(
            f"[{self.environment_name}] ALFWORLD_DATA={os.environ['ALFWORLD_DATA']}",
            flush=True,
        )

        # Load base config
        config_path = Path(self.config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(config_path, "r") as f:
            config_text = f.read()

        # Expand environment variables in config
        config_text = config_text.replace("$ALFWORLD_DATA", os.environ["ALFWORLD_DATA"])
        config = yaml.safe_load(config_text)

        # Set task type in config
        config["env"]["task_types"] = [task_type_id]

        print(
            f"[{self.environment_name}] Config data_path: {config['dataset']['data_path']}",
            flush=True,
        )

        # Create independent environment instance for this specific game file
        # NO POOLING - each handler instance gets its own env
        print(
            f"[{self.environment_name}] Creating independent AlfredTWEnv for specific game",
            flush=True,
        )

        try:
            # Step 1: Create AlfredTWEnv (this scans and collects all game files for the task type)
            # This is the slow part (~5 seconds), but necessary
            self.env = alfred_env.AlfredTWEnv(config, train_eval=self._split)
            original_game_count = len(self.env.game_files)

            # Step 2: Filter to ONLY the specific game file BEFORE init_env()
            # This is CRITICAL - must happen before init_env() to work correctly
            print(
                f"[{self.environment_name}] 🎯 Filtering to specific game file (from {original_game_count} available)",
                flush=True,
            )
            self.env.game_files = [self.game_file]
            self.env.num_games = 1
            print(
                f"[{self.environment_name}] Filtered to 1 game file: {Path(self.game_file).name}",
                flush=True,
            )

            # Step 3: Initialize env (registers the filtered game file with TextWorld)
            self.env = self.env.init_env(batch_size=1)

            print(
                f"[{self.environment_name}] ✅ AlfredTWEnv initialized successfully for game: {Path(self.game_file).name}",
                flush=True,
            )

        except Exception as e:
            # Clear error message with context
            raise RuntimeError(
                f"Failed to initialize AlfredTWEnv for game '{self.game_file}': {e}"
            ) from e

    def reset(self) -> TrajectoryNode:
        """Reset environment and return initial state."""
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call initialize() first.")

        # Reset AlfredTWEnv (returns batched results)
        ob, info = self.env.reset()

        # Unwrap from batch (batch_size=1)
        ob_text = ob[0]
        self._last_info = info
        self._step_count = 0

        # Extract goal and observation from initial state
        # Format: "[observation text]\nYour task is to: [goal text]"
        # The last line contains the goal, everything before is the observation
        lines = ob_text.split("\n")

        # Last line is the goal (starts with "Your task is to:")
        if lines and "Your task is to:" in lines[-1]:
            goal_line = lines[-1].strip()
            self.goal_text = goal_line.replace("Your task is to:", "").strip()
            # Everything before last line is observation
            init_obs = "\n".join(lines[:-1])
        else:
            # This shouldn't happen in ALFWorld, but handle gracefully
            raise ValueError(
                f"Could not extract goal from initial observation. Got: {ob_text[:200]}"
            )

        # Clean up observation
        self._last_observation = process_ob(init_obs)

        print(f"[{self.environment_name}] Goal: '{self.goal_text}'", flush=True)
        print(
            f"[{self.environment_name}] Initial observation: '{self._last_observation[:200]}...'",
            flush=True,
        )

        # Get admissible actions from info dict (already clean and indexed)
        admissible = info.get("admissible_commands", [[]])[0]

        print(
            f"[{self.environment_name}] Initial admissible actions: {len(admissible)}",
            flush=True,
        )
        if admissible:
            print(
                f"[{self.environment_name}] Sample actions: {admissible[:3]}",
                flush=True,
            )

        # ALFWorld does not expose a separate inventory observation
        inventory = ""

        return TrajectoryNode(
            observation=self._last_observation,
            goal=self.goal_text,
            admissible_actions=admissible,
            inventory=inventory,
            step_number=0,
            internal_step_count=0,
            done=False,
            extra_info={
                "max_steps": self.max_steps,
                "is_unlimited": self.is_unlimited_mode,
            },
        )

    def step(self, action: str, current_node: TrajectoryNode) -> StepResult:
        """Execute an action in ALFWorld."""
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call initialize() first.")

        internal_step_before = self._step_count

        # Execute action in AlfredTWEnv (expects batch input)
        # Returns: new_ob, reward, done_arr, info
        try:
            new_ob, reward, done_arr, info = self.env.step([action])
        except Exception as e:
            # If action fails, return current state with no reward
            print(
                f"[{self.environment_name}] Action '{action}' failed: {e}", flush=True
            )
            new_ob = [self._last_observation]
            reward = 0.0
            done_arr = False
            info = self._last_info

        # Unwrap from batch
        new_ob_text = new_ob[0] if isinstance(new_ob, list) else new_ob

        # Process observation (no goal line in subsequent observations)
        self._last_observation = process_ob(new_ob_text)
        self._last_info = info

        # Update step count
        self._step_count += 1
        internal_steps_consumed = 1

        # Convert reward and done to proper types
        step_reward = reward_to_float(reward)
        done = done_to_bool(done_arr)

        # Check if task was won
        if info.get("won", [False])[0]:
            done = True
            step_reward = 1.0  # Success reward

        # Score is the reward (ALFWorld uses binary scoring: 0 or 1)
        score = step_reward

        return StepResult(
            observation=self._last_observation,
            reward=step_reward,
            done=done,
            score=score,
            info=info,
            internal_steps_consumed=internal_steps_consumed,
        )

    def get_current_node(self, step_number: int) -> TrajectoryNode:
        """Get current state as a TrajectoryNode."""
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call initialize() first.")

        # Get current admissible actions from last info dict
        admissible = self._last_info.get("admissible_commands", [[]])[0]

        if not admissible:
            admissible = ["look", "inventory", "examine"]

        # ALFWorld doesn't have inventory in the same way
        inventory = ""

        return TrajectoryNode(
            observation=self._last_observation,
            goal=self.goal_text,
            admissible_actions=admissible,
            inventory=inventory,
            step_number=step_number,
            internal_step_count=self._step_count,
            done=False,
            extra_info={
                "max_steps": self.max_steps,
                "is_unlimited": self.is_unlimited_mode,
                "internal_step_count": self._step_count,
            },
        )

    def close(self) -> None:
        """
        Close the environment and clean up resources.
        Each handler instance owns its env, so we always close it.
        """
        if self.env is not None:
            print(
                f"[{self.environment_name}] Closing environment for game: {Path(self.game_file).name if self.game_file else 'unknown'}",
                flush=True,
            )
            try:
                self.env.close()
            except Exception as e:
                print(
                    f"[{self.environment_name}] Warning: Error closing env: {e}",
                    flush=True,
                )
            finally:
                self.env = None

    def get_task_metadata(self) -> Dict[str, Any]:
        """Get task metadata for logging."""
        return {
            "task_name": self.task_name,
            "variation": self.variation_idx,
            "max_variations": 1,  # ALFWorld variations are file-based
            "goal_text": self.goal_text,
            "max_steps": self.max_steps,
            "is_unlimited": self.is_unlimited_mode,
            "environment": self.environment_name,
        }

    def is_successful_episode(self, final_score: float) -> bool:
        """
        Determine if an ALFWorld episode is successful.
        ALFWorld uses binary scoring: 1 for success, 0 for failure.
        """
        return final_score == 1

    def format_retrieval_result(
        self,
        raw_data: Dict[str, Any],
        max_steps: int = 20,
        retrieval_type: str = "trajectory",
    ) -> str:
        """
        Format retrieved trajectory for ALFWorld.
        """
        if retrieval_type != "trajectory":
            print(
                f"[ALFWorld] Unknown retrieval type: {retrieval_type}, skipping formatting"
            )
            return ""

        if not raw_data:
            return ""

        # Extract trajectory information (support both LanceDB and legacy formats)
        # LanceDB uses "trajectory_steps", legacy FAISS uses "remaining_action_observation_pairs"
        remaining_pairs = raw_data.get("trajectory_steps") or raw_data.get(
            "remaining_action_observation_pairs", []
        )
        task_description = raw_data.get("task_description", "")

        if not remaining_pairs:
            return ""

        print(
            f"[ALFWorld] Formatting trajectory: {len(remaining_pairs)} total steps, showing {min(len(remaining_pairs), max_steps)}"
        )

        # Format the trajectory steps
        formatted_steps = []
        for i, pair in enumerate(remaining_pairs[:max_steps]):
            action = pair.get("action", "")
            observation = pair.get("observation", "")

            formatted_steps.append(
                f"Step {i+1}: {action}\n" f"  Observation: {observation}"
            )

        # Create the context string
        context = f"""RETRIEVED TRAJECTORY:
Task: {task_description}

Retrieved successful trajectory sequence:
{chr(10).join(formatted_steps)}

Use this trajectory as a reference for your planning. Consider:
1. The sequence of actions that led to success
2. The observations and their progression
3. How to adapt this strategy to the current situation and your goal
4. What steps might be different or similar in your current context

"""
        return context

    def get_environment_description(self) -> str:
        """Get ALFWorld-specific environment description."""
        return """ENVIRONMENT LAYOUT:
- You are in a household environment (kitchen, bedroom, bathroom, living room, etc.)
- You can navigate between rooms and interact with objects
- Objects can be picked up, opened, closed, heated, cooled, cleaned, etc."""

    def get_action_types_description(self) -> str:
        """Get ALFWorld-specific action types."""
        return """AVAILABLE ACTIONS:
- go to <receptacle>: navigate to a location or receptacle
- take <object> from <receptacle>: pick up an object
- put <object> in/on <receptacle>: place an object
- open <receptacle>: open a container
- close <receptacle>: close a container
- toggle <object>: turn on/off a device
- clean <object> with <receptacle>: clean an object (e.g., with sinkbasin)
- heat <object> with <receptacle>: heat an object (e.g., with microwave)
- cool <object> with <receptacle>: cool an object (e.g., with fridge)
- use <object>: use a device or tool
- look: observe the current room
- inventory: check what you're carrying
- examine <object>: look at an object closely"""

    def get_example_episode(self, task_name: Optional[str] = None) -> str:
        """
        Get ALFWorld-specific one-shot example.

        Args:
            task_name: Optional task name to get specific example (e.g., 'pick_and_place', 'pick_clean_then_place')
                      If None, uses 'pick_clean_then_place' as default

        Returns:
            Formatted example string

        Raises:
            FileNotFoundError: If example JSON file not found
            ValueError: If task_name not found in examples
        """
        import json
        from pathlib import Path

        example_path = Path(__file__).parent / "alfworld_example.json"

        # Load examples - fail if file doesn't exist
        with open(example_path, "r") as f:
            examples = json.load(f)

        # Determine which task to use
        if task_name is None:
            task_name = "pick_clean_then_place"

        # Validate task exists (support fuzzy matching by substring)
        if task_name not in examples:
            # Try to find the best matching key by substring containment
            candidate_keys = [k for k in examples.keys() if k in task_name]
            if candidate_keys:
                # Choose the longest matching key to avoid accidental short matches
                best_key = max(candidate_keys, key=len)
                print(
                    f"[ALFWorld] Mapping task '{task_name}' to example key '{best_key}' via substring match",
                    flush=True,
                )
                task_name = best_key
            else:
                # As a fallback, try reverse containment (task_name inside key)
                reverse_candidates = [k for k in examples.keys() if task_name in k]
                if reverse_candidates:
                    best_key = max(reverse_candidates, key=len)
                    print(
                        f"[ALFWorld] Mapping task '{task_name}' to example key '{best_key}' via reverse substring match",
                        flush=True,
                    )
                    task_name = best_key
                else:
                    raise ValueError(
                        f"Task '{task_name}' not found in alfworld_example.json (no fuzzy match). "
                        f"Available tasks: {list(examples.keys())}"
                    )

        # Use the second trajectory (index 1) which tends to be more complete
        trajectories = examples[task_name]
        if not trajectories or len(trajectories) < 2:
            # Fall back to first trajectory if second doesn't exist
            trajectory = trajectories[0] if trajectories else []
        else:
            trajectory = trajectories[1]

        if not trajectory:
            raise ValueError(f"No trajectories found for task '{task_name}'")

        # Extract task from first user message
        task = "Unknown task"
        if trajectory and trajectory[0].get("role") == "user":
            content = trajectory[0]["content"]
            # Extract the task line
            lines = content.split("\n")
            for line in lines:
                if "task is to:" in line.lower():
                    task = line.split("task is to:")[1].strip()
                    break

        # Format the example
        formatted_turns = []
        turn_num = 0

        for i, message in enumerate(trajectory):
            if (
                message["role"] == "user" and i > 0
            ):  # Skip first user message (it's the task)
                turn_num += 1
                observation = message["content"]

                # Get corresponding assistant response
                if i + 1 < len(trajectory) and trajectory[i + 1]["role"] == "assistant":
                    response = trajectory[i + 1]["content"]

                    formatted_turns.append(
                        f"Turn {turn_num}:\n" f"{observation}\n" f"Response: {response}"
                    )

        example_text = f"""ONE-SHOT EXAMPLE:
Task: {task}

{chr(10).join(formatted_turns)}"""

        return example_text

    def build_retrieval_query(
        self,
        goal: str,
        observation: str,
        inventory: str = "",
        recent_history: str = "",
        current_step: int = 0,
        current_reward: float = 0.0,
    ) -> str:
        """
        Build query matching AlfWorld indexing format.

        Format matches traj_retrieval/preprocess/alfworld/trajectory_entry.py:
        - state: observation ONLY (no inventory in AlfWorld)
        - progress: step_till_now ONLY (no current_reward in AlfWorld)

        This format MUST match the full_key format in create_entry():
        full_key = f"goal: {goal} | state: {state_repr} | context: {context_repr} | progress: {progress_repr}"
        """
        # Match the format from create_seq_to_seq_indices.py -> trajectory_entry.py
        state = f"observation: {observation}"  # NO INVENTORY for AlfWorld
        context = recent_history if recent_history else ""
        progress = f"step_till_now: {current_step}"  # NO current_reward for AlfWorld

        # Full key format (must match what's embedded in database)
        full_query = (
            f"goal: {goal} | state: {state} | context: {context} | progress: {progress}"
        )
        return full_query

    def build_user_message(
        self,
        goal_text: str,
        observation: str,
        inventory: str,
        admissible_actions: List[str],
        recent_history_str: str = "",
        trajectory_context: str = "",
        current_step: int = 0,
        max_steps: int = 0,
        **kwargs,
    ) -> str:
        """
        Build user message for ALFWorld following the consistent template format.

        Format: GOAL | CURRENT STEP | RECENT HISTORY | RETRIEVED TRAJECTORY | CURRENT OBSERVATION | ADMISSIBLE ACTIONS
        (No inventory for ALFWorld)

        Args:
            goal_text: The task goal
            observation: Current observation
            inventory: Not used for ALFWorld (kept for interface consistency)
            admissible_actions: List of valid actions
            recent_history_str: History of previous steps
            trajectory_context: Retrieved trajectory for guidance
            current_step: Current step number
            max_steps: Maximum steps allowed
            **kwargs: Additional context

        Returns:
            Formatted user message string
        """
        message_parts = []

        # Goal
        message_parts.append(f"GOAL: {goal_text}")

        # Current step
        if current_step > 0 or max_steps > 0:
            message_parts.append(f"\nCURRENT STEP: {current_step} / {max_steps}")

        # Recent history (if present) - with clear visual separation
        if recent_history_str and recent_history_str.strip():
            message_parts.append(
                "\n--- RECENT HISTORY (Previous Steps - For Reference Only) ---"
            )
            message_parts.append(recent_history_str)
            message_parts.append("--- End of Recent History ---\n")

        # Retrieved trajectory guidance (if present) - with clear visual separation
        if trajectory_context and trajectory_context.strip():
            message_parts.append(
                "\n--- RETRIEVED TRAJECTORY GUIDANCE (Reference Examples) ---"
            )
            message_parts.append(trajectory_context)
            message_parts.append("--- End of Trajectory Guidance ---\n")

        # Current observation - with clear emphasis
        message_parts.append(
            "\n>>> CURRENT OBSERVATION (Focus on This - Current State):"
        )
        message_parts.append(observation)
        message_parts.append("<<< End of Current Observation\n")

        # Admissible actions
        message_parts.append(f"ADMISSIBLE ACTIONS ({len(admissible_actions)} total):")
        for i, action in enumerate(admissible_actions, 1):
            message_parts.append(f"  {i}. {action}")

        # Response format instructions (always at the end)
        message_parts.append("\n\nRESPONSE FORMAT:")
        message_parts.append("You MUST respond with valid JSON in this exact format:")
        message_parts.append(
            '{"reasoning": "Let\'s think step by step. [your detailed reasoning]", "action": "exact action from admissible_actions"}'
        )
        message_parts.append("\nWhere:")
        message_parts.append(
            "- reasoning: MUST start with 'Let's think step by step.' Then explain your thought process, what you observe, and why this action is best"
        )
        message_parts.append(
            "- action: Must be EXACTLY one string from the admissible_actions list above (character-for-character match)"
        )
        message_parts.append("\nIMPORTANT:")
        message_parts.append(
            "1. Your reasoning MUST begin with 'Let's think step by step.'"
        )
        message_parts.append(
            "2. Do not include any text before or after the JSON object."
        )

        return "\n".join(message_parts)

    def build_system_message(self, task_name: Optional[str] = None) -> str:
        """
        Build ALFWorld-specific system message.

        Args:
            task_name: Optional task name for example selection

        Returns:
            Complete system message with instructions and example
        """
        # ALFWorld-specific system prompt
        system_prompt = """Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given the observation of the last turn. Think step by step about what you observe and what action to take next. The available actions are: 1. go to recep 2. take obj from recep 3. put obj in/on recep 4. open recep 5. close recep 6. toggle obj recep 7. clean obj with recep 8. heat obj with recep 9. cool obj with recep where obj and recep correspond to objects and receptacles. After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options. Reminder: 1. The action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. 2. Think when necessary, try to act directly more in the process.

RESPONSE FORMAT:
For each turn, you must provide your response in JSON format with two fields:
- reasoning: MUST start with 'Let's think step by step.' followed by detailed reasoning about the situation and why this action is the best choice
- action: your chosen action (must be exactly from admissible_actions list)

CRITICAL RULES:
1. Your action MUST be character-for-character identical to one item in admissible_actions
2. Do NOT modify, abbreviate, or paraphrase actions
3. Do NOT use actions from retrieved trajectories unless they appear in current admissible_actions
4. If confused, pick a safe exploratory action like "look" or navigation

"""

        # Get example episode for the specified task
        example = self.get_example_episode(task_name)

        return system_prompt + example

    def format_history_entry(
        self,
        step: int,
        observation: str,
        action: str,
        reward: float,
        reasoning: str = "",
        inventory: str = "",
        url: str = "",
    ) -> str:
        """
        Format a single history entry for ALFWorld.
        No inventory, no URL.
        """
        entry = f"STEP {step}: OBSERVATION: {observation}"

        # Add action and reward
        entry += f" | ACTION: {action} | REWARD: {reward}"

        # Add reasoning if available
        if reasoning and reasoning.strip():
            entry += f" | REASONING: {reasoning}"

        return entry
