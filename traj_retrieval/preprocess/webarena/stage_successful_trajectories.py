#!/usr/bin/env python3
"""
Stage successful agent trajectories to gold_repo_stage.

This script reads trajectories from agent evaluation runs, identifies successful ones,
and stages them in the same format as stage_gold_trajectories.py for consumption by
create_lancedb_indices.py.

Input Structure:
- Base path: traj_logs/run_X_eval_webarena_agentic/
- Folders: run_{task_name}/ (e.g., run_intent_template_id_4/)
- Files per folder:
  - {task_name}_trajectories_rag.json: Contains episodes and trajectories
  - {task_name}_trajectories_rag_detailed_path.json: Contains model info

Output Structure: gold_repo_stage/{task_name}/{variation_idx}/{trajectory_id}.json
Example: gold_repo_stage/intent_template_id_14/task_id_42/abc123.json

Task Naming Convention:
- task_name: intent_template_id_{id} (e.g., "intent_template_id_14")
- variation_idx: task_id_{task_id} (e.g., "task_id_42")

Success Criteria:
- Episode must have done=True AND finalScore=1.0
"""
import json
import os
import sys
import uuid
import argparse
from typing import Dict, List, Optional
from tqdm import tqdm
from pathlib import Path
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from staged_trajectory_schema import StagedTrajectorySchema  # noqa: E402

# ============================================================================
# CONFIGURATION
# ============================================================================

# Base path to agent run logs
TRAJ_LOGS_BASE_PATH = os.path.join(os.environ.get("MATM_DATA_ROOT", "environments"), "webarena/traj_logs/run_3_eval_webarena_agentic")

# Path to WebArena config files directory
CONFIG_PATH = (
    os.path.join(os.environ.get("MATM_DATA_ROOT", "environments"), "webarena/webarena/config_files")
)

# Gold staging directory
GOLD_REPO_STAGE_PATH = (
    os.path.join(os.environ.get("MATM_DATA_ROOT", "environments"), "webarena/gold_repo_stage")
)

# ============================================================================


def load_config_file(config_path: str) -> Optional[Dict]:
    """Load a WebArena config file."""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"  ⚠️ Warning: Failed to load config {config_path}: {e}")
        return None


class AgentTrajectoryStager:
    """Stages successful agent trajectories to gold_repo_stage."""

    def __init__(self, gold_repo_stage_path: str):
        self.gold_repo_stage_path = Path(gold_repo_stage_path)
        self.gold_repo_stage_path.mkdir(parents=True, exist_ok=True)
        self.config_cache = {}
        print(f"Gold staging path: {self.gold_repo_stage_path}")

    def _load_config(self, task_id: int, config_base_path: str) -> Optional[Dict]:
        """Load config file with caching."""
        config_filename = f"{task_id}.json"

        if config_filename in self.config_cache:
            return self.config_cache[config_filename]

        config_full_path = os.path.join(config_base_path, config_filename)
        config = load_config_file(config_full_path)

        self.config_cache[config_filename] = config
        return config

    def _is_successful_episode(self, episode: Dict) -> bool:
        """
        Check if an episode is successful.

        Success criteria:
        - done = True
        - finalScore = 1.0
        """
        return episode.get("done", False) and episode.get("finalScore", 0.0) == 1.0

    def _transform_action_to_trajectory_step(
        self, action: Dict, objective: str, step_idx: int, is_last: bool
    ) -> Dict:
        """
        Transform an action from our logs to the expected trajectory step format.

        Input format (from our logs):
        - action: action taken
        - observation: observation received
        - reasoning: reasoning for action
        - isCompleted: completion status
        - reward: step reward
        - score: current score
        - url: current URL

        Output format (expected by gold trajectories):
        - objective: task objective
        - url: current URL
        - observation: observation
        - done: completion status
        - reward: step reward
        - success: success score (1.0 if successful, 0.0 otherwise)
        - action: action taken
        - reason: reasoning
        - ... (other fields optional)
        """
        step = {
            "objective": objective,
            "url": action.get("url", ""),
            "observation": action.get("observation", ""),
            "done": action.get("isCompleted", False) or is_last,
            "reward": action.get("reward", 0.0),
            "success": 1.0 if (action.get("isCompleted", False) or is_last) else 0.0,
            "action": action.get("action", ""),
            "reason": action.get("reasoning", ""),
            "score": action.get("score", 0.0),
        }

        return step

    def _transform_episode_to_trajectory(self, episode: Dict) -> List[Dict]:
        """
        Transform an episode's action sequences to trajectory format.

        Args:
            episode: Episode data with actionSequences

        Returns:
            List of trajectory steps
        """
        objective = episode.get("taskDescription", "")
        action_sequences = episode.get("actionSequences", [])

        trajectory = []
        for i, action in enumerate(action_sequences):
            is_last = i == len(action_sequences) - 1
            step = self._transform_action_to_trajectory_step(
                action, objective, i, is_last
            )
            trajectory.append(step)

        # Ensure the last step has done=True and success=1.0 for successful trajectories
        if (
            trajectory
            and episode.get("done", False)
            and episode.get("finalScore", 0.0) == 1.0
        ):
            trajectory[-1]["done"] = True
            trajectory[-1]["success"] = 1.0

        return trajectory

    def _stage_trajectory(
        self,
        task: str,
        model: str,
        trajectory: List[Dict],
        split: str,
        task_name: str,
        variation_idx: str,
        source_file: str,
        agent_type: str,
    ) -> tuple:
        """
        Stage trajectory to gold_repo_stage with deduplication.

        Args:
            task: Path to config file
            model: Model name
            trajectory: List of trajectory steps
            split: Split name
            task_name: Formatted task name (intent_template_id_{id})
            variation_idx: Formatted variation ID (task_id_{task_id})
            source_file: Original source file path
            agent_type: Agent type for metadata

        Returns:
            Tuple of (output_file_path, is_duplicate)
        """
        task_dir = self.gold_repo_stage_path / str(task_name)
        variation_dir = task_dir / str(variation_idx)
        variation_dir.mkdir(parents=True, exist_ok=True)

        # Prepare trajectory data for comparison (without staging metadata)
        trajectory_data_for_comparison = {
            "task": task,
            "model": model,
            "trajectory": trajectory,
            "split": split,
        }
        trajectory_content_str = json.dumps(
            trajectory_data_for_comparison, sort_keys=True
        )

        # Check for duplicates by comparing entire content
        for existing_file in variation_dir.glob("*.json"):
            try:
                with open(existing_file, "r") as f:
                    existing_data = json.load(f)

                # Remove staging metadata from existing data for comparison
                existing_content = {
                    k: v
                    for k, v in existing_data.items()
                    if k not in ["trajectory_id", "source_file", "metadata_info"]
                }
                existing_content_str = json.dumps(existing_content, sort_keys=True)

                if trajectory_content_str == existing_content_str:
                    return str(existing_file), True

            except (json.JSONDecodeError, IOError):
                continue

        # No duplicate, create staged trajectory with metadata
        trajectory_id = str(uuid.uuid4())
        trajectory_data = StagedTrajectorySchema.create(
            task=task,
            model=model,
            trajectory=trajectory,
            split=split,
            trajectory_id=trajectory_id,
            source_file=source_file,
            agent_type=agent_type,
        )

        output_file = variation_dir / f"{trajectory_id}.json"
        with open(output_file, "w") as f:
            json.dump(trajectory_data, f, indent=2)

        return str(output_file), False

    def _process_task_folder(self, task_folder: str, config_path: str) -> tuple:
        """
        Process a single task folder (e.g., run_intent_template_id_4).

        Returns:
            Tuple of (staged_count, duplicates, skipped_unsuccessful, skipped_no_config)
        """
        task_name = os.path.basename(task_folder).replace("run_", "")

        # Paths to the two required files
        traj_file = os.path.join(task_folder, f"{task_name}_trajectories_rag.json")
        detailed_file = os.path.join(
            task_folder, f"{task_name}_trajectories_rag_detailed_path.json"
        )

        # Check if files exist
        if not os.path.exists(traj_file):
            return 0, 0, 0, 0
        if not os.path.exists(detailed_file):
            return 0, 0, 0, 0

        # Load trajectory data
        try:
            with open(traj_file, "r") as f:
                traj_data = json.load(f)
        except Exception as e:
            print(f"  ❌ Error loading {traj_file}: {e}")
            return 0, 0, 0, 0

        # Load detailed data for model info
        try:
            with open(detailed_file, "r") as f:
                detailed_data = json.load(f)
        except Exception as e:
            print(f"  ❌ Error loading {detailed_file}: {e}")
            return 0, 0, 0, 0

        # Extract model name
        model_name = detailed_data.get("model", "unknown")

        # Process episodes
        episodes = traj_data.get("episodes", [])

        staged_count = 0
        duplicate_count = 0
        unsuccessful_count = 0
        no_config_count = 0

        for episode in episodes:
            # Check if successful
            if not self._is_successful_episode(episode):
                unsuccessful_count += 1
                continue

            # Extract variation info
            variation_idx_num = episode.get("variationIdx")
            if variation_idx_num is None:
                no_config_count += 1
                continue

            # Load config to get intent_template_id
            config = self._load_config(variation_idx_num, config_path)
            if not config:
                no_config_count += 1
                continue

            intent_template_id = config.get("intent_template_id")
            config_task_id = config.get("task_id")

            if intent_template_id is None or config_task_id is None:
                no_config_count += 1
                continue

            # Format names
            task_name_formatted = f"intent_template_id_{intent_template_id}"
            variation_idx_formatted = f"task_id_{config_task_id}"

            # Transform episode to trajectory
            trajectory = self._transform_episode_to_trajectory(episode)

            if not trajectory:
                continue

            # Prepare trajectory data
            config_file_path = os.path.join(config_path, f"{config_task_id}.json")
            split = episode.get("fold", "test")

            # Stage trajectory
            saved_file, is_duplicate = self._stage_trajectory(
                task=config_file_path,
                model=model_name,
                trajectory=trajectory,
                split=split,
                task_name=task_name_formatted,
                variation_idx=variation_idx_formatted,
                source_file=traj_file,
                agent_type=model_name,
            )

            if is_duplicate:
                duplicate_count += 1
            else:
                staged_count += 1

        return staged_count, duplicate_count, unsuccessful_count, no_config_count

    def stage_trajectories(self, traj_logs_base_path: str, config_path: str):
        """Stage all successful trajectories from agent run logs."""
        total_staged = 0
        total_duplicates = 0
        total_unsuccessful = 0
        total_no_config = 0

        # Find all task folders
        task_folders = sorted(
            glob.glob(os.path.join(traj_logs_base_path, "run_intent_template_id_*"))
        )

        if not task_folders:
            print(f"⚠️  Warning: No task folders found in {traj_logs_base_path}")
            return

        print(f"Found {len(task_folders)} task folders\n")

        for task_folder in tqdm(
            task_folders, desc="Processing task folders", unit="folder"
        ):
            staged, duplicates, unsuccessful, no_config = self._process_task_folder(
                task_folder, config_path
            )
            total_staged += staged
            total_duplicates += duplicates
            total_unsuccessful += unsuccessful
            total_no_config += no_config

        print(f"\n{'='*70}")
        print(f"Staging Summary")
        print(f"{'='*70}")
        print(f"✓ Successful trajectories staged: {total_staged:,}")
        print(f"✓ Duplicates skipped: {total_duplicates:,}")
        print(f"✓ Unsuccessful trajectories skipped: {total_unsuccessful:,}")
        print(f"✓ Trajectories without config skipped: {total_no_config:,}")
        print(f"\n✓ Staged trajectories saved to: {GOLD_REPO_STAGE_PATH}")
        print(f"✓ Naming convention:")
        print(f"  - task_name: intent_template_id_{{id}}")
        print(f"  - variation_idx: task_id_{{task_id}}")
        print(f"{'='*70}\n")


def main():
    """Main function to stage successful agent trajectories."""
    parser = argparse.ArgumentParser(
        description="Stage successful agent trajectories to gold_repo_stage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths
  python stage_successful_trajectories.py
  
  # Specify custom paths
  python stage_successful_trajectories.py \\
    --traj-logs /path/to/traj_logs/run_X_eval_webarena_agentic \\
    --config /path/to/config_files \\
    --output /path/to/gold_repo_stage
        """,
    )

    parser.add_argument(
        "--traj-logs",
        default=TRAJ_LOGS_BASE_PATH,
        help=f"Path to trajectory logs base directory (default: {TRAJ_LOGS_BASE_PATH})",
    )
    parser.add_argument(
        "--config",
        default=CONFIG_PATH,
        help=f"Path to config files directory (default: {CONFIG_PATH})",
    )
    parser.add_argument(
        "--output",
        default=GOLD_REPO_STAGE_PATH,
        help=f"Path to gold staging directory (default: {GOLD_REPO_STAGE_PATH})",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("WebArena Successful Agent Trajectory Stager")
    print("=" * 70)
    print(f"Trajectory logs base path: {args.traj_logs}")
    print(f"Config files: {args.config}")
    print(f"Gold staging path: {args.output}")
    print(f"Naming convention:")
    print(f"  - task_name: intent_template_id_{{id}}")
    print(f"  - variation_idx: task_id_{{task_id}}")
    print(f"Success criteria: done=True AND finalScore=1.0")
    print("=" * 70)
    print()

    # Verify paths
    if not os.path.exists(args.traj_logs):
        print(f"ERROR: Trajectory logs path not found: {args.traj_logs}")
        sys.exit(1)

    if not os.path.exists(args.config):
        print(f"ERROR: Config path not found: {args.config}")
        sys.exit(1)

    # Create stager
    stager = AgentTrajectoryStager(gold_repo_stage_path=args.output)

    # Stage trajectories
    stager.stage_trajectories(args.traj_logs, args.config)

    print("\n" + "=" * 70)
    print("✓ Staging completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
