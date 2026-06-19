#!/usr/bin/env python3
"""
Standalone trajectory evaluator that analyzes agent performance without gold comparison.
Computes success rates, failure rates, progress metrics, and detailed statistics.

Success criteria:
- AlfWorld: success=1, failure=0
- WebArena: success when finalScore == 1
"""
import json
import os
import argparse
import numpy as np
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# GLOBAL CONFIGURATION
# Base path for trajectory logs
BASE_TRAJ_LOGS_PATH = os.environ.get("MATM_DATA_ROOT", "environments")

# Environment name (e.g., "alfworld", "webarena")
ENVIRONMENT_NAME = "alfworld"

# Output directory name (e.g., "traj_logs" or "simulation_traj_logs")
OUTPUT_DIR_NAME = "simulation_traj_logs"
# OUTPUT_DIR_NAME = "simulation_traj_logs"  # Alternative for simulation logs

# Run ID prefix - the script will find all folders starting with this prefix
# Example: "run_eval_agentic_gpt_oss_20b" will match:
#   - run_eval_agentic_gpt_oss_20b_1
#   - run_eval_agentic_gpt_oss_20b_2
#   - etc.
# If only one folder matches, it uses existing behavior
RUN_ID_PREFIX = (
    "run_e6_eval_simulate_till_tk_null_alfworld_gpt_oss_20b_run_number_2_rank_5_job"
)


def find_matching_run_folders(environment_name: str, run_id_prefix: str) -> List[Path]:
    """
    Find all folders matching the run_id_prefix in the environment's traj_logs directory.

    Args:
        environment_name: Name of the environment (e.g., "alfworld", "webarena")
        run_id_prefix: Prefix to match folder names (e.g., "run_eval_agentic_gpt_oss_20b")

    Returns:
        List of matching folder paths, sorted
    """
    base_path = Path(BASE_TRAJ_LOGS_PATH) / environment_name / OUTPUT_DIR_NAME

    if not base_path.exists():
        print(f"⚠️  Base trajectory logs path not found: {base_path}")
        return []

    matching_folders = []
    for folder in base_path.iterdir():
        if folder.is_dir() and folder.name.startswith(run_id_prefix):
            matching_folders.append(folder)

    return sorted(matching_folders)


def merge_trajectory_files(source_file: Path, dest_file: Path):
    """
    Merge episodes from source trajectory file into destination trajectory file.

    Args:
        source_file: Source trajectory JSON file
        dest_file: Destination trajectory JSON file to merge into
    """
    # Load both files
    with source_file.open("r", encoding="utf-8") as f:
        source_data = json.load(f)

    with dest_file.open("r", encoding="utf-8") as f:
        dest_data = json.load(f)

    # Merge episodes arrays
    if "episodes" in source_data and "episodes" in dest_data:
        dest_data["episodes"].extend(source_data["episodes"])

        # Update episode count if present
        if "total_episodes" in dest_data:
            dest_data["total_episodes"] = len(dest_data["episodes"])

    # Save merged data back to destination
    with dest_file.open("w", encoding="utf-8") as f:
        json.dump(dest_data, f, indent=2, ensure_ascii=False)


def extract_metadata_from_evaluation_results(folder: Path) -> Optional[Dict]:
    """
    Extract metadata from evaluation_results JSON file in a run folder.
    Normalizes metadata structure to ensure search_config is nested inside retrieval_config.

    Args:
        folder: Path to the run folder

    Returns:
        Dict with normalized metadata or None if not found
    """
    # Look for evaluation_results JSON files
    for file in folder.iterdir():
        if (
            file.is_file()
            and file.name.startswith("evaluation_results_")
            and file.suffix == ".json"
        ):
            try:
                with file.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                if "evaluation_metadata" in data:
                    metadata = dict(data["evaluation_metadata"])

                    # Normalize metadata structure: move search_config inside retrieval_config if it's at top level
                    # This ensures consistency across all environments (matching run_webarena.py format)
                    if "search_config" in metadata and "retrieval_config" in metadata:
                        # Check if search_config is already nested inside retrieval_config
                        if "search_config" not in metadata.get("retrieval_config", {}):
                            # Move search_config into retrieval_config
                            metadata["retrieval_config"]["search_config"] = metadata[
                                "search_config"
                            ]
                            # Remove from top level
                            del metadata["search_config"]
                            print(
                                f"      ✓ Normalized metadata: moved search_config into retrieval_config"
                            )

                    return metadata
            except Exception as e:
                print(f"      ⚠️  Failed to extract metadata from {file.name}: {e}")
                continue

    return None


def extract_diversity_strategy(combined_folder: Path) -> Tuple[bool, Optional[str]]:
    """
    Extract diversity_strategy from detailed path JSON files in the combined folder.

    Args:
        combined_folder: Path to the combined run folder

    Returns:
        Tuple of (found, value) where:
        - found: Boolean indicating if the key was found
        - value: The diversity_strategy value (can be None if explicitly set to null)
    """
    # Look for any task directory
    for task_dir in combined_folder.iterdir():
        if task_dir.is_dir() and task_dir.name.startswith("run_"):
            # Look for detailed_path.json files
            for file in task_dir.iterdir():
                if file.is_file() and file.name.endswith("_detailed_path.json"):
                    try:
                        with file.open("r", encoding="utf-8") as f:
                            data = json.load(f)

                        # Search for diversity_strategy in the JSON data
                        def find_diversity_strategy(obj, path=""):
                            """Recursively search for diversity_strategy key.
                            Returns (found, value) tuple."""
                            if isinstance(obj, dict):
                                if "diversity_strategy" in obj:
                                    return (True, obj["diversity_strategy"])
                                for key, value in obj.items():
                                    found, result = find_diversity_strategy(
                                        value, f"{path}.{key}"
                                    )
                                    if found:
                                        return (found, result)
                            elif isinstance(obj, list):
                                for idx, item in enumerate(obj):
                                    found, result = find_diversity_strategy(
                                        item, f"{path}[{idx}]"
                                    )
                                    if found:
                                        return (found, result)
                            return (False, None)

                        found, diversity = find_diversity_strategy(data)
                        if found:
                            print(f"   ✓ Found diversity_strategy: {diversity}")
                            return (True, diversity)
                    except Exception as e:
                        print(f"      ⚠️  Failed to read {file.name}: {e}")
                        continue

    return (False, None)


def combine_run_folders(
    matching_folders: List[Path], run_id_prefix: str
) -> Tuple[Path, Dict]:
    """
    Combine multiple run folders into a single folder with the common prefix name.
    Performs deep merge of episodes for duplicate task directories.

    Args:
        matching_folders: List of folder paths to combine
        run_id_prefix: Common prefix to use for the combined folder name

    Returns:
        Tuple of (Path to the combined folder, metadata dict)
    """
    if not matching_folders:
        raise ValueError("No folders to combine")

    # Create combined folder in the same parent directory as the matching folders
    parent_dir = matching_folders[0].parent
    combined_folder = parent_dir / run_id_prefix

    print(f"\n🔄 Combining {len(matching_folders)} run folders into: {combined_folder}")

    # Create the combined folder if it doesn't exist
    combined_folder.mkdir(parents=True, exist_ok=True)

    # Extract metadata from the first folder (should be common across all)
    print(f"\n📋 Extracting metadata from run folders...")
    combined_metadata = None
    for folder in matching_folders:
        metadata = extract_metadata_from_evaluation_results(folder)
        if metadata:
            if combined_metadata is None:
                combined_metadata = metadata
                print(f"   ✓ Extracted metadata from: {folder.name}")
            else:
                # Verify metadata is consistent across folders
                for key, value in metadata.items():
                    if combined_metadata.get(key) != value and value is not None:
                        print(
                            f"      ⚠️  Metadata mismatch for '{key}': {combined_metadata.get(key)} vs {value}"
                        )

    if combined_metadata is None:
        print(f"   ⚠️  No metadata found in evaluation results files")
        combined_metadata = {}

    # Track which task directories we've processed
    all_task_dirs = set()

    # Combine all task directories from matching folders
    for folder in matching_folders:
        print(f"   Processing: {folder.name}")

        for item in folder.iterdir():
            if item.is_dir() and item.name.startswith("run_"):
                task_name = item.name

                # Destination path in combined folder
                dest_path = combined_folder / task_name

                if task_name in all_task_dirs:
                    # Deep merge: merge trajectory files
                    print(f"      🔄 Merging episodes for duplicate task: {task_name}")

                    # Find all trajectory JSON files (skip metrics files)
                    for source_file in item.iterdir():
                        if source_file.is_file() and source_file.suffix == ".json":
                            # Skip metrics files
                            if source_file.name.startswith("metrics_"):
                                continue

                            # Check if it's a trajectory file
                            if "trajectories" in source_file.name:
                                dest_file = dest_path / source_file.name

                                if dest_file.exists():
                                    try:
                                        merge_trajectory_files(source_file, dest_file)
                                        print(f"         ✓ Merged: {source_file.name}")
                                    except Exception as e:
                                        print(
                                            f"         ⚠️  Failed to merge {source_file.name}: {e}"
                                        )
                                else:
                                    # File doesn't exist in dest, just copy it
                                    shutil.copy2(source_file, dest_file)
                                    print(f"         ✓ Copied: {source_file.name}")
                else:
                    # First time seeing this task directory, copy it entirely
                    if not dest_path.exists():
                        shutil.copytree(item, dest_path)
                        all_task_dirs.add(task_name)
                        print(f"      ✓ Copied: {task_name}")

    # Extract diversity_strategy from detailed path files
    print(f"\n🔍 Extracting diversity_strategy from detailed path files...")
    found, diversity_strategy = extract_diversity_strategy(combined_folder)
    if found:
        combined_metadata["diversity_strategy"] = diversity_strategy
    else:
        print(f"   ⚠️  No diversity_strategy found in detailed path files")

    print(f"✅ Combined folder created with {len(all_task_dirs)} task directories")
    return combined_folder, combined_metadata


def resolve_trajectory_logs_path(
    environment_name: str, run_id_prefix: str
) -> Tuple[Path, bool, Dict]:
    """
    Resolve the trajectory logs path based on environment name and run ID prefix.

    Args:
        environment_name: Environment name for constructing path
        run_id_prefix: Run ID prefix to search for

    Returns:
        Tuple of (resolved_path, is_combined, metadata) where:
        - resolved_path: Path to the trajectory logs directory
        - is_combined: Boolean indicating if multiple folders were combined
        - metadata: Dict with extracted metadata (empty if single folder or not found)
    """
    matching_folders = find_matching_run_folders(environment_name, run_id_prefix)

    if not matching_folders:
        raise ValueError(
            f"No folders found matching prefix '{run_id_prefix}' in "
            f"{BASE_TRAJ_LOGS_PATH}/{environment_name}/{OUTPUT_DIR_NAME}/"
        )

    print(f"🔍 Found {len(matching_folders)} matching folder(s):")
    for folder in matching_folders:
        print(f"   - {folder.name}")

    # If only one folder, use it directly
    if len(matching_folders) == 1:
        print(f"✓ Using single folder: {matching_folders[0]}")
        # Extract metadata even for single folder
        metadata = extract_metadata_from_evaluation_results(matching_folders[0])
        if not metadata:
            metadata = {}
        # Also try to get diversity_strategy
        found, diversity_strategy = extract_diversity_strategy(matching_folders[0])
        if found:
            metadata["diversity_strategy"] = diversity_strategy
        return matching_folders[0], False, metadata

    # If multiple folders, combine them
    else:
        combined_path, metadata = combine_run_folders(matching_folders, run_id_prefix)
        return combined_path, True, metadata


class TrajectoryEvaluator:
    """
    Evaluates agent trajectories and computes performance metrics without requiring gold trajectories.
    """

    def __init__(
        self,
        environment_name: str,
        run_id_prefix: str,
        specific_task: Optional[str] = None,
        verbose: bool = False,
        exclude_error_episodes: bool = False,
    ):
        # Resolve the trajectory logs path
        resolved_path, is_combined, run_metadata = resolve_trajectory_logs_path(
            environment_name=environment_name, run_id_prefix=run_id_prefix
        )

        self.trajectory_logs_dir = resolved_path
        self.is_combined = is_combined
        self.run_metadata = run_metadata
        self.environment_name = environment_name
        self.run_id_prefix = run_id_prefix
        self.specific_task = specific_task
        self.verbose = verbose
        self.exclude_error_episodes = exclude_error_episodes

        # Output directory: input_dir/evaluations
        self.output_dir = self.trajectory_logs_dir / "evaluations"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Extract run name from trajectory logs directory
        self.run_name = self.trajectory_logs_dir.name

        print(f"\n🔍 Trajectory Evaluator initialized:")
        print(f"   - Environment: {self.environment_name}")
        print(f"   - Run ID prefix: {self.run_id_prefix}")
        print(f"   - Trajectory logs: {self.trajectory_logs_dir}")
        print(f"   - Is combined: {self.is_combined}")
        print(f"   - Output directory: {self.output_dir}")
        print(f"   - Run name: {self.run_name}")
        if self.run_metadata:
            print(f"   - Metadata extracted: {len(self.run_metadata)} fields")
        if self.specific_task:
            print(f"   - Specific task: {self.specific_task}")
        if self.exclude_error_episodes:
            print(f"   - Excluding error episodes: enabled")

    def load_task_trajectories(self, task_name: str) -> Optional[Dict]:
        """Load trajectories for a specific task."""
        task_dir = self.trajectory_logs_dir / f"run_{task_name}"

        # Try different possible trajectory file names
        possible_files = [
            f"{task_name}_trajectories_rag.json",
            f"{task_name}_trajectories.json",
            f"{task_name}_trajectories_agentic.json",
        ]

        for filename in possible_files:
            traj_file = task_dir / filename
            if traj_file.exists():
                with traj_file.open("r", encoding="utf-8") as f:
                    return json.load(f)

        return None

    def is_problematic_episode(self, episode: Dict) -> Tuple[bool, str]:
        """
        Check if an episode should be excluded.
        Only excludes episodes with empty actionSequences or trajectory: None.

        Args:
            episode: Episode dictionary to check

        Returns:
            Tuple of (is_problematic, reason) where:
            - is_problematic: Boolean indicating if episode should be excluded
            - reason: String describing why it's problematic (empty if not problematic)
        """
        # Check for empty or missing actionSequences
        action_sequences = episode.get("actionSequences")
        if not action_sequences or len(action_sequences) == 0:
            return (True, "empty_actionSequences")

        # Check for missing trajectory data (if trajectory field exists and is None)
        if "trajectory" in episode and episode["trajectory"] is None:
            return (True, "trajectory=None")

        return (False, "")

    def classify_episode(self, episode: Dict) -> str:
        """
        Classify an episode as 'success', 'failure', or 'in_progress'.

        Classification rules:
        - AlfWorld: success=1, failure=0
        """
        if not episode.get("actionSequences"):
            return "unknown"

        final_score = episode["actionSequences"][-1].get("score", 0)

        # Detect environment from trajectory logs path
        trajectory_logs_lower = str(self.trajectory_logs_dir).lower()

        if "alfworld" in trajectory_logs_lower:
            # AlfWorld: 1 = success, 0 = failure
            if final_score == 1:
                return "success"
            elif final_score == 0:
                return "failure"
            else:
                return "in_progress"

        else:
            # Unknown environment - use generic thresholds
            if final_score >= 1.0:
                return "success"
            elif final_score <= 0.0:
                return "failure"
            else:
                return "in_progress"

    def analyze_episode(self, episode: Dict) -> Dict:
        """
        Analyze a single episode and extract key metrics.

        Returns:
            Dict with episode metrics
        """
        if not episode.get("actionSequences"):
            return {
                "variation_idx": episode.get("variationIdx", -1),
                "classification": "unknown",
                "total_steps": 0,
                "final_score": 0,
                "max_steps": episode.get("max_steps", 0),
            }

        classification = self.classify_episode(episode)
        total_steps = len(episode["actionSequences"])
        final_score = episode["actionSequences"][-1].get("score", 0)
        max_steps = episode.get("max_steps", total_steps)

        # Get score progression
        scores = [step.get("score", 0) for step in episode["actionSequences"]]

        # Count unique actions
        actions = [step.get("action", "") for step in episode["actionSequences"]]
        unique_actions = len(set(actions))

        return {
            "variation_idx": episode.get("variationIdx", -1),
            "classification": classification,
            "total_steps": total_steps,
            "final_score": final_score,
            "max_steps": max_steps,
            "reached_max_steps": total_steps >= max_steps,
            "unique_actions": unique_actions,
            "score_progression": scores,
            "initial_score": scores[0] if scores else 0,
            "score_improvement": final_score - (scores[0] if scores else 0),
        }

    def evaluate_task(self, task_name: str) -> Optional[Dict]:
        """
        Evaluate all variations of a specific task.

        Returns:
            Dict with task-level evaluation results
        """
        print(f"\n📊 Evaluating task: {task_name}")

        try:
            trajectories = self.load_task_trajectories(task_name)

            if not trajectories or "episodes" not in trajectories:
                print(f"   ⚠️  No trajectories found for task {task_name}")
                return None

            episodes = trajectories["episodes"]
            print(f"   Found {len(episodes)} episodes")

            # Filter out problematic episodes if requested
            excluded_episodes = []
            valid_episodes = []

            if self.exclude_error_episodes:
                for episode in episodes:
                    is_problematic, reason = self.is_problematic_episode(episode)
                    if is_problematic:
                        excluded_episodes.append(
                            {
                                "variation_idx": episode.get("variationIdx", -1),
                                "reason": reason,
                            }
                        )
                        if self.verbose:
                            print(
                                f"     ⚠️  Excluding variation {episode.get('variationIdx', -1)}: {reason}"
                            )
                    else:
                        valid_episodes.append(episode)
            else:
                valid_episodes = episodes

            excluded_count = len(excluded_episodes)
            if excluded_count > 0:
                print(f"   ⚠️  Excluded {excluded_count} problematic episode(s)")

            # Analyze each valid episode
            episode_analyses = []
            for episode in valid_episodes:
                analysis = self.analyze_episode(episode)
                episode_analyses.append(analysis)

                if self.verbose:
                    print(
                        f"     Variation {analysis['variation_idx']}: "
                        f"{analysis['classification']}, "
                        f"steps={analysis['total_steps']}, "
                        f"score={analysis['final_score']}"
                    )

            # Aggregate task-level metrics
            total_episodes = len(episode_analyses)

            # Count classifications
            success_count = sum(
                1 for ep in episode_analyses if ep["classification"] == "success"
            )
            failure_count = sum(
                1 for ep in episode_analyses if ep["classification"] == "failure"
            )
            in_progress_count = sum(
                1 for ep in episode_analyses if ep["classification"] == "in_progress"
            )

            # Success episodes
            success_episodes = [
                ep for ep in episode_analyses if ep["classification"] == "success"
            ]
            success_steps = [ep["total_steps"] for ep in success_episodes]
            success_scores = [ep["final_score"] for ep in success_episodes]

            # Failure episodes
            failure_episodes = [
                ep for ep in episode_analyses if ep["classification"] == "failure"
            ]
            failure_steps = [ep["total_steps"] for ep in failure_episodes]
            failure_scores = [ep["final_score"] for ep in failure_episodes]

            # In-progress episodes
            in_progress_episodes = [
                ep for ep in episode_analyses if ep["classification"] == "in_progress"
            ]
            in_progress_steps = [ep["total_steps"] for ep in in_progress_episodes]
            in_progress_scores = [ep["final_score"] for ep in in_progress_episodes]

            # Combined success + in_progress (non-failed episodes)
            non_failed_episodes = success_episodes + in_progress_episodes
            non_failed_steps = [ep["total_steps"] for ep in non_failed_episodes]
            non_failed_scores = [ep["final_score"] for ep in non_failed_episodes]

            # All episodes metrics
            all_steps = [ep["total_steps"] for ep in episode_analyses]
            all_scores = [ep["final_score"] for ep in episode_analyses]
            all_max_steps = [ep["max_steps"] for ep in episode_analyses]

            # Helper function for safe mean calculation
            def safe_mean(values):
                return round(float(np.mean(values)), 2) if values else 0.0

            def safe_std(values):
                return round(float(np.std(values)), 2) if values else 0.0

            def safe_min(values):
                return int(min(values)) if values else 0

            def safe_max(values):
                return int(max(values)) if values else 0

            task_results = {
                "task_name": task_name,
                "total_episodes": total_episodes,
                "excluded_episodes_count": len(excluded_episodes),
                "excluded_episodes": excluded_episodes if excluded_episodes else None,
                # Classification counts and percentages
                "success_count": success_count,
                "failure_count": failure_count,
                "in_progress_count": in_progress_count,
                "success_percentage": round(success_count / total_episodes * 100, 2)
                if total_episodes > 0
                else 0.0,
                "failure_percentage": round(failure_count / total_episodes * 100, 2)
                if total_episodes > 0
                else 0.0,
                "in_progress_percentage": round(
                    in_progress_count / total_episodes * 100, 2
                )
                if total_episodes > 0
                else 0.0,
                # Success metrics
                "success_metrics": {
                    "avg_steps": safe_mean(success_steps),
                    "std_steps": safe_std(success_steps),
                    "min_steps": safe_min(success_steps),
                    "max_steps": safe_max(success_steps),
                    "avg_score": safe_mean(success_scores),
                    "all_scores": success_scores,
                },
                # Failure metrics
                "failure_metrics": {
                    "avg_steps": safe_mean(failure_steps),
                    "std_steps": safe_std(failure_steps),
                    "min_steps": safe_min(failure_steps),
                    "max_steps": safe_max(failure_steps),
                    "avg_score": safe_mean(failure_scores),
                    "all_scores": failure_scores,
                },
                # In-progress metrics
                "in_progress_metrics": {
                    "avg_steps": safe_mean(in_progress_steps),
                    "std_steps": safe_std(in_progress_steps),
                    "min_steps": safe_min(in_progress_steps),
                    "max_steps": safe_max(in_progress_steps),
                    "avg_score": safe_mean(in_progress_scores),
                    "std_score": safe_std(in_progress_scores),
                    "min_score": safe_min(in_progress_scores),
                    "max_score": safe_max(in_progress_scores),
                    "all_scores": in_progress_scores,
                },
                # Non-failed (success + in_progress) metrics
                "non_failed_metrics": {
                    "count": len(non_failed_episodes),
                    "avg_steps": safe_mean(non_failed_steps),
                    "std_steps": safe_std(non_failed_steps),
                    "avg_score": safe_mean(non_failed_scores),
                    "std_score": safe_std(non_failed_scores),
                    "min_score": safe_min(non_failed_scores),
                    "max_score": safe_max(non_failed_scores),
                },
                # Overall metrics (all episodes)
                "overall_metrics": {
                    "avg_steps": safe_mean(all_steps),
                    "std_steps": safe_std(all_steps),
                    "avg_score": safe_mean(all_scores),
                    "std_score": safe_std(all_scores),
                    "avg_max_steps": safe_mean(all_max_steps),
                    "reached_max_steps_count": sum(
                        1 for ep in episode_analyses if ep["reached_max_steps"]
                    ),
                },
                # Detailed episode data
                "episode_details": episode_analyses,
            }

            return task_results

        except Exception as e:
            print(f"   ❌ Error evaluating task {task_name}: {e}")
            import traceback

            traceback.print_exc()
            return None

    def get_available_tasks(self) -> List[str]:
        """Get list of tasks available in trajectory logs."""
        tasks = []

        if not self.trajectory_logs_dir.exists():
            print(
                f"⚠️  Trajectory logs directory not found: {self.trajectory_logs_dir}"
            )
            return tasks

        for task_dir in self.trajectory_logs_dir.iterdir():
            if task_dir.is_dir() and task_dir.name.startswith("run_"):
                task_name = task_dir.name[4:]  # Remove "run_" prefix
                tasks.append(task_name)

        if self.specific_task:
            if self.specific_task in tasks:
                return [self.specific_task]
            else:
                print(
                    f"⚠️  Specific task '{self.specific_task}' not found in available tasks: {tasks}"
                )
                return []

        return sorted(tasks)

    def run_evaluation(self) -> Dict:
        """
        Run complete evaluation of all tasks.

        Returns:
            Dict with global evaluation results
        """
        print(f"\n🚀 Starting trajectory evaluation...")
        print(f"   Run: {self.run_name}")
        print("=" * 80)

        available_tasks = self.get_available_tasks()

        if not available_tasks:
            raise ValueError(
                f"No tasks found in trajectory logs directory: {self.trajectory_logs_dir}"
            )

        print(f"📋 Found {len(available_tasks)} tasks to evaluate: {available_tasks}")

        # Evaluate each task
        task_results = {}
        global_metrics = {
            "total_episodes": 0,
            "excluded_episodes_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "in_progress_count": 0,
            "all_steps": [],
            "all_scores": [],
            "success_steps": [],
            "success_scores": [],
            "failure_steps": [],
            "failure_scores": [],
            "in_progress_steps": [],
            "in_progress_scores": [],
            "non_failed_steps": [],
            "non_failed_scores": [],
        }

        for task_name in available_tasks:
            task_result = self.evaluate_task(task_name)

            if task_result:
                task_results[task_name] = task_result

                # Aggregate global metrics
                global_metrics["total_episodes"] += task_result["total_episodes"]
                global_metrics["excluded_episodes_count"] += task_result.get(
                    "excluded_episodes_count", 0
                )
                global_metrics["success_count"] += task_result["success_count"]
                global_metrics["failure_count"] += task_result["failure_count"]
                global_metrics["in_progress_count"] += task_result["in_progress_count"]

                # Collect all episode data for global statistics
                for episode in task_result["episode_details"]:
                    global_metrics["all_steps"].append(episode["total_steps"])
                    global_metrics["all_scores"].append(episode["final_score"])

                    if episode["classification"] == "success":
                        global_metrics["success_steps"].append(episode["total_steps"])
                        global_metrics["success_scores"].append(episode["final_score"])
                        global_metrics["non_failed_steps"].append(
                            episode["total_steps"]
                        )
                        global_metrics["non_failed_scores"].append(
                            episode["final_score"]
                        )
                    elif episode["classification"] == "failure":
                        global_metrics["failure_steps"].append(episode["total_steps"])
                        global_metrics["failure_scores"].append(episode["final_score"])
                    elif episode["classification"] == "in_progress":
                        global_metrics["in_progress_steps"].append(
                            episode["total_steps"]
                        )
                        global_metrics["in_progress_scores"].append(
                            episode["final_score"]
                        )
                        global_metrics["non_failed_steps"].append(
                            episode["total_steps"]
                        )
                        global_metrics["non_failed_scores"].append(
                            episode["final_score"]
                        )

        # Calculate global statistics
        def safe_mean(values):
            return round(float(np.mean(values)), 2) if values else 0.0

        def safe_std(values):
            return round(float(np.std(values)), 2) if values else 0.0

        def safe_min(values):
            return int(min(values)) if values else 0

        def safe_max(values):
            return int(max(values)) if values else 0

        total_episodes = global_metrics["total_episodes"]

        global_summary = {
            "run_name": self.run_name,
            "trajectory_logs_dir": str(self.trajectory_logs_dir),
            "is_combined_run": self.is_combined,
            "run_metadata": self.run_metadata,
            "evaluation_timestamp": datetime.now().isoformat(),
            "exclude_error_episodes": self.exclude_error_episodes,
            "total_tasks": len(task_results),
            "total_episodes": total_episodes,
            "excluded_episodes_count": global_metrics["excluded_episodes_count"],
            # Global average score (across ALL episodes)
            "global_avg_score": safe_mean(global_metrics["all_scores"]),
            # Global classification statistics
            "global_success_count": global_metrics["success_count"],
            "global_failure_count": global_metrics["failure_count"],
            "global_in_progress_count": global_metrics["in_progress_count"],
            "global_success_percentage": round(
                global_metrics["success_count"] / total_episodes * 100, 2
            )
            if total_episodes > 0
            else 0.0,
            "global_failure_percentage": round(
                global_metrics["failure_count"] / total_episodes * 100, 2
            )
            if total_episodes > 0
            else 0.0,
            "global_in_progress_percentage": round(
                global_metrics["in_progress_count"] / total_episodes * 100, 2
            )
            if total_episodes > 0
            else 0.0,
            # Global success metrics
            "global_success_metrics": {
                "count": len(global_metrics["success_steps"]),
                "avg_steps": safe_mean(global_metrics["success_steps"]),
                "std_steps": safe_std(global_metrics["success_steps"]),
                "min_steps": safe_min(global_metrics["success_steps"]),
                "max_steps": safe_max(global_metrics["success_steps"]),
                "avg_score": safe_mean(global_metrics["success_scores"]),
            },
            # Global failure metrics
            "global_failure_metrics": {
                "count": len(global_metrics["failure_steps"]),
                "avg_steps": safe_mean(global_metrics["failure_steps"]),
                "std_steps": safe_std(global_metrics["failure_steps"]),
                "min_steps": safe_min(global_metrics["failure_steps"]),
                "max_steps": safe_max(global_metrics["failure_steps"]),
                "avg_score": safe_mean(global_metrics["failure_scores"]),
            },
            # Global in-progress metrics
            "global_in_progress_metrics": {
                "count": len(global_metrics["in_progress_steps"]),
                "avg_steps": safe_mean(global_metrics["in_progress_steps"]),
                "std_steps": safe_std(global_metrics["in_progress_steps"]),
                "min_steps": safe_min(global_metrics["in_progress_steps"]),
                "max_steps": safe_max(global_metrics["in_progress_steps"]),
                "avg_score": safe_mean(global_metrics["in_progress_scores"]),
                "std_score": safe_std(global_metrics["in_progress_scores"]),
                "min_score": safe_min(global_metrics["in_progress_scores"]),
                "max_score": safe_max(global_metrics["in_progress_scores"]),
            },
            # Global non-failed (success + in_progress) metrics
            "global_non_failed_metrics": {
                "count": len(global_metrics["non_failed_steps"]),
                "avg_steps": safe_mean(global_metrics["non_failed_steps"]),
                "std_steps": safe_std(global_metrics["non_failed_steps"]),
                "avg_score": safe_mean(global_metrics["non_failed_scores"]),
                "std_score": safe_std(global_metrics["non_failed_scores"]),
                "min_score": safe_min(global_metrics["non_failed_scores"]),
                "max_score": safe_max(global_metrics["non_failed_scores"]),
            },
            # Global overall metrics
            "global_overall_metrics": {
                "avg_steps": safe_mean(global_metrics["all_steps"]),
                "std_steps": safe_std(global_metrics["all_steps"]),
                "avg_score": safe_mean(global_metrics["all_scores"]),
                "std_score": safe_std(global_metrics["all_scores"]),
                "min_score": safe_min(global_metrics["all_scores"]),
                "max_score": safe_max(global_metrics["all_scores"]),
            },
            # Task-level results
            "task_results": task_results,
        }

        return global_summary

    def save_results(self, results: Dict):
        """Save evaluation results to output files."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Main evaluation results (JSON)
        results_file = self.output_dir / f"evaluation_results_{timestamp}.json"
        with results_file.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved evaluation results: {results_file}")

        # 2. Summary report (TXT)
        summary_file = self.output_dir / f"evaluation_summary_{timestamp}.txt"
        self.write_summary_report(results, summary_file)

        # 3. Also save a "latest" version (overwrites previous)
        latest_results_file = self.output_dir / "evaluation_results_latest.json"
        with latest_results_file.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved latest results: {latest_results_file}")

        latest_summary_file = self.output_dir / "evaluation_summary_latest.txt"
        self.write_summary_report(results, latest_summary_file)

    def write_summary_report(self, results: Dict, output_file: Path):
        """Write a human-readable summary report."""
        with output_file.open("w", encoding="utf-8") as f:
            f.write(f"TRAJECTORY EVALUATION SUMMARY\n")
            f.write(f"{'=' * 60}\n\n")

            f.write(f"Run Name: {results['run_name']}\n")
            f.write(f"Trajectory Logs: {results['trajectory_logs_dir']}\n")
            f.write(f"Is Combined Run: {results['is_combined_run']}\n")
            f.write(f"Evaluation Date: {results['evaluation_timestamp']}\n")
            f.write(f"Total Tasks: {results['total_tasks']}\n")
            f.write(f"Total Episodes: {results['total_episodes']}\n")
            if results.get("exclude_error_episodes"):
                f.write(
                    f"Excluded Episodes: {results.get('excluded_episodes_count', 0)}\n"
                )
            f.write(f"Global Average Score: {results['global_avg_score']}\n\n")

            # Write run metadata if available
            if results.get("run_metadata"):
                f.write(f"RUN METADATA\n")
                f.write(f"{'-' * 40}\n")
                metadata = results["run_metadata"]

                if metadata.get("model"):
                    f.write(f"Model: {metadata['model']}\n")
                if metadata.get("max_steps_mode"):
                    f.write(f"Max Steps Mode: {metadata['max_steps_mode']}\n")
                if metadata.get("retrieval_type"):
                    f.write(f"Retrieval Type: {metadata['retrieval_type']}\n")
                if metadata.get("frequency_strategy"):
                    f.write(f"Frequency Strategy: {metadata['frequency_strategy']}\n")
                if metadata.get("diversity_strategy"):
                    f.write(f"Diversity Strategy: {metadata['diversity_strategy']}\n")
                if metadata.get("evaluation_set_path"):
                    f.write(f"Evaluation Set: {metadata['evaluation_set_path']}\n")
                if metadata.get("execution_mode"):
                    f.write(f"Execution Mode: {metadata['execution_mode']}\n")
                if metadata.get("max_concurrent"):
                    f.write(f"Max Concurrent: {metadata['max_concurrent']}\n")
                if metadata.get("retrieval_config"):
                    f.write(f"Retrieval Config: {metadata['retrieval_config']}\n")
                f.write(f"\n")

            f.write(f"GLOBAL CLASSIFICATION STATISTICS\n")
            f.write(f"{'-' * 40}\n")
            f.write(
                f"Success Count: {results['global_success_count']} ({results['global_success_percentage']}%)\n"
            )
            f.write(
                f"Failure Count: {results['global_failure_count']} ({results['global_failure_percentage']}%)\n"
            )
            f.write(
                f"In-Progress Count: {results['global_in_progress_count']} ({results['global_in_progress_percentage']}%)\n\n"
            )

            f.write(f"GLOBAL SUCCESS METRICS\n")
            f.write(f"{'-' * 40}\n")
            success = results["global_success_metrics"]
            f.write(f"Count: {success['count']}\n")
            f.write(
                f"Average Steps: {success['avg_steps']} (±{success['std_steps']})\n"
            )
            f.write(f"Step Range: [{success['min_steps']}, {success['max_steps']}]\n")
            f.write(f"Average Score: {success['avg_score']}\n\n")

            f.write(f"GLOBAL IN-PROGRESS METRICS\n")
            f.write(f"{'-' * 40}\n")
            progress = results["global_in_progress_metrics"]
            f.write(f"Count: {progress['count']}\n")
            f.write(
                f"Average Steps: {progress['avg_steps']} (±{progress['std_steps']})\n"
            )
            f.write(f"Step Range: [{progress['min_steps']}, {progress['max_steps']}]\n")
            f.write(
                f"Average Score: {progress['avg_score']} (±{progress['std_score']})\n"
            )
            f.write(
                f"Score Range: [{progress['min_score']}, {progress['max_score']}]\n\n"
            )

            f.write(f"GLOBAL NON-FAILED METRICS (Success + In-Progress)\n")
            f.write(f"{'-' * 40}\n")
            non_failed = results["global_non_failed_metrics"]
            f.write(f"Count: {non_failed['count']}\n")
            f.write(
                f"Average Steps: {non_failed['avg_steps']} (±{non_failed['std_steps']})\n"
            )
            f.write(
                f"Average Score: {non_failed['avg_score']} (±{non_failed['std_score']})\n"
            )
            f.write(
                f"Score Range: [{non_failed['min_score']}, {non_failed['max_score']}]\n\n"
            )

            f.write(f"GLOBAL FAILURE METRICS\n")
            f.write(f"{'-' * 40}\n")
            failure = results["global_failure_metrics"]
            f.write(f"Count: {failure['count']}\n")
            f.write(
                f"Average Steps: {failure['avg_steps']} (±{failure['std_steps']})\n"
            )
            f.write(f"Step Range: [{failure['min_steps']}, {failure['max_steps']}]\n")
            f.write(f"Average Score: {failure['avg_score']}\n\n")

            f.write(f"GLOBAL OVERALL METRICS\n")
            f.write(f"{'-' * 40}\n")
            overall = results["global_overall_metrics"]
            f.write(
                f"Average Steps: {overall['avg_steps']} (±{overall['std_steps']})\n"
            )
            f.write(
                f"Average Score: {overall['avg_score']} (±{overall['std_score']})\n"
            )
            f.write(
                f"Score Range: [{overall['min_score']}, {overall['max_score']}]\n\n"
            )

            f.write(f"TASK-LEVEL BREAKDOWN\n")
            f.write(f"{'-' * 40}\n")

            for task_name, task_data in results["task_results"].items():
                f.write(f"\n{task_name.upper()}:\n")
                f.write(f"  Total Episodes: {task_data['total_episodes']}\n")
                if task_data.get("excluded_episodes_count", 0) > 0:
                    f.write(f"  Excluded: {task_data['excluded_episodes_count']}\n")
                f.write(
                    f"  Success: {task_data['success_count']} ({task_data['success_percentage']}%)\n"
                )
                f.write(
                    f"  Failure: {task_data['failure_count']} ({task_data['failure_percentage']}%)\n"
                )
                f.write(
                    f"  In-Progress: {task_data['in_progress_count']} ({task_data['in_progress_percentage']}%)\n"
                )
                f.write(
                    f"  Success - Avg Steps: {task_data['success_metrics']['avg_steps']}\n"
                )
                f.write(
                    f"  In-Progress - Avg Steps: {task_data['in_progress_metrics']['avg_steps']}\n"
                )
                f.write(
                    f"  In-Progress - Avg Score: {task_data['in_progress_metrics']['avg_score']}\n"
                )
                f.write(
                    f"  Non-Failed - Avg Steps: {task_data['non_failed_metrics']['avg_steps']}\n"
                )
                f.write(
                    f"  Non-Failed - Avg Score: {task_data['non_failed_metrics']['avg_score']}\n"
                )

        print(f"📄 Saved summary report: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate agent trajectories without gold comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use global config (ENVIRONMENT_NAME and RUN_ID_PREFIX)
  python evaluate_trajectories.py
  
  # Specify environment and run ID prefix
  python evaluate_trajectories.py --environment webarena --run-id run_eval_agentic_gpt_oss_20b
  
  # Evaluate specific task only
  python evaluate_trajectories.py --task boil-plant --verbose
  
  # Exclude error episodes from evaluation
  python evaluate_trajectories.py --exclude-errors
        """,
    )
    parser.add_argument(
        "--environment",
        type=str,
        default=ENVIRONMENT_NAME,
        help=f"Environment name (default: {ENVIRONMENT_NAME})",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=RUN_ID_PREFIX,
        help=f"Run ID prefix to search for (default: {RUN_ID_PREFIX})",
    )
    parser.add_argument(
        "--task", type=str, default=None, help="Evaluate only specific task (optional)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--exclude-errors",
        action="store_true",
        dest="exclude_errors",
        help="Exclude episodes that errored out or have empty steps from evaluation",
    )

    args = parser.parse_args()

    print("📊 Multi-Environment Trajectory Evaluator")
    print("=" * 80)

    try:
        # Create evaluator
        evaluator = TrajectoryEvaluator(
            environment_name=args.environment,
            run_id_prefix=args.run_id,
            specific_task=args.task,
            verbose=args.verbose,
            exclude_error_episodes=args.exclude_errors,
        )

        # Run evaluation
        results = evaluator.run_evaluation()

        # Save results
        evaluator.save_results(results)

        # Print summary
        print(f"\n{'=' * 80}")
        print(f"🎉 Evaluation completed successfully!")
        print(f"{'=' * 80}")
        print(f"   Tasks evaluated: {results['total_tasks']}")
        print(f"   Episodes evaluated: {results['total_episodes']}")
        if results.get("excluded_episodes_count", 0) > 0:
            print(f"   Episodes excluded: {results['excluded_episodes_count']}")
        print(f"   Global Average Score: {results['global_avg_score']}")
        print(f"   Success rate: {results['global_success_percentage']}%")
        print(f"   Failure rate: {results['global_failure_percentage']}%")
        print(f"   In-Progress rate: {results['global_in_progress_percentage']}%")
        print(
            f"   Avg steps (success): {results['global_success_metrics']['avg_steps']}"
        )
        print(
            f"   Avg steps (in-progress): {results['global_in_progress_metrics']['avg_steps']}"
        )
        print(
            f"   Avg score (success): {results['global_success_metrics']['avg_score']}"
        )
        print(
            f"   Avg score (in-progress): {results['global_in_progress_metrics']['avg_score']}"
        )
        print(f"   Results saved to: {evaluator.output_dir}")
        print(f"{'=' * 80}")

        return 0

    except Exception as e:
        print(f"\n💥 Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
