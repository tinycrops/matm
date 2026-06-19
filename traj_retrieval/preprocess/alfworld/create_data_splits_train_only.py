import os
#!/usr/bin/env python3
"""
create_data_splits_train_only.py

Create train/validation splits for AlfWorld trajectories based on task types and
difficulty, and save all outputs under preprocess/new_splits/alfworld.

Usage:
    python traj_retrieval/preprocess/alfworld/create_data_splits_train_only.py
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

# ============================================================================
# GLOBAL CONFIGURATION - Edit these paths as needed
# ============================================================================

# Path to gold trajectories (output from create_gold_trajectory.py)
GOLD_TRAJECTORIES_DIR = (
    os.path.join(os.environ.get("MATM_DATA_ROOT", "environments"), "alfworld/traj_logs/run_gold")
)

# Output directory for new (train-only workflow) data splits
SPLITS_OUTPUT_DIR = "traj_retrieval/preprocess/new_splits/alfworld"

# Path to AlfWorld validation data (contains valid_seen, valid_train, valid_unseen)
ALFWORLD_DATA_PATH = (
    os.path.join(os.environ.get("MATM_DATA_ROOT", "environments"), "alfworld/data/json_2.1.1")
)

# Maximum number of instances per validation split
MAX_VALIDATION_SIZE = 100

# Difficulty thresholds based on step count
DIFFICULTY_THRESHOLDS = {
    "easy_max": 10,  # Easy: <= 10 steps
    "medium_max": 17,  # Medium: > 10 and <= 17 steps, Hard: > 17 steps
}

# Random seed for reproducibility
RANDOM_SEED = 42

# Train trajectory manifest used by train-only LanceDB index creation
INDEX_SOURCE_FILENAME = "index_source_train_all.json"

# ============================================================================


class AlfWorldDataSplitCreator:
    """
    Creates train/validation splits for AlfWorld based on:
    1. Validation split: seen.json, train.json, unseen.json (from original validation data)
    2. Train split: test/validation splits by difficulty (easy/medium/hard) based on step count
    """

    def __init__(
        self, gold_dir: str = GOLD_TRAJECTORIES_DIR, splits_dir: str = SPLITS_OUTPUT_DIR
    ):
        self.gold_dir = Path(gold_dir)
        self.splits_dir = Path(splits_dir)
        self.splits_dir.mkdir(parents=True, exist_ok=True)

        self.task_data = {}
        self.task_types = []

    @staticmethod
    def _create_variation_id(game_file: str, goal: str = "") -> str:
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
            goal: Natural language goal description (not used, kept for compatibility)

        Returns:
            variation_id string in format: variation_id_{hash}
        """
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

    def load_all_trajectories(self):
        """Load all gold trajectories"""

        print("📖 Loading gold trajectories...")

        task_dirs = [
            d
            for d in self.gold_dir.iterdir()
            if d.is_dir() and d.name.startswith("run_")
        ]

        for task_dir in sorted(task_dirs):
            task_name = task_dir.name.replace("run_", "")

            # Load all variations for this task
            variations = []
            for var_file in sorted(task_dir.glob("variation_*.json")):
                with open(var_file, "r") as f:
                    traj_data = json.load(f)
                    variations.append(
                        {
                            "file": str(var_file),
                            "variation_id": var_file.stem,
                            "task_id": traj_data["task_id"],
                            "task_type": traj_data["task_type"],
                            "split": traj_data["split"],
                            "num_steps": traj_data["num_steps"],
                            "goal": traj_data["goal"],
                            "floor_plan": traj_data["floor_plan"],
                        }
                    )

            if variations:
                self.task_data[task_name] = variations
                self.task_types.append(task_name)

        print(f"✅ Loaded {len(self.task_types)} task types")
        for task_name in sorted(self.task_types):
            print(f"   • {task_name:45s}: {len(self.task_data[task_name])} variations")

    def create_official_test_splits(self):
        """
        Create official test splits from original validation data directories:
        - seen.json (from valid_seen) - 140 tasks in familiar environments
        - unseen.json (from valid_unseen) - 134 tasks in novel environments
        - train.json (from valid_train) - Additional validation tasks

        These are game file references (NO expert trajectories).
        This is the official ALFWorld test set used in papers.
        """

        print("\n📊 Creating official test splits (from original validation data)...")

        # Create official_test subdirectory
        official_test_dir = self.splits_dir / "official_test"
        official_test_dir.mkdir(exist_ok=True)

        # Map validation directory names to split names
        val_data_path = Path(ALFWORLD_DATA_PATH)
        split_mapping = {
            "valid_seen": "seen",
            "valid_train": "train",
            "valid_unseen": "unseen",
        }

        split_stats = {}  # Store statistics for README

        for val_dir_name, split_name in split_mapping.items():
            val_dir = val_data_path / val_dir_name

            if not val_dir.exists():
                print(f"⚠️  Directory not found: {val_dir}")
                continue

            # Find all game files in this validation split
            game_files = list(val_dir.rglob("game.tw-pddl"))

            split_data = []
            task_types = {}
            floor_plans = {}

            for game_path in game_files:
                # Load traj_data.json to get metadata
                traj_file = game_path.parent / "traj_data.json"
                if traj_file.exists():
                    with open(traj_file, "r") as f:
                        traj_data = json.load(f)

                    task_type = traj_data.get("task_type", "unknown")
                    floor_plan = traj_data.get("scene", {}).get("floor_plan", "unknown")
                    game_file_str = str(game_path)

                    # Extract goal before creating variation_id
                    goal = (
                        traj_data.get("turk_annotations", {})
                        .get("anns", [{}])[0]
                        .get("task_desc", "Unknown goal")
                        if traj_data.get("turk_annotations", {}).get("anns")
                        else "Unknown goal"
                    )

                    # Create variation_id from game_file path and goal
                    variation_id = self._create_variation_id(game_file_str, goal)

                    # Extract num_steps from expert demonstration or use default
                    num_steps = 30  # Default value
                    try:
                        # Try to get from plan.high_pddl (expert actions)
                        if "plan" in traj_data and "high_pddl" in traj_data["plan"]:
                            high_pddl = traj_data["plan"]["high_pddl"]
                            if isinstance(high_pddl, list):
                                num_steps = len(high_pddl)
                        # Alternative: check for expert_actions or low_actions
                        elif "plan" in traj_data and "low_actions" in traj_data["plan"]:
                            low_actions = traj_data["plan"]["low_actions"]
                            if isinstance(low_actions, list):
                                num_steps = len(low_actions)
                        # Alternative: check num_steps field directly
                        elif "num_steps" in traj_data:
                            num_steps = traj_data["num_steps"]
                    except Exception as e:
                        # If any error, use default value
                        pass

                    split_data.append(
                        {
                            "task_type": task_type,
                            "task_id": traj_data.get("task_id", "unknown"),
                            "variation_id": variation_id,
                            "game_file": game_file_str,
                            "floor_plan": floor_plan,
                            "goal": goal,
                            "num_steps": num_steps,
                        }
                    )

                    # Track statistics
                    task_types[task_type] = task_types.get(task_type, 0) + 1
                    floor_plans[floor_plan] = floor_plans.get(floor_plan, 0) + 1

            # Save split
            output_file = official_test_dir / f"{split_name}.json"
            with open(output_file, "w") as f:
                json.dump(split_data, f, indent=2)
            print(f"✅ Saved official_test/{split_name}.json: {len(split_data)} games")

            # Store statistics
            split_stats[split_name] = {
                "total_tasks": len(split_data),
                "task_types": task_types,
                "floor_plans": floor_plans,
                "unique_task_types": len(task_types),
                "unique_floor_plans": len(floor_plans),
            }

        # Create README with statistics
        self._create_official_test_readme(official_test_dir, split_stats)

        # Create combined seen + unseen split
        self._create_combined_seen_unseen(official_test_dir)

    def _create_official_test_readme(self, output_dir, split_stats):
        """
        Create README.md with comprehensive statistics for official test splits.

        Args:
            output_dir: Path to official_test directory
            split_stats: Dict containing statistics for each split
        """
        readme_path = output_dir / "README.md"

        with readme_path.open("w", encoding="utf-8") as f:
            f.write("# Official ALFWorld Test Sets\n\n")
            f.write(
                "This directory contains the **official ALFWorld test sets** used for evaluation in publications.\n\n"
            )
            f.write(
                "These test sets are derived from the original ALFWorld validation data and are used to assess agent performance on embodied AI tasks.\n\n"
            )
            f.write("---\n\n")

            # Overview
            f.write("## 📊 Overview\n\n")
            f.write("| Split | Tasks | Description |\n")
            f.write("|-------|-------|-------------|\n")

            for split_name in ["seen", "unseen", "train"]:
                if split_name in split_stats:
                    stats = split_stats[split_name]
                    if split_name == "seen":
                        desc = (
                            "In-distribution: Familiar room layouts (FloorPlans 1-30)"
                        )
                    elif split_name == "unseen":
                        desc = "Out-of-distribution: Novel room layouts"
                    else:
                        desc = "Additional validation tasks"
                    f.write(
                        f"| **{split_name.capitalize()}** | {stats['total_tasks']} | {desc} |\n"
                    )

            f.write("\n---\n\n")

            # Detailed statistics for each split
            for split_name in ["seen", "unseen", "train"]:
                if split_name not in split_stats:
                    continue

                stats = split_stats[split_name]
                f.write(f"## 📋 {split_name.capitalize()} Split Details\n\n")
                f.write(f"**Total Tasks**: {stats['total_tasks']}\n\n")

                # Task type breakdown
                f.write("### Task Type Distribution\n\n")
                f.write("| Task Type | Count | Percentage |\n")
                f.write("|-----------|-------|------------|\n")

                sorted_task_types = sorted(
                    stats["task_types"].items(), key=lambda x: x[1], reverse=True
                )
                for task_type, count in sorted_task_types:
                    percentage = (count / stats["total_tasks"]) * 100
                    # Format task type name
                    display_name = task_type.replace("_", " ").title()
                    f.write(f"| {display_name} | {count} | {percentage:.1f}% |\n")

                f.write("\n")

                # Floor plan breakdown
                f.write("### Floor Plan Distribution\n\n")
                f.write(f"**Unique Floor Plans**: {stats['unique_floor_plans']}\n\n")

                sorted_floor_plans = sorted(
                    stats["floor_plans"].items(),
                    key=lambda x: int(x[0].replace("FloorPlan", ""))
                    if "FloorPlan" in x[0]
                    else 999,
                )

                # Group floor plans in a compact table
                f.write(
                    "| Floor Plan | Tasks | Floor Plan | Tasks | Floor Plan | Tasks |\n"
                )
                f.write(
                    "|------------|-------|------------|-------|------------|-------|\n"
                )

                # Print in groups of 3
                for i in range(0, len(sorted_floor_plans), 3):
                    row_items = sorted_floor_plans[i : i + 3]
                    row = []
                    for fp, count in row_items:
                        row.extend([fp, str(count)])
                    # Pad if less than 3 items
                    while len(row) < 6:
                        row.extend(["", ""])
                    f.write(f"| {' | '.join(row)} |\n")

                f.write("\n---\n\n")

            # Usage section
            f.write("## 🚀 Usage\n\n")
            f.write("### For Publication Results\n\n")
            f.write("Report **both** Seen and Unseen results separately:\n\n")
            f.write("```bash\n")
            f.write("# Evaluate on SEEN test set (in-distribution)\n")
            f.write("python -m traj_retrieval.run_evaluation \\\n")
            f.write("  --config traj_retrieval/run_evaluation_config.yaml \\\n")
            f.write(
                "  --evaluation-set traj_retrieval/preprocess/new_splits/alfworld/official_test/seen.json\n\n"
            )
            f.write("# Evaluate on UNSEEN test set (out-of-distribution)\n")
            f.write("python -m traj_retrieval.run_evaluation \\\n")
            f.write("  --config traj_retrieval/run_evaluation_config.yaml \\\n")
            f.write(
                "  --evaluation-set traj_retrieval/preprocess/new_splits/alfworld/official_test/unseen.json\n"
            )
            f.write("```\n\n")

            # Data format
            f.write("## 📝 Data Format\n\n")
            f.write("Each task in the JSON files is a dictionary with:\n\n")
            f.write("```json\n")
            f.write("{\n")
            f.write('  "task_type": "pick_and_place_simple",\n')
            f.write('  "task_id": "trial_T20190908_123456_789012",\n')
            f.write(
                '  "variation_id": "variation_id_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",\n'
            )
            f.write('  "game_file": "/path/to/game.tw-pddl",\n')
            f.write('  "floor_plan": "FloorPlan1",\n')
            f.write('  "goal": "Put a clean mug on the coffee table."\n')
            f.write("}\n")
            f.write("```\n\n")
            f.write("### Key Fields\n\n")
            f.write(
                "- **task_type**: Type of task (e.g., pick_and_place_simple, pick_heat_then_place_in_recep)\n"
            )
            f.write("- **task_id**: Original ALFWorld task ID from traj_data.json\n")
            f.write(
                "- **variation_id**: Unique identifier in format `variation_id_{hash}` where hash is MD5 of the game_file path from json_2.1.1 onwards\n"
            )
            f.write("- **game_file**: Absolute path to the game.tw-pddl file\n")
            f.write("- **floor_plan**: Floor plan name (e.g., FloorPlan1)\n")
            f.write("- **goal**: Natural language task description\n\n")

            # Evaluation protocol
            f.write("## 📊 Evaluation Protocol\n\n")
            f.write("### Standard Reporting Format\n\n")
            f.write("| Method | Training | Seen (%) | Unseen (%) |\n")
            f.write("|--------|----------|----------|------------|\n")
            f.write("| BUTLER | ALFWorld (3.5K) | 38.4 | 23.9 |\n")
            f.write("| ReAct  | Zero-shot | 32.0 | 18.5 |\n")
            f.write("| **Your Method** | - | **X.X** | **Y.Y** |\n\n")

            f.write("### Key Points\n\n")
            f.write(
                "- **Seen**: Tests generalization within familiar environments (easier)\n"
            )
            f.write(
                "- **Unseen**: Tests generalization to novel environments (harder)\n"
            )
            f.write("- **Both must be reported** for fair comparison with prior work\n")
            f.write("- Even zero-shot methods should report both splits\n\n")

            # Source information
            f.write("## 🔍 Source\n\n")
            f.write(
                "These test sets are extracted from the official ALFWorld dataset:\n\n"
            )
            f.write("- **Seen**: `$ALFWORLD_DATA/json_2.1.1/valid_seen` (140 tasks)\n")
            f.write(
                "- **Unseen**: `$ALFWORLD_DATA/json_2.1.1/valid_unseen` (134 tasks)\n"
            )
            f.write(
                "- **Train**: `$ALFWORLD_DATA/json_2.1.1/valid_train` (additional validation)\n\n"
            )
            f.write("These are game file references without expert trajectories.\n\n")

            # Citation
            f.write("## 📄 Citation\n\n")
            f.write("If you use these test sets, please cite the ALFWorld paper:\n\n")
            f.write("```bibtex\n")
            f.write("@inproceedings{alfworld2021,\n")
            f.write(
                "  title={ALFWorld: Aligning Text and Embodied Environments for Interactive Learning},\n"
            )
            f.write(
                "  author={Shridhar, Mohit and Yuan, Xingdi and C\\^ot\\'e, Marc-Alexandre and Bisk, Yonatan and Trischler, Adam and Hausknecht, Matthew},\n"
            )
            f.write(
                "  booktitle={International Conference on Learning Representations (ICLR)},\n"
            )
            f.write("  year={2021}\n")
            f.write("}\n")
            f.write("```\n\n")

            # Additional notes
            f.write("## 📌 Notes\n\n")
            f.write(
                "- **Seen vs Unseen**: The split refers to environment complexity, not your model's training\n"
            )
            f.write(
                "- **Zero-shot methods**: Should still report both splits (shows robustness)\n"
            )
            f.write(
                "- **Trained methods**: Both splits show in-domain vs out-of-domain generalization\n"
            )
            f.write(
                "- **Expected pattern**: Unseen scores are typically lower than Seen scores\n"
            )

        print(f"✅ Created README.md with detailed statistics")

    def _create_combined_seen_unseen(self, output_dir):
        """
        Create a combined seen + unseen split file.

        Args:
            output_dir: Path to official_test directory
        """
        seen_file = output_dir / "seen.json"
        unseen_file = output_dir / "unseen.json"
        combined_file = output_dir / "complete_seen_unseen.json"

        combined_data = []

        # Load seen split
        if seen_file.exists():
            with open(seen_file, "r") as f:
                seen_data = json.load(f)
                combined_data.extend(seen_data)

        # Load unseen split
        if unseen_file.exists():
            with open(unseen_file, "r") as f:
                unseen_data = json.load(f)
                combined_data.extend(unseen_data)

        # Save combined split
        with open(combined_file, "w") as f:
            json.dump(combined_data, f, indent=2)

        print(
            f"✅ Created complete_seen_unseen.json: {len(combined_data)} total tasks (seen + unseen)"
        )

    def create_train_splits(self):
        """
        Create train splits by difficulty (based on step count):
        - Easy: <= {easy_max} steps
        - Medium: > {easy_max} and <= {medium_max} steps
        - Hard: > {medium_max} steps

        For each difficulty:
        - validation_{{difficulty}}.json: max {max_val} instances
        - test_{{difficulty}}.json: remaining instances

        All reference expert trajectories in run_gold.
        """.format(
            easy_max=DIFFICULTY_THRESHOLDS["easy_max"],
            medium_max=DIFFICULTY_THRESHOLDS["medium_max"],
            max_val=MAX_VALIDATION_SIZE,
        )

        print("\n📊 Creating train splits by difficulty...")

        # Create train subdirectory
        train_dir = self.splits_dir / "train"
        train_dir.mkdir(parents=True, exist_ok=True)

        # Collect all training trajectories
        train_trajs = []
        for task_name, variations in self.task_data.items():
            for var in variations:
                if var["split"] == "train":
                    # Load the trajectory file to get the game_file
                    traj_file = Path(var["file"])
                    with open(traj_file, "r") as f:
                        traj_data = json.load(f)

                    # Get game_file and ensure it's absolute path
                    game_file = traj_data.get("game_file", "")
                    if game_file and not game_file.startswith("/"):
                        # Relative path - need to make it absolute
                        # The game_file might be like "environments/alfworld/data/json_2.1.1/train/..."
                        # We need to strip the "environments/alfworld/data/json_2.1.1" part and use ALFWORLD_DATA_PATH
                        relative_prefix = "environments/alfworld/data/json_2.1.1/"
                        if game_file.startswith(relative_prefix):
                            # Strip the prefix and prepend ALFWORLD_DATA_PATH
                            game_file_rel = game_file[len(relative_prefix) :]
                            game_file = str(Path(ALFWORLD_DATA_PATH) / game_file_rel)
                        else:
                            # Just prepend ALFWORLD_DATA_PATH directly
                            game_file = str(Path(ALFWORLD_DATA_PATH) / game_file)

                    # Create variation_id from game_file path and goal
                    variation_id = self._create_variation_id(game_file, var["goal"])

                    train_trajs.append(
                        {
                            "task_type": var["task_type"],
                            "variation_id": variation_id,
                            "task_id": var["task_id"],
                            "trajectory_file": var[
                                "file"
                            ],  # Points to run_gold trajectory
                            "game_file": game_file,  # Points to original game file (absolute path)
                            "num_steps": var["num_steps"],
                            "goal": var["goal"],
                            "floor_plan": var["floor_plan"],
                        }
                    )

        print(f"   Total training trajectories: {len(train_trajs)}")

        # Save full train-only index source manifest for train-only LanceDB builds.
        # This intentionally contains all train trajectories (not split by difficulty).
        manifest_path = train_dir / INDEX_SOURCE_FILENAME
        manifest_data = sorted(
            train_trajs,
            key=lambda x: (x["task_type"], x["variation_id"], x["task_id"]),
        )
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=2)
        print(f"   ✅ train/{INDEX_SOURCE_FILENAME:20s}: {len(manifest_data):4d} trajs")

        # Split by difficulty based on step count
        easy_max = DIFFICULTY_THRESHOLDS["easy_max"]
        medium_max = DIFFICULTY_THRESHOLDS["medium_max"]

        easy = [t for t in train_trajs if t["num_steps"] <= easy_max]
        medium = [t for t in train_trajs if easy_max < t["num_steps"] <= medium_max]
        hard = [t for t in train_trajs if t["num_steps"] > medium_max]

        print(f"   • Easy (≤{easy_max} steps): {len(easy)} trajectories")
        print(
            f"   • Medium ({easy_max+1}-{medium_max} steps): {len(medium)} trajectories"
        )
        print(f"   • Hard (>{medium_max} steps): {len(hard)} trajectories")

        # For each difficulty, split into validation (max MAX_VALIDATION_SIZE) and test (rest)
        def split_validation_test(data, max_validation=MAX_VALIDATION_SIZE):
            """Split data into validation (max N) and test (rest)"""
            # Shuffle for random split
            import random

            random.seed(RANDOM_SEED)
            data_copy = data.copy()
            random.shuffle(data_copy)

            val_size = min(len(data_copy), max_validation)
            validation = data_copy[:val_size]
            test = data_copy[val_size:]

            return validation, test

        easy_val, easy_test = split_validation_test(easy)
        medium_val, medium_test = split_validation_test(medium)
        hard_val, hard_test = split_validation_test(hard)

        # Save splits
        splits_to_save = {
            "validation_easy": easy_val,
            "validation_medium": medium_val,
            "validation_hard": hard_val,
            "test_easy": easy_test,
            "test_medium": medium_test,
            "test_hard": hard_test,
        }

        print("\n   Saving splits to train/ directory:")
        for split_name, split_data in splits_to_save.items():
            output_file = train_dir / f"{split_name}.json"
            with open(output_file, "w") as f:
                json.dump(split_data, f, indent=2)

            if split_data:
                min_steps = min(t["num_steps"] for t in split_data)
                max_steps = max(t["num_steps"] for t in split_data)
                avg_steps = sum(t["num_steps"] for t in split_data) / len(split_data)
                print(
                    f"   ✅ train/{split_name:20s}: {len(split_data):4d} trajs, "
                    f"steps: {min_steps:2d}-{max_steps:3d} (avg: {avg_steps:.1f})"
                )
            else:
                print(f"   ✅ train/{split_name:20s}: {len(split_data):4d} trajs")

    def create_all_splits(self):
        """Create all splits"""

        print("🏆 ALFWorld Data Split Creator")
        print("=" * 60)
        print(f"Gold trajectories: {self.gold_dir}")
        print(f"Output directory: {self.splits_dir}")
        print("=" * 60)

        # Load trajectories
        self.load_all_trajectories()

        # Create split types
        print("\n" + "=" * 60)
        print("SPLIT TYPE 1: OFFICIAL TEST SPLITS")
        print("=" * 60)
        self.create_official_test_splits()

        print("\n" + "=" * 60)
        print("SPLIT TYPE 2: TRAIN SPLITS (BY DIFFICULTY)")
        print("=" * 60)
        self.create_train_splits()

        print("\n" + "=" * 60)
        print("✅ ALL SPLITS CREATED")
        print("=" * 60)
        print(f"\nSplits saved to: {self.splits_dir}")
        print("\nDirectory structure:")
        print("  📁 train/")
        print(
            f"     • {INDEX_SOURCE_FILENAME} (all train trajectories for index source)"
        )
        print(
            f"     • validation_easy.json, validation_medium.json, validation_hard.json (max {MAX_VALIDATION_SIZE} each)"
        )
        print("     • test_easy.json, test_medium.json, test_hard.json (remaining)")
        print("     → All reference expert trajectories in run_gold/")
        print("\n  📁 official_test/")
        print("     • seen.json (140 tasks - in-distribution)")
        print("     • unseen.json (134 tasks - out-of-distribution)")
        print("     • train.json (additional validation tasks)")
        print("     • complete_seen_unseen.json (combined seen + unseen)")
        print("     • README.md (comprehensive statistics)")
        print("     → Game file references (NO expert trajectories)")
        print("     → Use for publication results")
        easy_max = DIFFICULTY_THRESHOLDS["easy_max"]
        medium_max = DIFFICULTY_THRESHOLDS["medium_max"]
        print(
            f"\nDifficulty criteria: Easy (≤{easy_max} steps), Medium ({easy_max+1}-{medium_max} steps), Hard (>{medium_max} steps)"
        )
        print("=" * 60)


def main():
    """Main entry point"""
    creator = AlfWorldDataSplitCreator(
        gold_dir=GOLD_TRAJECTORIES_DIR, splits_dir=SPLITS_OUTPUT_DIR
    )

    creator.create_all_splits()

    print("\n✅ ALFWorld data split creation complete!")
    print("\n📋 Summary:")
    print(
        "  • train/ folder: Contains splits with expert trajectories (from run_gold/)"
    )
    print("    - Use validation_{easy,medium,hard}.json for hyperparameter tuning")
    print("    - Use test_{easy,medium,hard}.json for final evaluation")
    print(
        "\n  • official_test/ folder: Official test sets for publication (NO expert trajectories)"
    )
    print("    - seen.json: 140 tasks (in-distribution)")
    print("    - unseen.json: 134 tasks (out-of-distribution)")
    print("    - train.json: Additional validation tasks")
    print("    - complete_seen_unseen.json: Combined seen + unseen")
    print("    - README.md: Comprehensive statistics and usage guide")
    print("    - ⭐ Use BOTH seen and unseen for publication results!")


if __name__ == "__main__":
    main()
