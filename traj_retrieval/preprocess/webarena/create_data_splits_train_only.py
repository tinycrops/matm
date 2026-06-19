import os
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np

# ============================================================================
# GLOBAL CONFIGURATION - Edit these paths as needed
# ============================================================================
# Path to WebArena config files directory
WEBARENA_CONFIG_DIR = (
    os.path.join(os.environ.get("MATM_DATA_ROOT", "environments"), "webarena/webarena/config_files")
)
# Output directory for train-only workflow splits
SPLITS_OUTPUT_DIR = "traj_retrieval/preprocess/new_splits/webarena"
# Train-only index source manifest filename
TRAIN_INDEX_SOURCE_FILENAME = "index_source_train_augmented.json"
# Default max_steps for WebArena tasks
DEFAULT_MAX_STEPS = 30
# Sites to exclude from test split (only keep in validation)
EXCLUDE_FROM_TEST_SITES = []

# ============================================================================


def scan_webarena_configs(
    config_dir: str,
) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
    """
    Scan WebArena config files and group by intent_template_id.
    Separates configs into two groups:
    - configs_for_test_and_val: can be used for both test and validation
    - configs_for_val_only: only for validation (contains excluded sites)

    Args:
        config_dir: Path to directory containing WebArena config JSON files

    Returns:
        Tuple of (configs_for_test_and_val, configs_for_val_only)
        Each is a Dict mapping task_name to list of config info
    """
    config_path = Path(config_dir)
    if not config_path.exists():
        print(f"❌ Config directory not found: {config_dir}")
        return {}, {}

    print("🔍 Scanning WebArena config files...")
    print(f"   Excluding from test split: sites containing {EXCLUDE_FROM_TEST_SITES}")
    print("-" * 80)

    configs_for_test_and_val = defaultdict(list)
    configs_for_val_only = defaultdict(list)
    total_configs = 0
    skipped_configs = 0
    excluded_from_test = 0

    # Scan all JSON files in config directory
    for config_file in sorted(config_path.glob("*.json")):
        try:
            with config_file.open("r", encoding="utf-8") as f:
                config_data = json.load(f)

            # Some auxiliary files (e.g., test.json) are lists, not config objects.
            if not isinstance(config_data, dict):
                skipped_configs += 1
                print(f"⚠️  Skipping {config_file.name}: JSON root is not an object")
                continue

            # Extract required fields
            task_id = config_data.get("task_id")
            intent_template_id = config_data.get("intent_template_id")
            intent = config_data.get("intent", "")
            sites = config_data.get("sites", [])

            if task_id is None or intent_template_id is None:
                print(
                    f"⚠️  Skipping {config_file.name}: missing task_id or intent_template_id"
                )
                skipped_configs += 1
                continue

            # Task name format: intent_template_id_{intent_template_id}
            task_name = f"intent_template_id_{intent_template_id}"

            config_info = {
                "config_file": str(config_file.absolute()),
                "task_id": task_id,
                "intent_template_id": intent_template_id,
                "intent": intent,
                "task_name": task_name,
                "sites": sites,
            }

            # Check if any site contains excluded keywords
            should_exclude_from_test = any(
                any(excluded_site in site for excluded_site in EXCLUDE_FROM_TEST_SITES)
                for site in sites
            )

            if should_exclude_from_test:
                configs_for_val_only[task_name].append(config_info)
                excluded_from_test += 1
            else:
                configs_for_test_and_val[task_name].append(config_info)

            total_configs += 1

        except Exception as e:
            print(f"❌ Error reading {config_file.name}: {str(e)[:50]}")
            skipped_configs += 1

    print(f"\n✅ Successfully scanned {total_configs} config files")
    if skipped_configs > 0:
        print(f"⚠️  Skipped {skipped_configs} config files due to errors")

    print(f"\n📊 Split filtering results:")
    print(
        f"   Configs for test+val: {sum(len(v) for v in configs_for_test_and_val.values())}"
    )
    print(f"   Configs for val only: {excluded_from_test} (excluded from test)")

    # Print distribution by intent template for test+val configs
    print(f"\n📊 Distribution by intent_template_id (test+val configs):")
    for task_name in sorted(configs_for_test_and_val.keys()):
        configs = configs_for_test_and_val[task_name]
        print(f"   {task_name:<20} -> {len(configs):>4} configs")

    print(
        f"\n   Total unique intent templates (test+val): {len(configs_for_test_and_val)}"
    )

    # Print distribution for val-only configs
    if configs_for_val_only:
        print(f"\n📊 Distribution by intent_template_id (val-only configs):")
        for task_name in sorted(configs_for_val_only.keys()):
            configs = configs_for_val_only[task_name]
            sites_list = set()
            for c in configs:
                sites_list.update(c["sites"])
            print(
                f"   {task_name:<20} -> {len(configs):>4} configs (sites: {', '.join(sorted(sites_list))})"
            )

        print(
            f"\n   Total unique intent templates (val-only): {len(configs_for_val_only)}"
        )

    return dict(configs_for_test_and_val), dict(configs_for_val_only)


def _create_random_splits(
    tasks_by_intent: Dict[str, List[Dict]], test_size: int, max_steps: int
) -> Tuple[List[Dict], List[Dict]]:
    """
    Create random test/validation splits when stratification is not possible.

    Args:
        tasks_by_intent: Dict mapping task_name to list of config info
        test_size: Number of samples for test set
        max_steps: Max steps to assign to each episode

    Returns:
        Tuple of (test_episodes, validation_episodes)
    """
    # Flatten all configs into a single list
    all_configs = []
    for configs in tasks_by_intent.values():
        all_configs.extend(configs)

    # Shuffle and split
    random.shuffle(all_configs)
    test_configs = all_configs[:test_size]
    val_configs = all_configs[test_size:]

    # Convert to episode format
    test_episodes = []
    validation_episodes = []

    for config_info in test_configs:
        episode = {
            "task_name": config_info["task_name"],
            "variation_id": f"task_id_{config_info['task_id']}",
            "task_id": config_info["task_id"],
            "intent_template_id": config_info["intent_template_id"],
            "config_file": config_info["config_file"],
            "max_steps": max_steps,
            "intent": config_info["intent"],
            "sites": config_info["sites"],
        }
        test_episodes.append(episode)

    for config_info in val_configs:
        episode = {
            "task_name": config_info["task_name"],
            "variation_id": f"task_id_{config_info['task_id']}",
            "task_id": config_info["task_id"],
            "intent_template_id": config_info["intent_template_id"],
            "config_file": config_info["config_file"],
            "max_steps": max_steps,
            "intent": config_info["intent"],
            "sites": config_info["sites"],
        }
        validation_episodes.append(episode)

    print(f"\n📊 Random split summary:")
    print(f"   Test:       {len(test_episodes)} episodes")
    print(f"   Validation: {len(validation_episodes)} episodes")
    print(
        f"   Test unique intents: {len(set(ep['task_name'] for ep in test_episodes))}"
    )
    print(
        f"   Val unique intents: {len(set(ep['task_name'] for ep in validation_episodes))}"
    )

    return test_episodes, validation_episodes


def create_stratified_splits(
    configs_for_test_and_val: Dict[str, List[Dict]],
    configs_for_val_only: Dict[str, List[Dict]],
    test_size: int = 100,
    seed: int = 42,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Create stratified test and validation splits.

    Test split comes only from configs_for_test_and_val.
    Validation split includes configs from both groups.

    Args:
        configs_for_test_and_val: Dict mapping task_name to configs that can be in test or val
        configs_for_val_only: Dict mapping task_name to configs that can only be in val
        test_size: Target number of samples for test set
        seed: Random seed for reproducibility
        max_steps: Max steps to assign to each episode

    Returns:
        Tuple of (test_episodes, validation_episodes)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Calculate total configs
    total_test_val_configs = sum(
        len(configs) for configs in configs_for_test_and_val.values()
    )
    total_val_only_configs = sum(
        len(configs) for configs in configs_for_val_only.values()
    )
    num_intents = len(configs_for_test_and_val)

    print(f"\n🎯 Creating stratified splits...")
    print(f"   Target test size: {test_size} samples")
    print(f"   Configs for test+val: {total_test_val_configs}")
    print(f"   Configs for val only: {total_val_only_configs}")
    print(f"   Unique intents (test+val): {num_intents}")
    print(f"   Max steps per episode: {max_steps}")
    print(f"   Random seed: {seed}")
    print("-" * 80)

    # Check if stratified sampling is possible
    if num_intents > test_size:
        print(f"\n⚠️  Unique intents ({num_intents}) > test size ({test_size})")
        print(f"   Stratified sampling not possible - using random selection instead")
        test_episodes, val_episodes_from_test_val = _create_random_splits(
            configs_for_test_and_val, test_size, max_steps
        )
    else:
        if total_test_val_configs < test_size:
            print(
                f"⚠️  Warning: Total test+val configs ({total_test_val_configs}) < test size ({test_size})"
            )
            print(
                f"   Using all configs for test, validation will only have val-only configs"
            )
            test_size = total_test_val_configs

        # Calculate samples per intent for test set (proportional stratified sampling)
        test_samples_per_intent = {}

        # Sort by intent name for deterministic ordering
        sorted_intents = sorted(configs_for_test_and_val.keys())

        # First pass: allocate proportionally
        for task_name in sorted_intents:
            configs = configs_for_test_and_val[task_name]
            n_configs = len(configs)

            # Proportional allocation
            proportion = n_configs / total_test_val_configs
            allocated = max(1, int(proportion * test_size))  # At least 1 per intent

            # Don't allocate more than available
            allocated = min(allocated, n_configs)

            test_samples_per_intent[task_name] = allocated

        # Adjust to match exact test_size
        current_test_total = sum(test_samples_per_intent.values())

        # If we're over, reduce from intents with most samples
        while current_test_total > test_size:
            # Find intent with most test samples that can be reduced
            max_intent = max(
                [k for k, v in test_samples_per_intent.items() if v > 1],
                key=lambda k: test_samples_per_intent[k],
                default=None,
            )
            if max_intent:
                test_samples_per_intent[max_intent] -= 1
                current_test_total -= 1
            else:
                break

        # If we're under, add to intents with available samples
        while current_test_total < test_size:
            # Find intent with most available samples
            candidates = [
                k
                for k in sorted_intents
                if test_samples_per_intent[k] < len(configs_for_test_and_val[k])
            ]
            if not candidates:
                break

            # Add to intent with most remaining samples
            max_intent = max(
                candidates,
                key=lambda k: len(configs_for_test_and_val[k])
                - test_samples_per_intent[k],
            )
            test_samples_per_intent[max_intent] += 1
            current_test_total += 1

        # Now split each intent's configs into test and validation
        test_episodes = []
        val_episodes_from_test_val = []

        print(f"\n📋 Stratified sampling by intent_template_id (test+val configs):")
        print("-" * 80)

        for task_name in sorted_intents:
            configs = configs_for_test_and_val[task_name]
            n_test = test_samples_per_intent[task_name]
            n_val = len(configs) - n_test

            # Shuffle configs for random sampling
            shuffled_configs = configs.copy()
            random.shuffle(shuffled_configs)

            # Split into test and validation
            test_configs = shuffled_configs[:n_test]
            val_configs = shuffled_configs[n_test:]

            # Convert to episode format
            for config_info in test_configs:
                episode = {
                    "task_name": config_info["task_name"],
                    "variation_id": f"task_id_{config_info['task_id']}",
                    "task_id": config_info["task_id"],
                    "intent_template_id": config_info["intent_template_id"],
                    "config_file": config_info["config_file"],
                    "max_steps": max_steps,
                    "intent": config_info["intent"],
                    "sites": config_info["sites"],
                }
                test_episodes.append(episode)

            for config_info in val_configs:
                episode = {
                    "task_name": config_info["task_name"],
                    "variation_id": f"task_id_{config_info['task_id']}",
                    "task_id": config_info["task_id"],
                    "intent_template_id": config_info["intent_template_id"],
                    "config_file": config_info["config_file"],
                    "max_steps": max_steps,
                    "intent": config_info["intent"],
                    "sites": config_info["sites"],
                }
                val_episodes_from_test_val.append(episode)

            print(
                f"   {task_name:<20} -> Total: {len(configs):>3}  |  Test: {n_test:>3}  |  Val: {n_val:>3}"
            )

    # Add val-only configs to validation
    val_only_episodes = []
    if configs_for_val_only:
        print(f"\n📋 Adding val-only configs (excluded from test):")
        print("-" * 80)

        for task_name in sorted(configs_for_val_only.keys()):
            configs = configs_for_val_only[task_name]

            for config_info in configs:
                episode = {
                    "task_name": config_info["task_name"],
                    "variation_id": f"task_id_{config_info['task_id']}",
                    "task_id": config_info["task_id"],
                    "intent_template_id": config_info["intent_template_id"],
                    "config_file": config_info["config_file"],
                    "max_steps": max_steps,
                    "intent": config_info["intent"],
                    "sites": config_info["sites"],
                }
                val_only_episodes.append(episode)

            sites_list = set()
            for c in configs:
                sites_list.update(c["sites"])
            print(
                f"   {task_name:<20} -> Val: {len(configs):>3} (sites: {', '.join(sorted(sites_list))})"
            )

    # Combine validation episodes
    validation_episodes = val_episodes_from_test_val + val_only_episodes

    print(f"\n📊 Final split sizes:")
    print(f"   Test:       {len(test_episodes):>3} episodes")
    print(f"   Validation: {len(validation_episodes):>3} episodes")
    print(f"     - From test+val pool: {len(val_episodes_from_test_val):>3}")
    print(f"     - From val-only pool: {len(val_only_episodes):>3}")
    print(f"   Total:      {len(test_episodes) + len(validation_episodes):>3} episodes")

    return test_episodes, validation_episodes


def save_official_test_split(
    all_episodes: List[Dict],
    output_dir: str,
    seed: int = 42,
    max_steps: int = DEFAULT_MAX_STEPS,
):
    """
    Save complete dataset as official test split with README containing statistics.

    Args:
        all_episodes: List of all episodes (complete dataset)
        output_dir: Output directory for official test split
        seed: Random seed used for reproducibility
        max_steps: Max steps assigned to episodes
    """
    output_path = Path(output_dir) / "official_test"
    output_path.mkdir(parents=True, exist_ok=True)

    # Sort episodes by task_id for deterministic ordering
    all_episodes_sorted = sorted(all_episodes, key=lambda x: x["task_id"])

    # Save complete test set (flat list)
    test_file = output_path / "test.json"
    with test_file.open("w", encoding="utf-8") as f:
        json.dump(all_episodes_sorted, f, indent=2, ensure_ascii=False)

    unique_intents = len(set(ep["task_name"] for ep in all_episodes_sorted))
    print(f"✅ Saved official test split: {test_file}")
    print(f"   Episodes: {len(all_episodes_sorted)}")
    print(f"   Unique intent templates: {unique_intents}")

    # Create comprehensive statistics
    stats = _calculate_comprehensive_stats(all_episodes_sorted)

    # Create README with statistics
    _create_official_test_readme(output_path, stats, seed, max_steps)

    return True


def _calculate_comprehensive_stats(episodes: List[Dict]) -> Dict:
    """Calculate comprehensive statistics for the episode list."""

    # Intent distribution
    intent_distribution = defaultdict(int)
    intent_to_sites = defaultdict(set)

    for episode in episodes:
        intent = episode["task_name"]
        intent_distribution[intent] += 1
        intent_to_sites[intent].update(episode.get("sites", []))

    # Site distribution
    all_sites = []
    for episode in episodes:
        all_sites.extend(episode.get("sites", []))

    site_distribution = defaultdict(int)
    for site in all_sites:
        site_distribution[site] += 1

    return {
        "total_episodes": len(episodes),
        "unique_intents": len(intent_distribution),
        "intent_distribution": dict(sorted(intent_distribution.items())),
        "intent_to_sites": {k: sorted(v) for k, v in intent_to_sites.items()},
        "site_distribution": dict(
            sorted(site_distribution.items(), key=lambda x: x[1], reverse=True)
        ),
        "unique_sites": len(site_distribution),
    }


def _create_official_test_readme(
    output_dir: Path, stats: Dict, seed: int, max_steps: int
):
    """Create README.md with comprehensive statistics."""
    readme_path = output_dir / "README.md"

    with readme_path.open("w", encoding="utf-8") as f:
        f.write("# Official WebArena Test Set\n\n")
        f.write(
            "This is the **official WebArena test set** comprising all available WebArena task configurations.\n\n"
        )
        f.write(
            "WebArena is a benchmark for web agents that can follow natural language instructions to complete tasks on websites.\n\n"
        )
        f.write(f"**Created:** {datetime.now().isoformat()}\n\n")
        f.write("---\n\n")

        # Global statistics
        f.write("## 📊 Global Statistics\n\n")
        f.write(f"- **Total Tasks:** {stats['total_episodes']:,}\n")
        f.write(f"- **Unique Intent Templates:** {stats['unique_intents']}\n")
        f.write(f"- **Unique Sites:** {stats['unique_sites']}\n")
        f.write(f"- **Max Steps per Task:** {max_steps}\n")
        f.write(f"- **Random Seed:** {seed} (deterministic ordering)\n\n")

        # Site distribution
        f.write("## 🌐 Site Distribution\n\n")
        f.write("| Site | Tasks | Percentage |\n")
        f.write("|------|-------|------------|\n")

        total_site_refs = sum(stats["site_distribution"].values())
        for site, count in stats["site_distribution"].items():
            percentage = (count / total_site_refs) * 100
            f.write(f"| {site} | {count} | {percentage:.1f}% |\n")

        f.write(
            f"\n**Note:** Tasks can involve multiple sites, so percentages may sum to >100%.\n\n"
        )

        # Intent template distribution
        f.write("## 📋 Intent Template Distribution\n\n")
        f.write("| Intent Template | Tasks | Percentage | Sites |\n")
        f.write("|-----------------|-------|------------|-------|\n")

        sorted_intents = sorted(
            stats["intent_distribution"].items(), key=lambda x: int(x[0].split("_")[-1])
        )
        for intent, count in sorted_intents:
            percentage = (count / stats["total_episodes"]) * 100
            sites = ", ".join(stats["intent_to_sites"].get(intent, []))
            f.write(f"| {intent} | {count} | {percentage:.1f}% | {sites} |\n")

        f.write("\n---\n\n")

        # Usage section
        f.write("## 🚀 Usage\n\n")
        f.write("### For Publication Results\n\n")
        f.write("Use this complete test set for evaluation:\n\n")
        f.write("```bash\n")
        f.write("python -m traj_retrieval.run_webarena \\\n")
        f.write("  --config traj_retrieval/run_evaluation_config.yaml \\\n")
        f.write(
            "  --evaluation-set traj_retrieval/preprocess/new_splits/webarena/official_test/test.json\n"
        )
        f.write("```\n\n")

        f.write("### Subset Evaluation (Optional)\n\n")
        f.write(
            "For faster iteration during development, use `--start-idx` and `--end-idx`:\n\n"
        )
        f.write("```bash\n")
        f.write("# Evaluate first 10 tasks\n")
        f.write("python -m traj_retrieval.run_webarena \\\n")
        f.write("  --config traj_retrieval/run_evaluation_config.yaml \\\n")
        f.write(
            "  --evaluation-set traj_retrieval/preprocess/new_splits/webarena/official_test/test.json \\\n"
        )
        f.write("  --start-idx 0 --end-idx 10\n")
        f.write("```\n\n")

        # Data format
        f.write("## 📝 Data Format\n\n")
        f.write("Each task in `test.json` is a dictionary with:\n\n")
        f.write("```json\n")
        f.write("{\n")
        f.write('  "task_name": "intent_template_id_0",\n')
        f.write('  "variation_id": "task_id_0",\n')
        f.write('  "task_id": 0,\n')
        f.write('  "intent_template_id": 0,\n')
        f.write('  "config_file": "/path/to/config_files/0.json",\n')
        f.write('  "max_steps": 30,\n')
        f.write('  "intent": "Check the price of...",\n')
        f.write('  "sites": ["shopping"]\n')
        f.write("}\n")
        f.write("```\n\n")

        # Key fields
        f.write("### Key Fields\n\n")
        f.write("- **task_name**: Intent template identifier (intent_template_id_N)\n")
        f.write("- **variation_id**: Task variation identifier (task_id_N)\n")
        f.write("- **task_id**: Unique task ID from WebArena\n")
        f.write("- **config_file**: Path to WebArena config JSON file\n")
        f.write("- **intent**: Natural language task instruction\n")
        f.write("- **sites**: List of websites involved in the task\n")
        f.write("- **max_steps**: Maximum number of steps allowed\n\n")

        # Evaluation protocol
        f.write("## 📊 Evaluation Protocol\n\n")
        f.write("### Standard Reporting Format\n\n")
        f.write("Report success rate on the complete test set:\n\n")
        f.write("| Method | Success Rate (%) | Notes |\n")
        f.write("|--------|------------------|-------|\n")
        f.write(
            "| **Your Method** | **X.X** | {tasks completed} / {total_episodes} |\n\n"
        )

        f.write("### Metrics to Report\n\n")
        f.write(
            "1. **Overall Success Rate**: Percentage of successfully completed tasks\n"
        )
        f.write("2. **Per-Site Success Rate**: Success rate broken down by website\n")
        f.write("3. **Per-Intent Success Rate**: Success rate by intent template\n")
        f.write(
            "4. **Average Steps**: Average number of steps taken (for successful tasks)\n\n"
        )

        # Source information
        f.write("## 🔍 Source\n\n")
        f.write("This test set is derived from the official WebArena benchmark:\n\n")
        f.write(f"- **Source Directory**: `{WEBARENA_CONFIG_DIR}`\n")
        f.write(f"- **Total Configurations**: {stats['total_episodes']}\n")
        f.write("- **Format**: WebArena config JSON files\n")
        f.write("- **Ordering**: Deterministic (sorted by task_id)\n\n")

        # Citation
        f.write("## 📄 Citation\n\n")
        f.write("If you use this test set, please cite the WebArena paper:\n\n")
        f.write("```bibtex\n")
        f.write("@inproceedings{webarena2023,\n")
        f.write(
            "  title={WebArena: A Realistic Web Environment for Building Autonomous Agents},\n"
        )
        f.write(
            "  author={Zhou, Shuyan and Xu, Frank F. and Zhu, Hao and Zhou, Xuhui and Lo, Robert and Sridhar, Abishek and Cheng, Xianyi and Bisk, Yonatan and Fried, Daniel and Alon, Uri and others},\n"
        )
        f.write("  booktitle={ICLR},\n")
        f.write("  year={2024}\n")
        f.write("}\n")
        f.write("```\n\n")

        # Notes
        f.write("## 📌 Notes\n\n")
        f.write(
            "- **Complete Dataset**: This contains ALL available WebArena task configurations\n"
        )
        f.write(
            "- **Deterministic**: Tasks are sorted by task_id for reproducibility\n"
        )
        f.write("- **Multi-Site Tasks**: Some tasks involve multiple websites\n")
        f.write("- **Browser-Based**: Requires Playwright and actual website access\n")
        f.write(
            "- **Synchronous Execution**: Use `run_webarena.py` (not `run_evaluation.py`)\n\n"
        )

        # Environment setup
        f.write("## ⚙️ Environment Requirements\n\n")
        f.write("WebArena requires:\n")
        f.write("1. **Playwright** with browser drivers installed\n")
        f.write("2. **WebArena websites** running and accessible\n")
        f.write("3. **Synchronous execution** (episodes run one at a time)\n\n")
        f.write("See WebArena documentation for detailed setup instructions.\n")

    print(f"✅ Created README.md with comprehensive statistics")


def save_splits(
    test_episodes: List[Dict],
    validation_episodes: List[Dict],
    output_dir: str,
    seed: int = 42,
    max_steps: int = DEFAULT_MAX_STEPS,
):
    """
    Save test and validation splits to JSON files (flat list format).

    Args:
        test_episodes: List of test episodes
        validation_episodes: List of validation episodes
        output_dir: Output directory for split files
        seed: Random seed used for split creation
        max_steps: Max steps assigned to episodes
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save test split (flat list)
    test_file = output_path / "test.json"
    with test_file.open("w", encoding="utf-8") as f:
        json.dump(test_episodes, f, indent=2, ensure_ascii=False)

    test_unique_intents = len(set(ep["task_name"] for ep in test_episodes))
    print(f"✅ Saved test split: {test_file}")
    print(f"   Episodes: {len(test_episodes)}")
    print(f"   Unique intents: {test_unique_intents}")

    # Save validation split (flat list)
    validation_file = output_path / "validation.json"
    with validation_file.open("w", encoding="utf-8") as f:
        json.dump(validation_episodes, f, indent=2, ensure_ascii=False)

    val_unique_intents = len(set(ep["task_name"] for ep in validation_episodes))
    print(f"✅ Saved validation split: {validation_file}")
    print(f"   Episodes: {len(validation_episodes)}")
    print(f"   Unique intents: {val_unique_intents}")

    return True


def save_train_index_source(validation_episodes: List[Dict], output_dir: str):
    """
    Save train-only index source manifest.

    For WebArena train-only workflow, we use validation split as train source.
    """
    output_path = Path(output_dir) / "train"
    output_path.mkdir(parents=True, exist_ok=True)

    # Deterministic ordering
    manifest_episodes = sorted(
        validation_episodes,
        key=lambda x: (x.get("task_name", ""), int(x.get("task_id", -1))),
    )

    # Keep full episode fields and annotate source split
    manifest_payload = []
    for episode in manifest_episodes:
        item = dict(episode)
        item["source_split"] = "validation"
        manifest_payload.append(item)

    output_file = output_path / TRAIN_INDEX_SOURCE_FILENAME
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved train index source: {output_file}")
    print(f"   Episodes: {len(manifest_payload)}")
    return True


def print_split_statistics(test_episodes: List[Dict], validation_episodes: List[Dict]):
    """Print detailed statistics about the splits."""
    print(f"\n📈 Split Statistics:")
    print("=" * 100)

    # Calculate distribution by intent
    test_by_intent = defaultdict(int)
    val_by_intent = defaultdict(int)

    for episode in test_episodes:
        test_by_intent[episode["task_name"]] += 1

    for episode in validation_episodes:
        val_by_intent[episode["task_name"]] += 1

    # Print per-intent statistics
    all_intents = sorted(set(list(test_by_intent.keys()) + list(val_by_intent.keys())))

    print(
        f"\n{'Intent Template':<20} | {'Test':>6} | {'Val':>6} | {'Total':>6} | {'Test %':>8}"
    )
    print("-" * 80)

    for intent in all_intents:
        test_count = test_by_intent[intent]
        val_count = val_by_intent[intent]
        total_count = test_count + val_count
        test_pct = (test_count / total_count * 100) if total_count > 0 else 0

        print(
            f"{intent:<20} | {test_count:>6} | {val_count:>6} | {total_count:>6} | {test_pct:>7.1f}%"
        )

    # Overall statistics
    print("-" * 80)
    total_test = len(test_episodes)
    total_val = len(validation_episodes)
    total_all = total_test + total_val
    test_pct_overall = (total_test / total_all * 100) if total_all > 0 else 0

    print(
        f"{'TOTAL':<20} | {total_test:>6} | {total_val:>6} | {total_all:>6} | {test_pct_overall:>7.1f}%"
    )
    print(f"\nUnique intent templates: {len(all_intents)}")


def main():
    """Main entry point for WebArena train-only split creation."""
    # Configuration
    TEST_SIZE = 100
    SEED = 42
    MAX_STEPS = DEFAULT_MAX_STEPS
    SHOW_STATS = True

    print("🌐 WebArena Data Splits Creator (TRAIN-ONLY INDEX + TEST EVAL)")
    print("=" * 80)
    print(f"Creating train-only index source + test/validation + official test:")
    print(f"  - Config directory: {WEBARENA_CONFIG_DIR}")
    print(f"  - Output directory: {SPLITS_OUTPUT_DIR}")
    print(f"  - Test size: {TEST_SIZE} episodes")
    print(f"  - Strategy: Stratified (if possible), otherwise random")
    print(f"  - Max steps: {MAX_STEPS} per episode")
    print(f"  - Random seed: {SEED} (ensures reproducibility)")
    print(f"  - Excluding from test: sites containing {EXCLUDE_FROM_TEST_SITES}")
    print("=" * 80)

    # Step 1: Scan config files and separate by site filtering
    configs_for_test_and_val, configs_for_val_only = scan_webarena_configs(
        WEBARENA_CONFIG_DIR
    )

    if not configs_for_test_and_val and not configs_for_val_only:
        print("❌ No config files found! Check your config directory.")
        return 1

    # Step 2: Create stratified splits
    test_episodes, validation_episodes = create_stratified_splits(
        configs_for_test_and_val,
        configs_for_val_only,
        test_size=TEST_SIZE,
        seed=SEED,
        max_steps=MAX_STEPS,
    )

    if not test_episodes and not validation_episodes:
        print("❌ Failed to create splits!")
        return 1

    # Step 3: Print statistics
    if SHOW_STATS:
        print_split_statistics(test_episodes, validation_episodes)

    # Step 4: Save stratified splits
    print(f"\n💾 Saving stratified splits to {SPLITS_OUTPUT_DIR}...")
    print("=" * 80)

    if not save_splits(
        test_episodes, validation_episodes, SPLITS_OUTPUT_DIR, SEED, MAX_STEPS
    ):
        print(f"\n❌ Failed to save stratified splits")
        return 1

    # Step 4.5: Save train-only index source (from validation split)
    print(f"\n💾 Saving train-only index source manifest...")
    print("=" * 80)

    if not save_train_index_source(validation_episodes, SPLITS_OUTPUT_DIR):
        print(f"\n❌ Failed to save train index source manifest")
        return 1

    # Step 5: Create and save official test set (complete dataset)
    print(f"\n💾 Creating official test set (complete dataset)...")
    print("=" * 80)

    # Combine all episodes for official test set
    all_episodes = test_episodes + validation_episodes

    if not save_official_test_split(all_episodes, SPLITS_OUTPUT_DIR, SEED, MAX_STEPS):
        print(f"\n❌ Failed to save official test split")
        return 1

    print(f"\n🎉 WebArena splits created successfully!")
    print(f"\n📁 Stratified Splits (for development/ablations):")
    print(f"   Test: {len(test_episodes)} episodes")
    print(f"   Validation: {len(validation_episodes)} episodes")
    print(f"   Files: {SPLITS_OUTPUT_DIR}/test.json, validation.json")

    print(f"\n📁 Train-Only Index Source:")
    print(f"   Source split: validation")
    print(f"   Episodes: {len(validation_episodes)}")
    print(f"   File: {SPLITS_OUTPUT_DIR}/train/{TRAIN_INDEX_SOURCE_FILENAME}")

    print(f"\n📁 Official Test Set (for publication):")
    print(f"   Complete dataset: {len(all_episodes)} episodes")
    print(
        f"   Unique intent templates: {len(set(ep['task_name'] for ep in all_episodes))}"
    )
    print(f"   Files: {SPLITS_OUTPUT_DIR}/official_test/test.json, README.md")

    print(f"\n✨ Key Information:")
    print(f"   - Random seed: {SEED} (deterministic ordering)")
    print(f"   - Max steps: {MAX_STEPS} per episode")
    print(f"   - All files ready for run_webarena.py")

    return 0


if __name__ == "__main__":
    exit(main())
