"""
Synchronous evaluation runner specifically for WebArena.

WebArena uses Playwright's sync API which is incompatible with asyncio event loops.
This runner executes episodes SERIALLY (one at a time) in a completely synchronous manner.

Usage:
    python -m traj_retrieval.run_webarena --config traj_retrieval/run_evaluation_config.yaml
"""

import json
import argparse
import time
import os
import sys
import uuid
import threading
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Import core functionality
from .core.sync_with_traj_rag import (
    run_episode_sync,
)  # Synchronous analog of async_with_traj_rag
from .core.environment_factory import EnvironmentFactory
from .core.retrieval_strategy import RetrievalManager
from .core.simulation_loader import SimulationActionsLoader  # NEW: For simulate_till_tk
from .core.webarena_online_memory import (
    WebArenaOnlineMemoryWriter,
    online_memory_enabled_from_env,
)
from .utils.metrics_util import (
    create_enhanced_metrics,
    write_detailed_debug_json,
    write_metrics_json,
    write_metrics_csv,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class WebArenaRunner:
    """
    Synchronous runner for WebArena episodes.

    This runner executes episodes serially due to Playwright sync API limitations.
    """

    def __init__(
        self,
        environment_name: str,
        evaluation_set_path: str,
        indices_dir: str,
        out_dir: str,
        model: str,
        retrieval_config: Dict,
        specific_task: Optional[str] = None,
        max_steps_override: Optional[Any] = None,
        evaluation_run_id: str = "random",
        frequency_strategy: str = "none",
        retrieval_type: str = "trajectory",
        history_window: int = 10,
        experiment_type: str = "action_string_direct",
        env_config: Optional[Dict] = None,
        remove_actions_after_use: Optional[List[str]] = None,
        ultimate_fallback_action: str = "none",  # WebArena uses "none" action as fallback
        agent_call_policy: str = "normal_then_structured",
        start_idx: int = 0,  # Starting index for evaluation set slicing (0 = beginning)
        end_idx: int = -1,  # Ending index for evaluation set slicing (-1 = to the end)
        environment_specific_config: Optional[
            Dict
        ] = None,  # New: complete environment config from YAML for logging
        global_python_cut_off_step: Optional[
            int
        ] = None,  # New: global hard limit on Python steps (overrides max_steps)
    ):
        self.environment_name = environment_name
        self.evaluation_set_path = evaluation_set_path
        self.indices_dir = indices_dir
        self.out_dir = out_dir
        self.model = model
        self.retrieval_config = retrieval_config
        self.specific_task = specific_task
        self.max_steps_override = max_steps_override
        self.evaluation_run_id = evaluation_run_id
        self.frequency_strategy = frequency_strategy
        self.retrieval_type = retrieval_type
        self.agentic_until_step_k = self.retrieval_config.get("agentic_until_step_k")
        self.retrieve_once_step_k = self.retrieval_config.get("retrieve_once_step_k")
        self.history_window = history_window
        self.experiment_type = experiment_type
        self.env_config = env_config or {}
        self.remove_actions_after_use = remove_actions_after_use or []
        self.ultimate_fallback_action = ultimate_fallback_action
        self.agent_call_policy = agent_call_policy
        self.start_idx = start_idx  # Store start index for slicing
        self.end_idx = end_idx  # Store end index for slicing
        self.environment_specific_config = (
            environment_specific_config or {}
        )  # Store complete environment config from YAML for logging
        self.global_python_cut_off_step = (
            global_python_cut_off_step  # Store global Python step cutoff
        )
        self.online_memory_enabled = online_memory_enabled_from_env()
        self.online_memory_commit_policy = (
            os.environ.get("ONLINE_MEMORY_COMMIT_POLICY", "per_episode").strip()
            or "per_episode"
        )
        self.online_memory_refresh_before_episode = os.environ.get(
            "ONLINE_MEMORY_REFRESH_BEFORE_EPISODE", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.online_memory_refresh_after_commit = os.environ.get(
            "ONLINE_MEMORY_REFRESH_AFTER_COMMIT", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.online_memory_writer = None
        self._online_memory_commit_lock = threading.Lock()

        # NEW: Initialize simulation loader for simulate_till_tk strategy
        self.simulation_loader = None
        if self.frequency_strategy == "simulate_till_tk":
            # Extract no_retrieval_runs_file from retrieval config (with {environment_name} substitution)
            no_retrieval_runs_file_template = self.retrieval_config.get(
                "no_retrieval_runs_file"
            )
            if not no_retrieval_runs_file_template:
                raise ValueError(
                    "simulate_till_tk strategy requires 'no_retrieval_runs_file' in retrieval_config"
                )

            # Substitute {environment_name} placeholder
            no_retrieval_runs_file = no_retrieval_runs_file_template.replace(
                "{environment_name}", self.environment_name
            )

            # Initialize simulation loader (will fail-fast if file doesn't exist)
            # Note: simulate_step_k is now read from each episode descriptor, not from config
            self.simulation_loader = SimulationActionsLoader(
                json_file_path=no_retrieval_runs_file,
                environment_name=self.environment_name,
            )

            print(f"✅ Initialized simulate_till_tk strategy:")
            print(f"   - Simulation file: {no_retrieval_runs_file}")
            print(
                f"   - simulate_step_k will be read from each episode in evaluation set"
            )
            print(
                f"   - Episodes with simulation data: {self.simulation_loader.get_episode_count()}"
            )
        elif self.frequency_strategy == "pseudo_simulate_till_closest_tk":
            # Initialize simulation loader for pseudo_simulate_till_closest_tk strategy
            # This strategy needs the simulation data to get target observations at step k
            no_retrieval_runs_file_template = self.retrieval_config.get(
                "no_retrieval_runs_file"
            )
            if not no_retrieval_runs_file_template:
                raise ValueError(
                    "pseudo_simulate_till_closest_tk strategy requires 'no_retrieval_runs_file' in retrieval_config"
                )

            # Substitute {environment_name} placeholder
            no_retrieval_runs_file = no_retrieval_runs_file_template.replace(
                "{environment_name}", self.environment_name
            )

            # Initialize simulation loader (will fail-fast if file doesn't exist)
            self.simulation_loader = SimulationActionsLoader(
                json_file_path=no_retrieval_runs_file,
                environment_name=self.environment_name,
            )

            print(f"✅ Initialized pseudo_simulate_till_closest_tk strategy:")
            print(f"   - Simulation file: {no_retrieval_runs_file}")
            print(
                f"   - Target step k (simulate_step_k) will be read from each episode in evaluation set"
            )
            print(
                f"   - Will start with no retrieval, monitoring for observation similarity"
            )
            print(
                f"   - Retrieval triggers when current observation matches target observation at step k"
            )
            print(
                f"   - Episodes with simulation data: {self.simulation_loader.get_episode_count()}"
            )
        elif self.frequency_strategy == "agentic_until_tk":
            if self.agentic_until_step_k is None:
                raise ValueError(
                    "agentic_until_tk strategy requires 'agentic_until_step_k' in retrieval_config"
                )
            if (
                not isinstance(self.agentic_until_step_k, int)
                or self.agentic_until_step_k < 0
            ):
                raise ValueError(
                    "retrieval.agentic_until_step_k must be a non-negative integer"
                )
            print(f"✅ Initialized agentic_until_tk strategy:")
            print(
                f"   - Agentic retrieval window: steps [0, {self.agentic_until_step_k})"
            )
            print(f"   - Retrieval mode from step {self.agentic_until_step_k}: none")
        elif self.frequency_strategy == "retrieve_once_at_tk":
            if self.retrieve_once_step_k is None:
                raise ValueError(
                    "retrieve_once_at_tk strategy requires 'retrieve_once_step_k' in retrieval_config"
                )
            if (
                not isinstance(self.retrieve_once_step_k, int)
                or self.retrieve_once_step_k < 0
            ):
                raise ValueError(
                    "retrieval.retrieve_once_step_k must be a non-negative integer"
                )
            print(f"✅ Initialized retrieve_once_at_tk strategy:")
            print(f"   - Forced one-time retrieval step: {self.retrieve_once_step_k}")
            print(f"   - Retrieval mode before/after that step: none")

        # Store environment name for creating handlers
        # Note: We'll create handlers dynamically per episode

        # Load and parse evaluation set
        self._load_evaluation_set()

        # Apply max_steps override if specified
        self._apply_max_steps_override()

        # Filter by specific task if requested
        if self.specific_task:
            self.all_episodes = [
                ep for ep in self.all_episodes if ep["task_name"] == self.specific_task
            ]
            print(f"🎯 Filtering to specific task: {self.specific_task}")
            print(f"   Found {len(self.all_episodes)} episodes for this task")

        # Apply evaluation set slicing if specified
        self._apply_evaluation_set_slicing()

        # Generate or use provided evaluation_run_id
        if self.evaluation_run_id == "random":
            self.actual_evaluation_run_id = str(uuid.uuid4())[:8]
            print(
                f"[INFO] Generated random evaluation run_id: {self.actual_evaluation_run_id}"
            )
        else:
            self.actual_evaluation_run_id = self.evaluation_run_id
            print(
                f"[INFO] Using provided evaluation run_id: {self.actual_evaluation_run_id}"
            )

        # Create evaluation run-specific directory
        self.base_output_dir = Path(self.out_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_run_dir = (
            self.base_output_dir / f"run_{self.actual_evaluation_run_id}"
        )
        self.evaluation_run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Created evaluation run directory: {self.evaluation_run_dir}")

        # Get unique task names
        unique_tasks = set(ep["task_name"] for ep in self.all_episodes)

        print(f"\n🎯 WebArena Synchronous Runner initialized:")
        print(f"   - Environment: {self.environment_name}")
        print(f"   - ⚠️  Execution Mode: SERIAL (one episode at a time)")
        print(f"   - Unique tasks to run: {len(unique_tasks)}")
        print(f"   - Total episodes: {len(self.all_episodes)}")
        print(f"   - Retrieval Strategy: {self.frequency_strategy}")
        print(f"   - Experiment type: {self.experiment_type}")
        if self.env_config:
            print(f"   - Environment Config:")
            for key, value in self.env_config.items():
                print(f"     - {key}: {value}")
        print(f"   - Online memory enabled: {self.online_memory_enabled}")
        if self.online_memory_enabled:
            print(
                f"   - Online memory commit policy: {self.online_memory_commit_policy}"
            )
        print()

    def _load_evaluation_set(self):
        """Load evaluation set from JSON file."""
        with open(self.evaluation_set_path, "r") as f:
            eval_data = json.load(f)

        # Parse using handler
        temp_handler = EnvironmentFactory.create_handler(
            self.environment_name, **self.env_config
        )
        self.all_episodes = temp_handler.parse_evaluation_set(eval_data)
        temp_handler.close()

        print(f"✅ Loaded evaluation set: {self.evaluation_set_path}")
        print(f"   - Total episodes: {len(self.all_episodes)}")

    def _apply_max_steps_override(self):
        """Apply max_steps override based on configuration."""
        if self.max_steps_override is None:
            print(
                f"[MaxSteps] Using max_steps from evaluation set (default: no override)"
            )
            return

        if isinstance(self.max_steps_override, int):
            print(
                f"[MaxSteps] Overriding all episodes with max_steps={self.max_steps_override}"
            )
            for episode in self.all_episodes:
                episode["max_steps"] = self.max_steps_override
            return

        if isinstance(
            self.max_steps_override, str
        ) and self.max_steps_override.endswith("_given"):
            try:
                multiplier = float(self.max_steps_override.replace("_given", ""))
                print(f"[MaxSteps] Multiplying max_steps by {multiplier}")
                for episode in self.all_episodes:
                    episode["max_steps"] = int(episode["max_steps"] * multiplier)
                return
            except ValueError:
                raise ValueError(
                    f"Invalid max_steps multiplier: {self.max_steps_override}"
                )

        raise ValueError(f"Invalid max_steps value: {self.max_steps_override}")

    def _apply_evaluation_set_slicing(self):
        """
        Apply evaluation set slicing based on start_idx and end_idx.

        - start_idx: starting index (0-indexed, inclusive). Default: 0 (beginning)
        - end_idx: ending index (0-indexed, exclusive). Use -1 to mean "to the end"
        - Both use Python slicing semantics (0-indexed, end exclusive)
        """
        original_count = len(self.all_episodes)

        # Convert -1 to None for Python slicing (means "to the end")
        start = self.start_idx
        end = None if self.end_idx == -1 else self.end_idx

        # If start is 0 and end is None, no slicing needed
        if start == 0 and end is None:
            print(
                f"[Slicing] No slicing applied (processing all {original_count} episodes)"
            )
            return

        # Apply slicing
        self.all_episodes = self.all_episodes[start:end]

        sliced_count = len(self.all_episodes)

        # Log the slicing operation
        if sliced_count < original_count:
            print(f"[Slicing] Applied evaluation set slicing:")
            print(f"   - Original episodes: {original_count}")
            print(f"   - Start index: {start}")
            print(f"   - End index: {end if end is not None else 'end'}")
            print(f"   - Sliced episodes: {sliced_count}")
        else:
            print(f"[Slicing] No episodes filtered (slice covers entire range)")

    def _validate_retrieval_resources(self):
        """
        Validate that retrieval resources are available if retrieval is enabled.

        This provides fail-fast behavior: if retrieval is requested but indices
        are not available, the evaluation fails immediately at startup instead of
        during the first retrieval attempt.

        Raises:
            RuntimeError: If retrieval is enabled but required resources are missing
        """
        # If retrieval is disabled, no validation needed
        if self.frequency_strategy == "none":
            print(
                f"[Validation] Retrieval disabled (frequency_strategy='none'), skipping resource validation"
            )
            return

        print(
            f"\n[Validation] Validating retrieval resources (frequency_strategy='{self.frequency_strategy}')..."
        )

        # Check 1: indices_dir must exist
        indices_path = Path(self.indices_dir)
        if not indices_path.exists():
            raise RuntimeError(
                f"\n{'='*80}\n"
                f"❌ VALIDATION FAILED: Retrieval indices directory not found\n"
                f"{'='*80}\n"
                f"Configuration specifies retrieval strategy: '{self.frequency_strategy}'\n"
                f"But indices directory does not exist: {self.indices_dir}\n\n"
                f"Options:\n"
                f"  1. Set frequency_strategy to 'none' in config to disable retrieval\n"
                f"  2. Create indices using: python -m traj_retrieval.preprocess.create_lancedb_indices\n"
                f"  3. Update indices_dir path in config to point to existing indices\n"
                f"{'='*80}\n"
            )

        print(f"[Validation]   ✓ Indices directory exists: {self.indices_dir}")

        # Check 2: LanceDB database must be valid
        try:
            import lancedb
        except ImportError:
            raise RuntimeError(
                f"\n{'='*80}\n"
                f"❌ VALIDATION FAILED: lancedb package not installed\n"
                f"{'='*80}\n"
                f"Configuration specifies retrieval strategy: '{self.frequency_strategy}'\n"
                f"But lancedb package is not installed.\n\n"
                f"Install it with: pip install lancedb\n"
                f"{'='*80}\n"
            )

        try:
            db = lancedb.connect(str(indices_path))
            table_names = db.table_names()

            if not table_names:
                raise RuntimeError(
                    f"\n{'='*80}\n"
                    f"❌ VALIDATION FAILED: No tables found in LanceDB\n"
                    f"{'='*80}\n"
                    f"Configuration specifies retrieval strategy: '{self.frequency_strategy}'\n"
                    f"Indices directory exists: {self.indices_dir}\n"
                    f"But no tables found in database.\n\n"
                    f"Create indices using: python -m traj_retrieval.preprocess.create_lancedb_indices\n"
                    f"{'='*80}\n"
                )

            print(f"[Validation]   ✓ LanceDB connection successful")
            print(f"[Validation]   ✓ Found {len(table_names)} table(s): {table_names}")

            # Check 3: Required table must exist
            if self.environment_name not in table_names:
                raise RuntimeError(
                    f"\n{'='*80}\n"
                    f"❌ VALIDATION FAILED: Required table not found in LanceDB\n"
                    f"{'='*80}\n"
                    f"Configuration specifies retrieval strategy: '{self.frequency_strategy}'\n"
                    f"Environment: {self.environment_name}\n"
                    f"Required table: '{self.environment_name}'\n"
                    f"Available tables: {table_names}\n\n"
                    f"Create the table using: python -m traj_retrieval.preprocess.create_lancedb_indices\n"
                    f"  --environment {self.environment_name}\n"
                    f"{'='*80}\n"
                )

            print(f"[Validation]   ✓ Required table '{self.environment_name}' exists")

            # Check 4: Table must have data
            table = db.open_table(self.environment_name)
            row_count = table.count_rows()

            if row_count == 0:
                raise RuntimeError(
                    f"\n{'='*80}\n"
                    f"❌ VALIDATION FAILED: Table is empty\n"
                    f"{'='*80}\n"
                    f"Configuration specifies retrieval strategy: '{self.frequency_strategy}'\n"
                    f"Table '{self.environment_name}' exists but contains 0 rows.\n\n"
                    f"Populate the table using: python -m traj_retrieval.preprocess.create_lancedb_indices\n"
                    f"  --environment {self.environment_name}\n"
                    f"{'='*80}\n"
                )

            print(f"[Validation]   ✓ Table contains {row_count:,} rows")

        except FileNotFoundError as e:
            raise RuntimeError(
                f"\n{'='*80}\n"
                f"❌ VALIDATION FAILED: LanceDB database not properly initialized\n"
                f"{'='*80}\n"
                f"Configuration specifies retrieval strategy: '{self.frequency_strategy}'\n"
                f"Indices directory: {self.indices_dir}\n"
                f"Error: {e}\n\n"
                f"Create indices using: python -m traj_retrieval.preprocess.create_lancedb_indices\n"
                f"{'='*80}\n"
            )
        except Exception as e:
            raise RuntimeError(
                f"\n{'='*80}\n"
                f"❌ VALIDATION FAILED: Error validating LanceDB\n"
                f"{'='*80}\n"
                f"Configuration specifies retrieval strategy: '{self.frequency_strategy}'\n"
                f"Indices directory: {self.indices_dir}\n"
                f"Error: {e}\n"
                f"{'='*80}\n"
            )

        print(f"[Validation] ✅ All retrieval resources validated successfully\n")

    def _initialize_resources(self):
        """Initialize resources (API key, retrieval manager)."""
        load_dotenv(REPO_ROOT / ".env")
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        # Validate retrieval resources BEFORE creating manager (fail-fast)
        self._validate_retrieval_resources()

        # Initialize retrieval manager
        self.retrieval_manager = RetrievalManager.create_from_config(
            frequency_strategy=self.frequency_strategy,
            retrieval_type=self.retrieval_type,
            indices_dir=self.indices_dir,
            search_config=self.retrieval_config.get("search_config"),
            table_name=self.environment_name,
        )

        print(f"✅ Initialized resources:")
        print(f"   - API key: [loaded from environment]")
        print(f"   - Retrieval manager: {self.retrieval_manager.get_retrieval_stats()}")

        if self.online_memory_enabled:
            self.online_memory_writer = WebArenaOnlineMemoryWriter(
                indices_dir=self.indices_dir,
                table_name=self.environment_name,
                embedding_model_name="intfloat/e5-base",
            )
            print(f"   - Online memory writer: {self.indices_dir}")

    def _build_episode_metadata(
        self, episode_descriptor: Dict[str, Any]
    ) -> Dict[str, Any]:
        metadata = dict(episode_descriptor.get("metadata", {}) or {})
        metadata.update(
            {
                "task_name": episode_descriptor.get("task_name"),
                "variation_id": episode_descriptor.get("variation_id"),
                "variation_idx": episode_descriptor.get("variation_idx"),
                "task_id": episode_descriptor.get("task_id"),
                "intent_template_id": episode_descriptor.get("intent_template_id"),
                "config_file": episode_descriptor.get("config_file"),
                "intent": episode_descriptor.get("intent"),
                "max_steps": episode_descriptor.get("max_steps"),
                "consumer_model": episode_descriptor.get("consumer_model"),
                "consumer_split_seed": episode_descriptor.get("consumer_split_seed"),
                "source_split": episode_descriptor.get("source_split"),
                "sites": metadata.get("sites", []),
            }
        )
        return metadata

    def _resolve_source_model_label(self, episode_metadata: Dict[str, Any]) -> str:
        return (
            episode_metadata.get("consumer_model")
            or episode_metadata.get("model_name")
            or self.model
        )

    def _commit_online_memory(
        self, trajectory: Dict[str, Any], episode_metadata: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if not self.online_memory_enabled or self.online_memory_writer is None:
            return None
        if self.online_memory_commit_policy != "per_episode":
            return None

        source_model_label = self._resolve_source_model_label(episode_metadata)
        with self._online_memory_commit_lock:
            summary = self.online_memory_writer.commit_runtime_trajectory(
                trajectory=trajectory,
                episode_metadata=episode_metadata,
                source_model_label=source_model_label,
            )
            if (
                self.online_memory_refresh_after_commit
                and self.retrieval_manager is not None
            ):
                self.retrieval_manager.refresh()
        print(
            "[OnlineMemory] Committed episode: "
            f"new_entries={summary['new_entries']}, duplicates={summary['skipped_duplicates']}"
        )
        return summary

    def _get_task_run_dir(self, task_name: str) -> Path:
        """Get or create task-specific run directory."""
        task_run_dir = self.evaluation_run_dir / f"run_{task_name}"
        task_run_dir.mkdir(parents=True, exist_ok=True)
        return task_run_dir

    def run_all_episodes(self):
        """Run all episodes serially and collect results."""
        self._initialize_resources()

        # CRITICAL: Ensure no asyncio event loop exists before using Playwright sync API
        try:
            import asyncio

            loop = asyncio.get_event_loop()
            if loop and not loop.is_closed():
                loop.close()
                print(
                    "[INFO] Closed existing asyncio event loop for Playwright sync API compatibility"
                )
            asyncio.set_event_loop(None)
        except Exception as e:
            print(f"[INFO] Asyncio event loop handling: {e}")

        print(f"\n🎬 Starting evaluation with {len(self.all_episodes)} episodes...")
        print(f"   Running SERIALLY (one at a time)\n")
        print("=" * 80)

        all_results = []
        results_by_task = {}
        start_time = time.time()

        for idx, episode_descriptor in enumerate(self.all_episodes, 1):
            episode_id = episode_descriptor["episode_id"]
            task_name = episode_descriptor["task_name"]
            variation_idx = episode_descriptor.get("variation_idx", 0)
            max_steps = episode_descriptor.get("max_steps", 50)
            episode_metadata = self._build_episode_metadata(episode_descriptor)

            is_unlimited = max_steps >= 10000
            max_steps_display = "unlimited" if is_unlimited else str(max_steps)
            print(
                f"\n🚀 Starting episode {idx}/{len(self.all_episodes)}: {episode_id} (max_steps: {max_steps_display})"
            )

            # CRITICAL: Clean up any lingering asyncio loops BEFORE starting episode
            # This ensures we start fresh even if previous episode had errors
            try:
                import asyncio

                try:
                    running_loop = asyncio.get_running_loop()
                    if running_loop:
                        print(
                            f"[INFO] ⚠️  Found lingering running loop before episode {episode_id}, cleaning up...",
                            flush=True,
                        )
                        try:
                            if running_loop.is_running():
                                running_loop.stop()
                            if not running_loop.is_closed():
                                running_loop.close()
                        except:
                            pass
                        asyncio.set_event_loop(None)
                        print(f"[INFO] ✓ Cleaned up lingering loop", flush=True)
                except RuntimeError:
                    pass  # No running loop, good

                # Ensure we have no current event loop
                try:
                    current_loop = asyncio.get_event_loop()
                    if current_loop and not current_loop.is_closed():
                        current_loop.close()
                    asyncio.set_event_loop(None)
                except RuntimeError:
                    pass  # No loop, good
            except Exception as e:
                print(f"[INFO] Pre-episode cleanup warning: {e}", flush=True)

            episode_start = time.time()

            try:
                if (
                    self.online_memory_enabled
                    and self.online_memory_refresh_before_episode
                    and self.retrieval_manager is not None
                ):
                    self.retrieval_manager.refresh()

                # NEW: Extract simulate_step_k from episode descriptor for simulation-based strategies
                simulate_step_k = None
                if self.frequency_strategy in [
                    "simulate_till_tk",
                    "pseudo_simulate_till_closest_tk",
                ]:
                    simulate_step_k = episode_descriptor.get("simulate_step_k")
                    if simulate_step_k is None:
                        raise RuntimeError(
                            f"\n{'='*80}\n"
                            f"❌ MISSING simulate_step_k IN EPISODE\n"
                            f"{'='*80}\n"
                            f"Frequency Strategy: {self.frequency_strategy}\n"
                            f"Episode: {episode_id}\n"
                            f"Task: {task_name}, Variation: {variation_idx}\n"
                            f"\n"
                            f"The {self.frequency_strategy} strategy requires 'simulate_step_k' field in each episode.\n"
                            f"Please add 'simulate_step_k' to the episode descriptor in the evaluation set.\n"
                            f"\n"
                            f"Example:\n"
                            f"  {{\n"
                            f'    "task_name": "{task_name}",\n'
                            f'    "variation_id": "{variation_idx}",\n'
                            f'    "simulate_step_k": 5,  <-- Add this field (step k for target observation)\n'
                            f'    "config_file": "...",\n'
                            f'    "max_steps": 30\n'
                            f"  }}\n"
                            f"{'='*80}\n"
                        )
                    print(
                        f"[Episode Config] Strategy: {self.frequency_strategy}, simulate_step_k: {simulate_step_k}"
                    )

                # Create handler for this episode (with environment-specific config)
                env_handler = EnvironmentFactory.create_handler(
                    self.environment_name, **self.env_config
                )

                # Set agent call policy on the handler (matches run_evaluation.py)
                env_handler._agent_call_policy = self.agent_call_policy

                # Initialize handler from episode descriptor (matches run_evaluation.py)
                env_handler.initialize_from_episode(episode_descriptor)

                # Preflight check: verify retrieval is possible if retrieval is enabled
                # This includes pseudo_simulate_till_closest_tk since it WILL retrieve when observation matches
                # (if no candidates exist, retrieval will fail when triggered)
                if self.frequency_strategy != "none":
                    print(
                        f"[Preflight] Checking if retrieval is possible for episode {episode_id}..."
                    )

                    # Get task/variation info for filter construction
                    check_task_name = task_name
                    check_variation_idx = variation_idx

                    # Call preflight check on retrieval strategy
                    try:
                        (
                            is_possible,
                            candidate_count,
                            filter_sql,
                        ) = self.retrieval_manager.strategy.check_retrieval_possible(
                            task_name=check_task_name, variation_idx=check_variation_idx
                        )

                        if not is_possible:
                            # No candidates available - skip this episode
                            execution_time = time.time() - episode_start

                            result = {
                                "task_name": task_name,
                                "variation_idx": variation_idx,
                                "episode_id": episode_id,
                                "max_steps": max_steps,
                                "success": False,
                                "skipped": True,
                                "skip_reason": "no_retrieval_candidates",
                                "final_score": 0,
                                "total_steps": 0,
                                "execution_time": execution_time,
                                "timestamp": datetime.now().isoformat(),
                                "trajectory": None,
                                "debug": {
                                    "preflight_check": {
                                        "candidates_found": candidate_count,
                                        "filter_sql": filter_sql,
                                        "frequency_strategy": self.frequency_strategy,
                                        "retrieval_type": self.retrieval_type,
                                    }
                                },
                            }

                            print(
                                f"⏭️  SKIPPED: {episode_id} - No retrieval candidates available"
                            )
                            print(f"   Task: {task_name}, Variation: {variation_idx}")
                            print(f"   Frequency Strategy: {self.frequency_strategy}")
                            print(f"   Filter SQL: {filter_sql}")
                            print(f"   Candidates Found: {candidate_count}")
                            print(
                                f"   → Episode skipped to avoid running without retrieval"
                            )

                            # Add to results and continue to next episode
                            all_results.append(result)
                            if task_name not in results_by_task:
                                results_by_task[task_name] = []
                            results_by_task[task_name].append(result)
                            continue
                        else:
                            print(
                                f"[Preflight] ✓ Retrieval possible: {candidate_count} candidate(s) found"
                            )
                            if filter_sql:
                                print(f"[Preflight]   Filter SQL: {filter_sql}")

                    except Exception as e:
                        # If preflight check fails, log warning but continue (fail open)
                        print(f"[Preflight] ⚠️  Preflight check failed with error: {e}")
                        print(
                            f"[Preflight]   Continuing with episode execution (fail-open policy)"
                        )

                # Run episode synchronously (standard path for WebArena)
                # Note: run_episode_sync is the synchronous analog of run_episode_async
                traj, detailed_debug = run_episode_sync(
                    env_handler=env_handler,
                    api_key=self.api_key,
                    model_slug=self.model,
                    retrieval_manager=self.retrieval_manager,
                    frequency_strategy=self.frequency_strategy,
                    history_window=self.history_window,
                    experiment_type=self.experiment_type,
                    remove_actions_after_use=self.remove_actions_after_use,
                    ultimate_fallback_action=self.ultimate_fallback_action,
                    simulation_loader=self.simulation_loader,  # NEW: For simulate_till_tk
                    simulate_step_k=simulate_step_k,  # NEW: Per-episode simulate_step_k
                    agentic_until_step_k=self.agentic_until_step_k,
                    retrieve_once_step_k=self.retrieve_once_step_k,
                    global_python_cut_off_step=self.global_python_cut_off_step,  # NEW: Global Python step cutoff
                )

                execution_time = time.time() - episode_start

                result = {
                    "task_name": task_name,
                    "variation_idx": variation_idx,
                    "variation_id": episode_descriptor.get(
                        "variation_id", variation_idx
                    ),
                    "episode_id": episode_id,
                    "max_steps": max_steps,
                    "success": traj.get("done", False),
                    "final_score": traj.get("final_score", 0),
                    "total_steps": traj.get("total_python_steps", 0),
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat(),
                    "trajectory": traj,
                    "debug": detailed_debug,
                    "episode_metadata": episode_metadata,
                }

                commit_summary = self._commit_online_memory(traj, episode_metadata)
                if commit_summary is not None:
                    result["online_memory_commit"] = commit_summary

                print(
                    f"✅ Completed: {episode_id} in {execution_time:.1f}s (Score: {result['final_score']}, steps: {result['total_steps']})"
                )

            except Exception as e:
                execution_time = time.time() - episode_start
                print(
                    f"❌ Failed: {episode_id} after {execution_time:.1f}s - {str(e)[:100]}"
                )

                # Print full traceback for debugging
                import traceback

                print(f"\n[DEBUG] Full exception trace:")
                traceback.print_exc()
                print()

                result = {
                    "task_name": task_name,
                    "variation_idx": variation_idx,
                    "episode_id": episode_id,
                    "max_steps": max_steps,
                    "success": False,
                    "final_score": 0,
                    "total_steps": 0,
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                    "error_traceback": traceback.format_exc(),
                    "trajectory": None,
                    "debug": None,
                    "episode_metadata": episode_metadata,
                }

            # CRITICAL: Clean up async loop after each episode (especially after errors)
            # This prevents cascading errors when Playwright sync API encounters a running loop
            try:
                import asyncio
                import threading

                print(
                    f"[INFO] Starting aggressive asyncio cleanup after episode {episode_id}...",
                    flush=True,
                )

                # Step 1: Try to get and stop any running loop
                try:
                    running_loop = asyncio.get_running_loop()
                    if running_loop:
                        print(
                            f"[INFO] ⚠️  Found running asyncio loop after episode {episode_id}",
                            flush=True,
                        )
                        print(
                            f"[INFO]   Loop: {running_loop}, running={running_loop.is_running()}, closed={running_loop.is_closed()}",
                            flush=True,
                        )

                        # AGGRESSIVE: Try to stop the running loop
                        # This is normally not possible, but we can try several approaches
                        try:
                            # Approach 1: Stop the loop if we can
                            if running_loop.is_running():
                                running_loop.stop()
                                print(f"[INFO] ✓ Stopped running loop", flush=True)

                            # Approach 2: Close the loop after stopping
                            if not running_loop.is_closed():
                                running_loop.close()
                                print(f"[INFO] ✓ Closed loop", flush=True)
                        except Exception as e:
                            print(
                                f"[INFO] ⚠️  Could not stop/close running loop: {e}",
                                flush=True,
                            )

                        # Always set event loop to None to break the association
                        asyncio.set_event_loop(None)
                        print(f"[INFO] ✓ Set event loop to None", flush=True)
                except RuntimeError:
                    # No running loop, which is good
                    print(f"[INFO] ✓ No running loop found (good)", flush=True)

                # Step 2: Close any existing non-running event loop
                try:
                    current_loop = asyncio.get_event_loop()
                    if current_loop:
                        print(
                            f"[INFO] Found current event loop: {current_loop}, running={current_loop.is_running()}, closed={current_loop.is_closed()}",
                            flush=True,
                        )

                        if (
                            not current_loop.is_running()
                            and not current_loop.is_closed()
                        ):
                            current_loop.close()
                            print(f"[INFO] ✓ Closed non-running event loop", flush=True)

                        # Always set to None
                        asyncio.set_event_loop(None)
                        print(f"[INFO] ✓ Set event loop to None", flush=True)
                except RuntimeError:
                    # No event loop exists, which is fine
                    print(f"[INFO] ✓ No current event loop (good)", flush=True)

                # Step 3: Create a fresh event loop policy (nuclear option)
                try:
                    # This ensures we start completely fresh
                    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
                    print(f"[INFO] ✓ Reset event loop policy to default", flush=True)
                except Exception as e:
                    print(
                        f"[INFO] ⚠️  Could not reset event loop policy: {e}", flush=True
                    )

                # Step 4: Verify cleanup was successful
                try:
                    verify_loop = asyncio.get_running_loop()
                    print(
                        f"[INFO] ❌ WARNING: Still have running loop after cleanup: {verify_loop}",
                        flush=True,
                    )
                except RuntimeError:
                    print(f"[INFO] ✅ Cleanup successful: no running loop", flush=True)

                print(
                    f"[INFO] Aggressive asyncio cleanup completed for episode {episode_id}",
                    flush=True,
                )

            except Exception as cleanup_error:
                print(
                    f"[WARNING] Error during async loop cleanup after episode {episode_id}: {cleanup_error}",
                    flush=True,
                )
                import traceback

                print(
                    f"[WARNING] Cleanup traceback: {traceback.format_exc()[:500]}",
                    flush=True,
                )

            all_results.append(result)

            # Group by task
            if task_name not in results_by_task:
                results_by_task[task_name] = []
            results_by_task[task_name].append(result)

        total_time = time.time() - start_time

        # Save results
        self._save_results(all_results, results_by_task, total_time)

        return all_results

    def _save_results(
        self, all_results: List[Dict], results_by_task: Dict, total_time: float
    ):
        """Save evaluation results with full task-based folder structure (matching run_evaluation.py)."""
        # Calculate overall metrics
        successful = sum(
            1
            for r in all_results
            if r.get("success", False) and not r.get("skipped", False)
        )
        skipped = sum(1 for r in all_results if r.get("skipped", False))
        failed = len(all_results) - successful - skipped
        avg_time = total_time / len(all_results) if all_results else 0

        # Save per-task results in organized folder structure (like run_evaluation.py)
        results_by_task_run = {}
        for task_name, task_results in results_by_task.items():
            task_run_id = f"{self.actual_evaluation_run_id}_{task_name}"
            results_by_task_run[task_run_id] = task_results

        self._save_task_results(results_by_task_run)

        # Create summary
        task_summaries = {}
        for task_name, task_results in results_by_task.items():
            total_episodes = len(task_results)
            completed_episodes = sum(1 for r in task_results if r.get("success", False))
            avg_score = (
                sum(r.get("final_score", 0) for r in task_results) / total_episodes
                if total_episodes > 0
                else 0
            )
            avg_steps = (
                sum(r.get("total_steps", 0) for r in task_results) / total_episodes
                if total_episodes > 0
                else 0
            )
            avg_execution_time = (
                sum(r.get("execution_time", 0) for r in task_results) / total_episodes
                if total_episodes > 0
                else 0
            )

            task_summaries[task_name] = {
                "total_episodes": total_episodes,
                "completed_episodes": completed_episodes,
                "completion_rate": completed_episodes / total_episodes
                if total_episodes > 0
                else 0,
                "avg_final_score": round(avg_score, 2),
                "avg_steps_per_episode": round(avg_steps, 1),
                "avg_execution_time": round(avg_execution_time, 2),
            }

        # Extract search config for easy access to top_k and rank_retrieve
        search_config = self.retrieval_config.get("search_config", {})
        candidate_config = search_config.get("candidate_generation", {})

        summary = {
            "evaluation_metadata": {
                "evaluation_set_path": self.evaluation_set_path,
                "retrieval_config": self.retrieval_config,
                "frequency_strategy": self.frequency_strategy,
                "retrieval_type": self.retrieval_type,
                "top_k": candidate_config.get("top_k", 100),  # For easy access
                "rank_retrieve": candidate_config.get(
                    "rank_retrieve", 1
                ),  # For easy access
                "evaluation_run_id": self.actual_evaluation_run_id,
                "environment": self.environment_name,
                "execution_mode": "serial_synchronous",
                "model": self.model,
                "total_episodes": len(all_results),
                "successful_episodes": successful,
                "failed_episodes": failed,
                "skipped_episodes": skipped,
                "success_rate": successful / len(all_results) if all_results else 0,
                "total_execution_time": round(total_time, 2),
                "avg_time_per_episode": round(avg_time, 2),
                "max_steps_mode": self.max_steps_override
                if self.max_steps_override is not None
                else "default",
                "start_time": datetime.fromtimestamp(
                    time.time() - total_time
                ).isoformat(),
                "end_time": datetime.now().isoformat(),
                "tasks_run": len(results_by_task),
                "environment_config": self.environment_specific_config,  # Complete environment-specific config from YAML
            },
            "task_summaries": task_summaries,
            "failed_episodes_summary": [
                {
                    "task_name": r.get("task_name", "unknown"),
                    "episode_id": r.get("episode_id", "unknown"),
                    "error": r.get("error", "Unknown error")[:200] + "..."
                    if len(r.get("error", "")) > 200
                    else r.get("error", "Unknown error"),
                }
                for r in all_results
                if not r.get("success", False) and not r.get("skipped", False)
            ][
                :10
            ],  # Limit to first 10 failures
            "skipped_episodes_summary": [
                {
                    "task_name": s.get("task_name", "unknown"),
                    "episode_id": s.get("episode_id", "unknown"),
                    "variation_idx": s.get("variation_idx", "unknown"),
                    "skip_reason": s.get("skip_reason", "unknown"),
                    "preflight_details": s.get("debug", {}).get("preflight_check", {}),
                }
                for s in all_results
                if s.get("skipped", False)
            ],
            "evaluation_set_info": {
                "total_episodes": len(self.all_episodes),
                "unique_tasks": len(set(ep["task_name"] for ep in self.all_episodes)),
                "environment": self.environment_name,
            },
        }

        # Save summary to JSON
        results_file = (
            self.evaluation_run_dir
            / f"evaluation_results_{self.frequency_strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(results_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n💾 Results saved to: {results_file}")
        print(f"   File size: {results_file.stat().st_size / 1024:.1f} KB")

        # Print summary (matching run_evaluation.py style)
        self._print_summary(summary)

    def _save_task_results(self, results_by_task_run: Dict[str, List[Dict]]):
        """Save per-task-run results in organized folder structure (matching run_evaluation.py)."""
        for task_run_id, task_results in results_by_task_run.items():
            try:
                # Get task name from the first result
                task_name = task_results[0]["task_name"]

                # Get task-specific run directory
                task_run_dir = self._get_task_run_dir(task_name)

                # Prepare episodes data for this task run (matching run_evaluation.py format)
                episodes = []
                detailed_episodes = []

                for result in task_results:
                    if result.get("trajectory"):
                        traj = result["trajectory"]
                        episode_metadata = dict(
                            result.get("episode_metadata", {}) or {}
                        )

                        # Transform trajectory to match expected format
                        action_seq = [
                            {
                                "action": s.get("action"),
                                "observation": s.get("observation"),
                                "reasoning": s.get("reasoning", ""),
                                "isCompleted": s.get("isCompleted", False),
                                "inventory": s.get("inventory", ""),
                                "reward": s.get("reward", 0),
                                "score": s.get("score", 0),
                                "url": s.get(
                                    "url", ""
                                ),  # URL for web environments (WebArena), empty string otherwise
                            }
                            for s in traj.get("steps", [])
                        ]

                        episodes.append(
                            {
                                "variationIdx": result.get(
                                    "variation_id", result.get("variation_idx", 0)
                                ),
                                "fold": "test",
                                "taskDescription": traj.get("goal_text", ""),
                                "actionSequences": action_seq,
                                "done": traj.get("done", False),
                                "finalScore": traj.get("final_score", 0),
                                "max_steps": result.get("max_steps", 0),
                                "consumer_model": episode_metadata.get(
                                    "consumer_model"
                                ),
                                "consumer_split_seed": episode_metadata.get(
                                    "consumer_split_seed"
                                ),
                            }
                        )

                        if result.get("debug"):
                            detailed_episode = dict(result["debug"])
                            detailed_episode["consumer_model"] = episode_metadata.get(
                                "consumer_model"
                            )
                            detailed_episode[
                                "consumer_split_seed"
                            ] = episode_metadata.get("consumer_split_seed")
                            detailed_episodes.append(detailed_episode)

                # Write task trajectory JSON
                traj_path = task_run_dir / f"{task_name}_trajectories_rag.json"
                with traj_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {"taskName": task_name, "episodes": episodes},
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                print(f"[IO] Wrote task trajectories: {traj_path}")

                # Write detailed debug JSON using utility function
                write_detailed_debug_json(
                    task_run_dir,
                    task_name,
                    self.frequency_strategy,
                    self.model,
                    detailed_episodes,
                    self.max_steps_override,
                    self.history_window,
                    self.indices_dir,
                )

                # Create and write metrics using utility functions with environment-specific success criteria
                metrics = create_enhanced_metrics(
                    task_name,
                    self.frequency_strategy,
                    episodes,
                    detailed_episodes,
                    self.environment_name,
                )
                write_metrics_json(
                    task_run_dir, task_name, self.frequency_strategy, metrics
                )
                write_metrics_csv(self.base_output_dir, task_run_id, metrics)

                print(
                    f"✅ Saved results for task run '{task_run_id}' in: {task_run_dir}"
                )

            except Exception as e:
                print(f"❌ Failed to save results for task run '{task_run_id}': {e}")
                import traceback

                traceback.print_exc()

    def _print_summary(self, results: Dict):
        """Print evaluation summary (matching run_evaluation.py style)."""
        metadata = results["evaluation_metadata"]

        print(f"\n📊 WebArena Evaluation Summary")
        print("=" * 60)
        print(f"Retrieval Config:")
        print(f"  - Frequency Strategy: {metadata['frequency_strategy']}")
        print(f"  - Retrieval Type: {metadata['retrieval_type']}")
        print(
            f"  - Top K: {metadata.get('top_k', 'N/A')} (trajectories retrieved for statistics)"
        )
        print(
            f"  - Rank Retrieve: {metadata.get('rank_retrieve', 'N/A')} (selected trajectory rank)"
        )
        print(f"Execution mode: {metadata['execution_mode']}")
        print(f"Total episodes: {metadata['total_episodes']}")
        print(
            f"Successful: {metadata['successful_episodes']} ({metadata['success_rate']:.1%})"
        )
        print(f"Failed: {metadata['failed_episodes']}")
        print(f"Skipped: {metadata['skipped_episodes']} (no retrieval candidates)")
        print(f"Total time: {metadata['total_execution_time']:.1f}s")
        print(f"Avg time per episode: {metadata['avg_time_per_episode']:.1f}s")
        print(f"Tasks run: {metadata['tasks_run']}")

        # Task-level summary
        if results["task_summaries"]:
            print(f"\n📈 Per-task Summary:")
            for task_name, task_summary in results["task_summaries"].items():
                print(
                    f"   - {task_name}: {task_summary['total_episodes']} episodes, "
                    f"avg score: {task_summary['avg_final_score']}, avg steps: {task_summary['avg_steps_per_episode']}, "
                    f"completed: {task_summary['completed_episodes']}/{task_summary['total_episodes']} "
                    f"({task_summary['completion_rate']:.1%})"
                )

        if results["failed_episodes_summary"]:
            print(f"\n❌ Failed episodes:")
            for failure in results["failed_episodes_summary"][
                :5
            ]:  # Show first 5 failures
                task_name = failure.get("task_name", "unknown")
                episode_id = failure.get("episode_id", "unknown")
                error = failure.get("error", "Unknown error")
                print(f"   - {episode_id} ({task_name}): {error}")

        if results["skipped_episodes_summary"]:
            print(f"\n⏭️  Skipped episodes (no retrieval candidates):")
            for skipped in results["skipped_episodes_summary"]:
                task_name = skipped.get("task_name", "unknown")
                episode_id = skipped.get("episode_id", "unknown")
                variation_idx = skipped.get("variation_idx", "unknown")
                preflight = skipped.get("preflight_details", {})
                filter_sql = preflight.get("filter_sql", "N/A")
                print(f"   - {episode_id} ({task_name}, var={variation_idx})")
                print(f"     Filter: {filter_sql}")

        print(f"\n🎉 Evaluation completed!")
        print(f"   Results saved to: {self.evaluation_run_dir}")
        print(f"   Individual episode results are organized in task folders")


def main():
    """Main entry point for WebArena synchronous evaluation."""
    parser = argparse.ArgumentParser(
        description="WebArena Synchronous Evaluation Runner"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="traj_retrieval/run_evaluation_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--evaluation-run-id",
        type=str,
        default=None,
        help="Override evaluation run ID from config file",
    )
    parser.add_argument(
        "--task", type=str, default=None, help="Override specific task from config file"
    )
    parser.add_argument(
        "--max-steps",
        type=str,
        default=None,
        help="Override max steps from config file (integer or N_given format)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Override retrieval strategy from config file",
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Override model from config file"
    )
    parser.add_argument(
        "--evaluation-set",
        type=str,
        default=None,
        help="Override evaluation set path from config file",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=-1,
        help="Starting index for evaluation set slicing (0-indexed, inclusive). Default: -1 (use config value)",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=-1,
        help="Ending index for evaluation set slicing (0-indexed, exclusive). Default: -1 (use config value)",
    )
    parser.add_argument(
        "--diversity-strategy",
        type=str,
        default=None,
        help="Override diversity strategy from config file. Options: null, different_task_or_variation, different_task, same_task_different_variation, same_task_same_variation",
    )
    parser.add_argument(
        "--rank-retrieve",
        type=int,
        default=None,
        help="Override rank_retrieve from config file. Selects which trajectory to use from top_k results (1-indexed, so 1=best, 2=second-best, etc.). Must be <= top_k.",
    )
    parser.add_argument(
        "--environment-name",
        type=str,
        default=None,
        help="Override environment name from config file. Options: alfworld, webarena",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Override output directory from config file. Default: environments/{environment_name}/traj_logs",
    )

    args = parser.parse_args()

    # Load config (using same function as run_evaluation.py for consistency)
    from .run_evaluation import load_config

    config, environment_specific_config = load_config(
        args.config, environment_name_override=args.environment_name
    )

    # Override config values with command line arguments if provided
    overrides_applied = []
    if args.evaluation_run_id is not None:
        config["evaluation_run_id"] = args.evaluation_run_id
        overrides_applied.append(f"evaluation_run_id: {args.evaluation_run_id}")
    if args.task is not None:
        config["task"] = args.task
        overrides_applied.append(f"task: {args.task}")
    if args.max_steps is not None:
        # Handle both integer and N_given formats
        try:
            config["max_steps"] = int(args.max_steps)
        except ValueError:
            config["max_steps"] = args.max_steps
        overrides_applied.append(f"max_steps: {args.max_steps}")
    if args.strategy is not None:
        if "retrieval" not in config:
            config["retrieval"] = {}
        config["retrieval"]["frequency_strategy"] = args.strategy
        overrides_applied.append(f"retrieval.frequency_strategy: {args.strategy}")
    if args.model is not None:
        config["model"] = args.model
        overrides_applied.append(f"model: {args.model}")
    if args.evaluation_set is not None:
        config["evaluation_set"] = args.evaluation_set
        overrides_applied.append(f"evaluation_set: {args.evaluation_set}")
    if args.diversity_strategy is not None:
        # Handle "null" string as None
        diversity_value = (
            None
            if args.diversity_strategy.lower() == "null"
            else args.diversity_strategy
        )

        # Ensure search_config exists in root config (for environments that use it directly)
        if "search_config" not in config:
            config["search_config"] = {}
        if "candidate_generation" not in config["search_config"]:
            config["search_config"]["candidate_generation"] = {}
        if "filters" not in config["search_config"]["candidate_generation"]:
            config["search_config"]["candidate_generation"]["filters"] = {}

        # Set diversity strategy in root search_config
        config["search_config"]["candidate_generation"]["filters"][
            "diversity_strategy"
        ] = diversity_value

        # Also ensure it's in retrieval config (for WebArena runner which uses retrieval_config.get('search_config'))
        if "retrieval" not in config:
            config["retrieval"] = {}
        config["retrieval"]["search_config"] = config["search_config"]

        overrides_applied.append(
            f"search_config.candidate_generation.filters.diversity_strategy: {args.diversity_strategy}"
        )
    if args.rank_retrieve is not None:
        # Ensure search_config exists in root config (for environments that use it directly)
        if "search_config" not in config:
            config["search_config"] = {}
        if "candidate_generation" not in config["search_config"]:
            config["search_config"]["candidate_generation"] = {}

        # Set rank_retrieve in root search_config
        config["search_config"]["candidate_generation"][
            "rank_retrieve"
        ] = args.rank_retrieve

        # Also ensure it's in retrieval config (for WebArena runner which uses retrieval_config.get('search_config'))
        if "retrieval" not in config:
            config["retrieval"] = {}
        config["retrieval"]["search_config"] = config["search_config"]

        overrides_applied.append(
            f"search_config.candidate_generation.rank_retrieve: {args.rank_retrieve}"
        )
    if args.environment_name is not None:
        config["environment_name"] = args.environment_name
        overrides_applied.append(f"environment_name: {args.environment_name}")
    if args.out_dir is not None:
        config["out_dir"] = args.out_dir
        overrides_applied.append(f"out_dir: {args.out_dir}")
    if args.no_retrieval_runs_file is not None:
        if "retrieval" not in config:
            config["retrieval"] = {}
        config["retrieval"]["no_retrieval_runs_file"] = args.no_retrieval_runs_file
        overrides_applied.append(
            f"retrieval.no_retrieval_runs_file: {args.no_retrieval_runs_file}"
        )

    # Only override start_idx/end_idx if explicitly provided on command line (not default -1)
    # Priority: command-line args > config file values
    # Use -1 as sentinel to mean "use config value"
    if args.start_idx != -1:
        config["start_idx"] = args.start_idx
        overrides_applied.append(f"start_idx: {args.start_idx}")
    if args.end_idx != -1:
        config["end_idx"] = args.end_idx
        overrides_applied.append(f"end_idx: {args.end_idx}")

    # Ensure start_idx and end_idx are always integers (never None)
    # Default: start_idx=0 (beginning), end_idx=-1 (means "to the end")
    start_idx_value = config.get("start_idx")
    end_idx_value = config.get("end_idx")

    # Convert None to proper integer defaults
    if start_idx_value is None:
        config["start_idx"] = 0
    if end_idx_value is None:
        config["end_idx"] = -1

    # Print override information
    if overrides_applied:
        print(f"🔧 Command-line overrides applied:")
        for override in overrides_applied:
            print(f"   - {override}")
    else:
        print(f"📋 Using all values from config file: {args.config}")

    # Ensure retrieval config exists (for backward compatibility)
    if "retrieval" not in config:
        config["retrieval"] = {
            "frequency_strategy": config.get("strategy", "none"),
            "retrieval_type": "trajectory",
        }

    print("=" * 80)
    print("🌐 WebArena Synchronous Evaluation Runner")
    print("=" * 80)
    print(f"✅ Loaded configuration from: {args.config}")
    print(f"📦 Environment: webarena")
    print(f"⚠️  Note: Episodes run SERIALLY due to Playwright sync API limitation")
    print("=" * 80 + "\n")

    # CRITICAL FIX: Ensure search_config is in retrieval dict
    # After load_config merges environment-specific config, search_config is at root level
    # but config["retrieval"] doesn't have it yet. We need to add it.
    if "search_config" in config and "search_config" not in config["retrieval"]:
        config["retrieval"]["search_config"] = config["search_config"]
        print(f"✅ Merged search_config into retrieval config")

    # Create runner (matching run_evaluation.py structure)
    runner = WebArenaRunner(
        environment_name="webarena",
        evaluation_set_path=config["evaluation_set"],
        indices_dir=config["indices_dir"],
        out_dir=config["out_dir"],
        model=config["model"],
        retrieval_config=config["retrieval"],  # Pass nested retrieval config
        specific_task=config.get("task"),
        max_steps_override=config.get("max_steps"),  # Can be int or "N_given"
        evaluation_run_id=config.get("evaluation_run_id", "random"),
        frequency_strategy=config["retrieval"]["frequency_strategy"],
        retrieval_type=config["retrieval"].get("retrieval_type", "trajectory"),
        history_window=config.get("history_window", 10),
        experiment_type=config.get("experiment_type", "action_string_direct"),
        env_config=config.get("env_config", {}),
        remove_actions_after_use=config.get("remove_actions_after_use", []),
        ultimate_fallback_action=config.get(
            "ultimate_fallback_action", "none"
        ),  # WebArena default is "none"
        agent_call_policy=config.get("agent_call_policy", "normal_then_structured"),
        start_idx=config.get("start_idx"),  # Starting index for evaluation set slicing
        end_idx=config.get("end_idx"),  # Ending index for evaluation set slicing
        environment_specific_config=environment_specific_config,  # Complete environment config from YAML for logging
        global_python_cut_off_step=config.get(
            "global_python_cut_off_step"
        ),  # NEW: Global Python step cutoff
    )

    # Run evaluation
    try:
        results = runner.run_all_episodes()
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
