import json
import asyncio
import argparse
import time
import os
import sys
import uuid
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dotenv import load_dotenv

# Import core functionality
from .core.async_with_traj_rag import run_episode_async, AsyncRateLimiter
from .core.environment_factory import EnvironmentFactory
from .core.retrieval_strategy import RetrievalManager
from .core.alfworld_online_memory import (
    AlfworldOnlineMemoryWriter,
    online_memory_enabled_from_env,
)
from .core.simulation_loader import SimulationActionsLoader  # NEW: For simulate_till_tk
from .utils.metrics_util import (
    create_enhanced_metrics,
    write_detailed_debug_json,
    write_metrics_json,
    write_metrics_csv,
)
from openai import AsyncOpenAI

########## Uncomment this when running AlfWorld on Babel ##########
import shutil

# ==========================================
# 🚑 HPC Monkeypatch Fix v3 (Anti-Recursion)
# ==========================================
_REAL_RMTREE = shutil.rmtree
_REAL_RMDIR = os.rmdir
_REAL_UNLINK = os.unlink


def _hpc_safe_rmtree(path, retries=3):
    for i in range(retries):
        try:
            if os.path.isdir(path):
                for root, dirs, files in os.walk(path, topdown=False):
                    for name in files:
                        try:
                            _REAL_UNLINK(os.path.join(root, name))
                        except OSError:
                            pass
                    for name in dirs:
                        try:
                            _REAL_RMDIR(os.path.join(root, name))
                        except OSError:
                            pass
                _REAL_RMDIR(path)
            return
        except OSError:
            if i < retries - 1:
                time.sleep(0.1)  # HPC Latency Wait
            else:
                pass  # Give up silently


# 2. Patch shutil.rmtree
if not getattr(shutil, "_is_patched_for_hpc", False):

    def _patched_rmtree(path, ignore_errors=False, onerror=None):
        retries = 3
        for i in range(retries):
            try:
                _REAL_RMTREE(path, ignore_errors=ignore_errors, onerror=onerror)
                return
            except OSError:
                if i < retries - 1:
                    time.sleep(0.1)
                else:
                    _hpc_safe_rmtree(path)

    shutil.rmtree = _patched_rmtree
    shutil._is_patched_for_hpc = True
    print("✅ [Monkeypatch] shutil.rmtree patched.")

# 3. Patch os.rmdir
if not getattr(os, "_is_patched_for_hpc", False):
    # shutil.rmtree may call os.rmdir(path, dir_fd=...) internally. Only
    # fall back to recursive deletion for the simple path-only case.
    def _patched_rmdir(path, *args, **kwargs):
        try:
            _REAL_RMDIR(path, *args, **kwargs)
        except OSError as e:
            if (not args and not kwargs) and e.errno in [39, 66]:  # Directory not empty
                _hpc_safe_rmtree(path)
            else:
                raise e

    os.rmdir = _patched_rmdir
    os._is_patched_for_hpc = True
    print("✅ [Monkeypatch] os.rmdir patched.")
# ==========================================

########## Uncomment this when running AlfWorld on Babel ##########


def _resolve_runtime_path(
    raw_value: Any,
    repo_root: Path,
    shared_root: Optional[Path] = None,
    *,
    prefer_shared: bool = False,
) -> Any:
    """Resolve a config path at runtime without storing absolute paths in the repo."""
    if not isinstance(raw_value, str) or not raw_value:
        return raw_value

    expanded = os.path.expandvars(raw_value)
    path = Path(expanded)
    if path.is_absolute():
        return str(path)

    repo_candidate = (repo_root / path).resolve()
    shared_candidate = (shared_root / path).resolve() if shared_root else None

    candidates = []
    if prefer_shared and shared_candidate is not None:
        candidates.append(shared_candidate)
    candidates.append(repo_candidate)
    if not prefer_shared and shared_candidate is not None:
        candidates.append(shared_candidate)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    # For outputs or yet-to-be-created paths, fall back to a repo-root-resolved path.
    return str(repo_candidate)


class FlatEvaluationRunner:
    """
    Generic evaluation runner for multi-environment trajectory-based agents.

    Supports interactive text-based environments (ALFWorld, WebArena)
    through the EnvironmentHandler interface and centralized EnvironmentFactory.

    All episodes from all tasks are executed in a single flat pool with controlled
    concurrency for optimal resource utilization.

    The environment is configured via YAML and can be changed without code modifications.
    """

    def __init__(
        self,
        evaluation_set_path: str,
        retrieval_config: Dict[str, str],  # New: nested retrieval config
        evaluation_run_id: str = "random",
        max_concurrent: int = 10,
        max_steps=None,  # Can be int or "N_given" (e.g., "1_given", "2_given")
        model: str = "openai/gpt-oss-20b:free",
        specific_task: str = None,
        rps: float = 1 / 2.5,
        history_window: int = 10,
        indices_dir: str = "alfworld/indices",
        out_dir: str = "traj_logs",
        experiment_type: str = "action_id_mapping",
        remove_actions_after_use: List[str] = None,
        environment_name: str = "alfworld",
        ultimate_fallback_action: str = "look around",
        agent_call_policy: str = "normal_then_structured",
        env_config: Optional[Dict] = None,  # New: environment-specific configuration
        search_config: Optional[
            Dict
        ] = None,  # New: search configuration for LanceDB filters
        start_idx: int = 0,  # Starting index for evaluation set slicing (0 = beginning)
        end_idx: int = -1,  # Ending index for evaluation set slicing (-1 = to the end)
        environment_specific_config: Optional[
            Dict
        ] = None,  # New: complete environment config from YAML for logging
        global_python_cut_off_step: Optional[
            int
        ] = None,  # New: global hard limit on Python steps (overrides max_steps)
    ):
        self.evaluation_set_path = Path(evaluation_set_path)

        # Validate retrieval config structure
        if not isinstance(retrieval_config, dict):
            raise ValueError(
                f"retrieval_config must be a dictionary, got {type(retrieval_config).__name__}"
            )

        if "frequency_strategy" not in retrieval_config:
            raise ValueError("retrieval_config must contain 'frequency_strategy' key")

        self.retrieval_config = retrieval_config
        self.frequency_strategy = self.retrieval_config["frequency_strategy"]
        self.retrieval_type = self.retrieval_config.get("retrieval_type", "trajectory")
        self.agentic_until_step_k = self.retrieval_config.get("agentic_until_step_k")
        self.retrieve_once_step_k = self.retrieval_config.get("retrieve_once_step_k")

        self.evaluation_run_id = evaluation_run_id
        self.max_concurrent = max_concurrent
        self.max_steps_override = max_steps  # Store the override config
        self.model = model
        self.specific_task = specific_task
        self.rps = rps
        self.history_window = history_window
        self.indices_dir = indices_dir
        self.out_dir = out_dir
        self.experiment_type = experiment_type
        self.remove_actions_after_use = remove_actions_after_use or ["look around"]
        self.environment_name = environment_name.lower()
        self.ultimate_fallback_action = ultimate_fallback_action
        self.agent_call_policy = agent_call_policy
        self.env_config = env_config or {}  # Store environment-specific configuration
        self.search_config = search_config  # Store search configuration for LanceDB
        self.start_idx = start_idx  # Store start index for slicing
        self.end_idx = end_idx  # Store end index for slicing
        self.environment_specific_config = (
            environment_specific_config or {}
        )  # Store complete environment config from YAML for logging
        self.global_python_cut_off_step = (
            global_python_cut_off_step  # Store global Python step cutoff
        )

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
            # Block this strategy - only allowed in WebArena (run_webarena.py)
            raise ValueError(
                f"\n{'='*80}\n"
                f"❌ UNSUPPORTED STRATEGY FOR THIS RUNNER\n"
                f"{'='*80}\n"
                f"Frequency Strategy: pseudo_simulate_till_closest_tk\n"
                f"Environment: {self.environment_name}\n"
                f"\n"
                f"The 'pseudo_simulate_till_closest_tk' strategy is currently only supported for WebArena.\n"
                f"Please use 'python -m traj_retrieval.run_webarena --config ...' for WebArena tasks.\n"
                f"\n"
                f"Available strategies for run_evaluation.py:\n"
                f"  - none: No retrieval\n"
                f"  - at_first: Retrieve once at the start\n"
                f"  - at_every_step: Retrieve at every step\n"
                f"  - agentic_until_tk: Agentic retrieval before step k, then no retrieval\n"
                f"  - retrieve_once_at_tk: Force exactly one retrieval at step k\n"
                f"  - simulate_till_tk: Simulate till step k, then retrieve once\n"
                f"{'='*80}\n"
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

        # Load evaluation set data
        self.evaluation_set_data = self._load_evaluation_set()

        # Create environment handler to parse evaluation set
        env_handler = self._create_environment_handler(self.env_config)
        self.all_episodes = env_handler.parse_evaluation_set(self.evaluation_set_data)

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

        # Generate or use provided evaluation_run_id for the overall evaluation folder
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

        # Create evaluation run-specific directory structure (ONE main folder)
        self.base_output_dir = Path(self.out_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_run_dir = (
            self.base_output_dir / f"run_{self.actual_evaluation_run_id}"
        )
        self.evaluation_run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Created evaluation run directory: {self.evaluation_run_dir}")

        # Execution tracking
        self.results = {}
        self.start_time = None
        self.online_memory_writer = None
        self.online_memory_enabled = False
        self.online_memory_commit_policy = (
            os.environ.get("ONLINE_MEMORY_COMMIT_POLICY", "").strip().lower()
        )
        self.online_memory_refresh_before_episode = os.environ.get(
            "ONLINE_MEMORY_REFRESH_BEFORE_EPISODE", "1"
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
        self.online_memory_refresh_after_commit = os.environ.get(
            "ONLINE_MEMORY_REFRESH_AFTER_COMMIT", "1"
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
        self._online_memory_commit_lock = None

        # Get unique task names for summary
        unique_tasks = set(ep["task_name"] for ep in self.all_episodes)

        print(f"🎯 Flat Evaluation Runner initialized:")
        print(f"   - Environment: {self.environment_name}")
        if self.specific_task:
            print(f"   - Running specific task: {self.specific_task}")
        else:
            print(f"   - Running all tasks from evaluation set")
        print(f"   - Unique tasks to run: {len(unique_tasks)}")
        print(f"   - Total episodes: {len(self.all_episodes)}")
        print(f"   - Retrieval Config:")
        print(f"     - Frequency Strategy: {self.frequency_strategy}")
        print(f"     - Retrieval Type: {self.retrieval_type}")
        print(f"   - Experiment type: {experiment_type}")
        print(f"   - Max concurrent episodes: {max_concurrent}")
        if self.environment_name == "webarena":
            print(f"   - ⚠️  WARNING: WebArena requires synchronous execution!")
            print(
                f"   - Please use: python -m traj_retrieval.run_webarena --config ..."
            )
        if self.env_config:
            print(f"   - Environment Config:")
            for key, value in self.env_config.items():
                print(f"     - {key}: {value}")
        print(f"   - Episodes will be mixed across tasks for optimal load balancing")
        print(
            f"   - All task outputs will be organized under: {self.evaluation_run_dir}"
        )

    def _create_environment_handler(self, env_config: Optional[Dict] = None):
        """
        Create an environment handler using the central factory.

        Args:
            env_config: Optional environment-specific configuration dict

        Returns:
            EnvironmentHandler instance

        Raises:
            ValueError: If environment name is not supported
        """
        # Pass environment-specific config if provided
        if env_config:
            return EnvironmentFactory.create_handler(
                self.environment_name, **env_config
            )
        else:
            return EnvironmentFactory.create_handler(self.environment_name)

    def _load_evaluation_set(self):
        """Load evaluation set from JSON file. Returns raw data for handler to parse."""
        if not self.evaluation_set_path.exists():
            raise FileNotFoundError(
                f"Evaluation set not found: {self.evaluation_set_path}"
            )

        try:
            with self.evaluation_set_path.open("r", encoding="utf-8") as f:
                evaluation_set = json.load(f)

            print(f"✅ Loaded evaluation set: {self.evaluation_set_path}")
            if isinstance(evaluation_set, list):
                print(f"   - Format: Flat list with {len(evaluation_set)} episodes")
            elif isinstance(evaluation_set, dict):
                print(f"   - Format: Structured dict")

            return evaluation_set

        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Invalid evaluation set file: {e}")

    def _apply_max_steps_override(self):
        """
        Apply max_steps override based on configuration.

        Modes supported:
        - integer (e.g., 40): Override all episodes to use this fixed max_steps
        - "N_given" (e.g., "1_given", "2_given", "3_given"): Multiply max_steps from evaluation set by N
        """
        if self.max_steps_override is None:
            # Default: use max_steps from evaluation set as-is (equivalent to "1_given")
            print(
                f"[MaxSteps] Using max_steps from evaluation set (default: no override)"
            )
            return

        if isinstance(self.max_steps_override, int):
            # Integer mode: override all episodes with fixed max_steps
            print(
                f"[MaxSteps] Overriding all episodes with max_steps={self.max_steps_override} (integer mode)"
            )
            for episode in self.all_episodes:
                episode["max_steps"] = self.max_steps_override
            return

        if isinstance(self.max_steps_override, str):
            # Check for "N_given" format (multiplier mode)
            if self.max_steps_override.endswith("_given"):
                try:
                    multiplier_str = self.max_steps_override.replace("_given", "")
                    multiplier = float(multiplier_str)

                    print(
                        f"[MaxSteps] Multiplying max_steps from evaluation set by {multiplier} ('{self.max_steps_override}' mode)"
                    )
                    for episode in self.all_episodes:
                        original_max_steps = episode["max_steps"]
                        episode["max_steps"] = int(original_max_steps * multiplier)
                    return
                except ValueError:
                    raise ValueError(
                        f"Invalid max_steps multiplier format: {self.max_steps_override}. "
                        f"Expected format: 'N_given' where N is a number (e.g., '1_given', '2_given', '3_given')"
                    )

        raise ValueError(
            f"Invalid max_steps value: {self.max_steps_override}. "
            f"Must be an integer or 'N_given' (e.g., '1_given', '2_given', '3_given')"
        )

    def _apply_evaluation_set_slicing(self):
        """
        Apply evaluation set slicing based on start_idx and end_idx.

        - start_idx: starting index (0-indexed, inclusive). Default: 0 (beginning)
        - end_idx: ending index (0-indexed, exclusive). Use -1 to mean "to the end"
        - Both use Python slicing semantics (0-indexed, end exclusive)
        """
        original_count = len(self.all_episodes)

        # Convert -1 to None for Python slicing (means "to the end")
        start = self.start_idx if self.start_idx is not None else 0
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

    async def _initialize_shared_resources(self):
        """Initialize shared resources that will be used by all episodes."""
        # Load environment variables
        load_dotenv()
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")

        # Validate retrieval resources BEFORE creating manager (fail-fast)
        self._validate_retrieval_resources()

        # Create shared resources
        self.aclient = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=api_key
        )
        self.limiter = AsyncRateLimiter(rps=self.rps)
        self.requests_sem = asyncio.Semaphore(2)  # LLM concurrency within each episode
        self._online_memory_commit_lock = asyncio.Lock()

        # Initialize retrieval manager using new factory method (LanceDB only)
        self.retrieval_manager = RetrievalManager.create_from_config(
            frequency_strategy=self.frequency_strategy,
            retrieval_type=self.retrieval_type,
            indices_dir=self.indices_dir,
            search_config=self.search_config,  # Pass search config for filters (from environment config)
            table_name=self.environment_name,  # Pass environment name as table name for LanceDB
        )

        if self.frequency_strategy == "none":
            print(f"[INFO] Retrieval disabled (frequency_strategy='none')")

        self.online_memory_enabled = (
            self.environment_name == "alfworld" and online_memory_enabled_from_env()
        )
        if self.online_memory_enabled:
            self.online_memory_writer = AlfworldOnlineMemoryWriter(
                indices_dir=self.indices_dir,
                table_name=self.environment_name,
                run_number=1,
            )
            existing_rows = self.online_memory_writer.count_rows()
            if not self.online_memory_commit_policy:
                self.online_memory_commit_policy = "per_episode"
            print(f"[OnlineMemory] Enabled for {self.environment_name}")
            print(f"[OnlineMemory]   Commit policy: {self.online_memory_commit_policy}")
            print(
                f"[OnlineMemory]   Refresh before episode: {self.online_memory_refresh_before_episode}"
            )
            print(
                f"[OnlineMemory]   Refresh after commit: {self.online_memory_refresh_after_commit}"
            )
            print(f"[OnlineMemory]   Existing rows in table: {existing_rows:,}")

        print(f"✅ Initialized shared resources:")
        print(f"   - OpenAI client: {self.aclient.base_url}")
        print(f"   - Rate limiter: {self.rps} RPS")
        print(f"   - Requests semaphore: {self.requests_sem._value}")
        print(f"   - Retrieval manager: {self.retrieval_manager.get_retrieval_stats()}")
        if self.frequency_strategy == "agentic_until_tk":
            print(
                f"   - Agentic retrieval window end (exclusive): {self.agentic_until_step_k}"
            )
        elif self.frequency_strategy == "retrieve_once_at_tk":
            print(f"   - Forced one-time retrieval step: {self.retrieve_once_step_k}")

    async def _maybe_commit_online_memory(
        self,
        *,
        trajectory: Optional[Dict[str, Any]],
        episode_metadata: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Append a completed ALFWorld trajectory into the online memory index."""
        if not self.online_memory_enabled:
            return None
        if self.online_memory_writer is None:
            return None
        if self.online_memory_commit_policy != "per_episode":
            return None
        if not trajectory or not episode_metadata:
            return None

        source_model_label = (
            episode_metadata.get("consumer_model")
            or episode_metadata.get("model_name")
            or self.model
        )
        async with self._online_memory_commit_lock:
            commit_summary = await asyncio.to_thread(
                self.online_memory_writer.commit_runtime_trajectory,
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
            "[OnlineMemory] Committed episode trajectory: "
            f"source_model={source_model_label}, "
            f"new_entries={commit_summary['new_entries']}, "
            f"duplicates={commit_summary['skipped_duplicates']}"
        )
        return commit_summary

    def _get_task_run_dir(self, task_name: str) -> Path:
        """Get or create task-specific run directory within the evaluation run folder."""
        # Use task_name directly - don't parse from task_run_id since task names contain underscores
        task_run_dir = self.evaluation_run_dir / f"run_{task_name}"
        task_run_dir.mkdir(parents=True, exist_ok=True)
        return task_run_dir

    async def _save_task_results(self, results_by_task_run: Dict[str, List[Dict]]):
        """Save per-task-run results in organized folder structure similar to async_with_traj_rag.py."""
        for task_run_id, task_results in results_by_task_run.items():
            try:
                # Get task name from the first result (all results in this group have same task)
                task_name = task_results[0]["task_name"]

                # Get task-specific run directory using task_name directly
                task_run_dir = self._get_task_run_dir(task_name)

                # Prepare episodes data for this task run (similar to async_with_traj_rag.py format)
                episodes = []
                detailed_episodes = []

                for result in task_results:
                    if result["trajectory"]:
                        episode_metadata = dict(
                            result.get("episode_metadata", {}) or {}
                        )
                        # Transform trajectory to match expected format
                        traj = result["trajectory"]
                        action_seq = [
                            {
                                "action": s.get("action"),
                                "observation": s.get("observation"),
                                "reasoning": s.get("reasoning", ""),
                                "isCompleted": s.get("isCompleted", False),
                                "inventory": s.get("inventory", ""),
                                "reward": s.get("reward", 0),  # Include reward
                                "score": s.get("score", 0),
                                "url": s.get(
                                    "url", ""
                                ),  # URL for web environments (WebArena), empty string otherwise
                            }
                            for s in traj.get("steps", [])
                        ]

                        episodes.append(
                            {
                                "variationIdx": result["variation_idx"],
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

                        if result["detailed_debug"]:
                            detailed_episode = dict(result["detailed_debug"])
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

    async def run_single_episode(self, episode_descriptor: Dict[str, Any]) -> Dict:
        """
        Run a single episode with error handling and timing.

        Args:
            episode_descriptor: Episode descriptor from handler containing all necessary info

        Returns:
            Dict containing episode results and metadata
        """
        start_time = time.time()

        # Extract generic episode info
        episode_id = episode_descriptor["episode_id"]
        task_name = episode_descriptor["task_name"]
        variation_idx = episode_descriptor["variation_idx"]
        episode_max_steps = episode_descriptor["max_steps"]
        task_run_id = f"{self.actual_evaluation_run_id}_{task_name}"

        try:
            is_unlimited = episode_max_steps >= 10000
            max_steps_display = "unlimited" if is_unlimited else str(episode_max_steps)
            print(f"🚀 Starting episode: {episode_id} (max_steps: {max_steps_display})")

            if (
                self.online_memory_enabled
                and self.online_memory_refresh_before_episode
                and self.retrieval_manager is not None
            ):
                self.retrieval_manager.refresh()

            # NEW: For simulate_till_tk strategy, extract simulate_step_k from episode descriptor
            simulate_step_k = None
            if self.frequency_strategy == "simulate_till_tk":
                simulate_step_k = episode_descriptor.get("simulate_step_k")
                if simulate_step_k is None:
                    raise RuntimeError(
                        f"\n{'='*80}\n"
                        f"❌ MISSING simulate_step_k IN EPISODE\n"
                        f"{'='*80}\n"
                        f"Frequency Strategy: simulate_till_tk\n"
                        f"Episode: {episode_id}\n"
                        f"Task: {task_name}, Variation: {variation_idx}\n"
                        f"\n"
                        f"The simulate_till_tk strategy requires 'simulate_step_k' field in each episode.\n"
                        f"Please add 'simulate_step_k' to the episode descriptor in the evaluation set:\n"
                        f"  {self.evaluation_set_path}\n"
                        f"\n"
                        f"Example:\n"
                        f"  {{\n"
                        f'    "task_type": "{task_name}",\n'
                        f'    "variation_id": "{variation_idx}",\n'
                        f'    "simulate_step_k": 2,  <-- Add this field\n'
                        f"    ...\n"
                        f"  }}\n"
                        f"{'='*80}\n"
                    )
                print(f"[Episode Config] simulate_step_k: {simulate_step_k}")

            # Create environment handler for this episode with environment-specific config
            env_handler = self._create_environment_handler(self.env_config)

            # Set agent call policy on the handler
            env_handler._agent_call_policy = self.agent_call_policy

            # Initialize handler from episode descriptor
            env_handler.initialize_from_episode(episode_descriptor)

            # Preflight check: verify retrieval is possible if retrieval is enabled
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
                        execution_time = time.time() - start_time

                        skip_result = {
                            "task_name": task_name,
                            "variation_idx": variation_idx,
                            "episode_id": episode_id,
                            "task_run_id": task_run_id,
                            "max_steps": episode_max_steps,
                            "execution_time": execution_time,
                            "success": False,
                            "skipped": True,
                            "skip_reason": "no_retrieval_candidates",
                            "timestamp": datetime.now().isoformat(),
                            "trajectory": None,
                            "detailed_debug": {
                                "preflight_check": {
                                    "candidates_found": candidate_count,
                                    "filter_sql": filter_sql,
                                    "frequency_strategy": self.frequency_strategy,
                                    "retrieval_type": self.retrieval_type,
                                }
                            },
                            "final_score": 0,
                            "steps_completed": 0,
                            "episode_done": False,
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

                        return skip_result
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

            # Run asynchronously (standard path for all async-compatible environments)
            # Note: WebArena requires synchronous execution and uses run_webarena.py instead
            traj, detailed_debug = await run_episode_async(
                env_handler=env_handler,
                aclient=self.aclient,
                limiter=self.limiter,
                requests_sem=self.requests_sem,
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

            execution_time = time.time() - start_time

            result = {
                "task_name": task_name,
                "variation_idx": variation_idx,
                "episode_id": episode_id,
                "task_run_id": task_run_id,
                "max_steps": episode_max_steps,
                "execution_time": execution_time,
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "trajectory": traj,
                "detailed_debug": detailed_debug,
                "final_score": traj.get("final_score", 0),
                "steps_completed": len(traj.get("steps", [])),
                "episode_done": traj.get("done", False),
                "episode_metadata": dict(episode_descriptor.get("metadata", {}) or {}),
            }

            commit_summary = await self._maybe_commit_online_memory(
                trajectory=traj,
                episode_metadata=result["episode_metadata"],
            )
            if commit_summary is not None:
                result["online_memory_commit"] = commit_summary

            print(
                f"✅ Completed: {episode_id} in {execution_time:.1f}s (score: {result['final_score']}, steps: {result['steps_completed']})"
            )
            return result

        except Exception as e:
            execution_time = time.time() - start_time

            error_result = {
                "task_name": task_name,
                "variation_idx": variation_idx,
                "episode_id": episode_id,
                "task_run_id": task_run_id,
                "max_steps": episode_max_steps,
                "execution_time": execution_time,
                "success": False,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "trajectory": None,
                "detailed_debug": None,
                "final_score": 0,
                "steps_completed": 0,
                "episode_done": False,
            }

            print(f"❌ Failed: {episode_id} after {execution_time:.1f}s - {e}")
            return error_result

    async def run_all_episodes(self) -> Dict:
        """
        Run all episodes in the evaluation set using flat parallelization.

        Returns:
            Dict containing all episode results and summary statistics
        """
        self.start_time = time.time()

        # Get unique tasks
        self.unique_tasks = set(ep["task_name"] for ep in self.all_episodes)

        print(f"\n🎬 Starting flat evaluation with {len(self.all_episodes)} episodes...")
        print(f"   Frequency Strategy: {self.frequency_strategy}")
        print(f"   Max concurrent episodes: {self.max_concurrent}")
        print(f"   Episodes mixed across {len(self.unique_tasks)} tasks")
        print("=" * 80)

        # Initialize shared resources
        await self._initialize_shared_resources()

        # Create global semaphore for episode-level concurrency control
        episode_semaphore = asyncio.Semaphore(self.max_concurrent)

        async def run_episode_with_semaphore(episode_descriptor: Dict[str, Any]):
            async with episode_semaphore:
                return await self.run_single_episode(episode_descriptor)

        # Shuffle episodes for better load balancing
        import random

        shuffled_episodes = self.all_episodes.copy()
        random.shuffle(shuffled_episodes)

        # Execute all episodes with controlled concurrency
        episode_tasks = [
            run_episode_with_semaphore(episode_descriptor)
            for episode_descriptor in shuffled_episodes
        ]

        print(
            f"🏃 Running {len(episode_tasks)} episodes with max {self.max_concurrent} concurrent..."
        )
        results = await asyncio.gather(*episode_tasks, return_exceptions=True)

        # Process results
        successful_results = []
        failed_results = []
        skipped_results = []

        for result in results:
            if isinstance(result, Exception):
                failed_results.append(
                    {
                        "task_name": "unknown",
                        "episode_id": "unknown",
                        "exception": str(result),
                        "success": False,
                        "skipped": False,
                    }
                )
            elif result.get("skipped", False):
                # Episode was skipped due to preflight check
                skipped_results.append(result)
            elif result.get("success", False):
                successful_results.append(result)
            else:
                failed_results.append(result)

        total_time = time.time() - self.start_time

        # Group results by task_run_id for organized saving
        results_by_task_run = {}
        for result in successful_results:
            task_run_id = result["task_run_id"]
            if task_run_id not in results_by_task_run:
                results_by_task_run[task_run_id] = []
            results_by_task_run[task_run_id].append(result)

        # Also group by task_name for summary statistics
        results_by_task = {}
        for result in successful_results:
            task_name = result["task_name"]
            if task_name not in results_by_task:
                results_by_task[task_name] = []
            results_by_task[task_name].append(result)

        # Save per-task-run results in organized folder structure
        await self._save_task_results(results_by_task_run)

        # Create task-level summary statistics (no detailed data)
        task_summaries = {}
        for task_name, task_results in results_by_task.items():
            total_episodes = len(task_results)
            completed_episodes = sum(1 for r in task_results if r["episode_done"])
            avg_score = (
                sum(r["final_score"] for r in task_results) / total_episodes
                if total_episodes > 0
                else 0
            )
            avg_steps = (
                sum(r["steps_completed"] for r in task_results) / total_episodes
                if total_episodes > 0
                else 0
            )
            avg_execution_time = (
                sum(r["execution_time"] for r in task_results) / total_episodes
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
                "variations_tested": sorted(
                    list(set(r["variation_idx"] for r in task_results))
                ),
            }

        # Create summary with only metadata and high-level statistics
        # Extract search config for easy access to top_k and rank_retrieve
        # Build retrieval_config with search_config nested inside (consistent with run_webarena.py)
        retrieval_config_with_search = dict(self.retrieval_config)
        if self.search_config:
            retrieval_config_with_search["search_config"] = self.search_config

        candidate_config = {}
        if self.search_config:
            candidate_config = self.search_config.get("candidate_generation", {})

        summary = {
            "evaluation_metadata": {
                "evaluation_set_path": str(self.evaluation_set_path),
                "retrieval_config": retrieval_config_with_search,  # Nested config with search_config inside
                "frequency_strategy": self.frequency_strategy,  # For convenience
                "retrieval_type": self.retrieval_type,
                "top_k": candidate_config.get("top_k", 100),  # For easy access
                "rank_retrieve": candidate_config.get(
                    "rank_retrieve", 1
                ),  # For easy access
                "evaluation_run_id": self.actual_evaluation_run_id,
                "total_episodes": len(self.all_episodes),
                "successful_episodes": len(successful_results),
                "failed_episodes": len(failed_results),
                "skipped_episodes": len(skipped_results),
                "success_rate": len(successful_results) / len(self.all_episodes)
                if self.all_episodes
                else 0,
                "total_execution_time": round(total_time, 2),
                "avg_time_per_episode": round(total_time / len(self.all_episodes), 2)
                if self.all_episodes
                else 0,
                "max_concurrent": self.max_concurrent,
                "model": self.model,
                "max_steps_mode": self.max_steps_override
                if self.max_steps_override is not None
                else "default",
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": datetime.now().isoformat(),
                "tasks_run": len(self.unique_tasks),
                "execution_mode": "flat_parallelization",
                "environment_config": self.environment_specific_config,  # Complete environment-specific config from YAML
            },
            "task_summaries": task_summaries,
            "failed_episodes_summary": [
                {
                    "task_name": f.get("task_name", "unknown"),
                    "episode_id": f.get("episode_id", "unknown"),
                    "error": f.get("error", f.get("exception", "Unknown error"))[:200]
                    + "..."
                    if len(f.get("error", f.get("exception", ""))) > 200
                    else f.get("error", f.get("exception", "Unknown error")),
                }
                for f in failed_results[:10]  # Limit to first 10 failures
            ],
            "skipped_episodes_summary": [
                {
                    "task_name": s.get("task_name", "unknown"),
                    "episode_id": s.get("episode_id", "unknown"),
                    "variation_idx": s.get("variation_idx", "unknown"),
                    "skip_reason": s.get("skip_reason", "unknown"),
                    "preflight_details": s.get("detailed_debug", {}).get(
                        "preflight_check", {}
                    ),
                }
                for s in skipped_results
            ],
            "evaluation_set_info": {
                "total_episodes": len(self.all_episodes),
                "unique_tasks": len(set(ep["task_name"] for ep in self.all_episodes)),
                "environment": self.environment_name,
            },
        }

        return summary

    def save_results(self, results: Dict, output_filename: str) -> bool:
        """Save evaluation results to JSON file inside the evaluation run directory."""
        output_path = self.evaluation_run_dir / output_filename

        try:
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            print(f"💾 Results saved to: {output_path}")
            print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
            return True

        except Exception as e:
            print(f"❌ Failed to save results: {e}")
            return False

    def print_summary(self, results: Dict):
        """Print evaluation summary."""
        print(f"[DEBUG] Results keys: {list(results.keys())}")
        metadata = results["evaluation_metadata"]

        print(f"\n📊 Flat Evaluation Summary")
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
        print(f"Max concurrent: {metadata['max_concurrent']}")
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
                print(f"     Variations tested: {task_summary['variations_tested']}")

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


def load_config(
    config_path: str = "run_evaluation_config.yaml",
    environment_name_override: Optional[str] = None,
) -> Tuple[Dict, Dict]:
    """
    Load configuration from YAML file and merge environment-specific settings.

    The config file has two sections:
    1. Generic settings (root level)
    2. Environment-specific settings (under 'environments' key)

    This function merges the appropriate environment settings into the root config.

    Args:
        config_path: Path to the YAML configuration file
        environment_name_override: Optional environment name to override the one in config file.
                                   This should be provided when command-line --environment-name is used.

    Returns:
        Tuple of (merged_config, environment_specific_config):
            - merged_config: Root config merged with environment-specific settings
            - environment_specific_config: Original environment-specific settings (for logging)
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    repo_root = Path(
        os.environ.get("REPO_ROOT", config_file.resolve().parent.parent)
    ).resolve()
    shared_root_env = os.environ.get("TRAJRET_SHARED_ROOT")
    shared_root = Path(shared_root_env).resolve() if shared_root_env else None

    try:
        with config_file.open("r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        print(f"✅ Loaded configuration from: {config_path}")

        # Get environment name (use override if provided, otherwise use config value)
        environment_name = environment_name_override or raw_config.get(
            "environment_name", "alfworld"
        )

        if environment_name_override:
            print(
                f"🔧 Using command-line environment override: {environment_name_override}"
            )

        # Get environment-specific settings
        environments = raw_config.get("environments", {})
        env_config = environments.get(environment_name, {})

        if not env_config:
            print(
                f"⚠️  Warning: No environment-specific config found for '{environment_name}'"
            )
            print(f"   Available environments: {list(environments.keys())}")

        # Merge environment-specific settings into root config
        # Environment-specific settings take precedence
        merged_config = raw_config.copy()
        merged_config.update(env_config)

        # Set the environment_name in merged config (use override if provided)
        merged_config["environment_name"] = environment_name

        # Remove the 'environments' key from merged config (not needed anymore)
        merged_config.pop("environments", None)

        # Resolve path-like settings at runtime. The repo keeps relative paths,
        # while shared resources can be provided via TRAJRET_SHARED_ROOT.
        if "evaluation_set" in merged_config:
            merged_config["evaluation_set"] = _resolve_runtime_path(
                merged_config["evaluation_set"], repo_root, shared_root
            )
        if "indices_dir" in merged_config:
            merged_config["indices_dir"] = _resolve_runtime_path(
                merged_config["indices_dir"], repo_root, shared_root, prefer_shared=True
            )
        if "out_dir" in merged_config:
            merged_config["out_dir"] = _resolve_runtime_path(
                merged_config["out_dir"], repo_root, shared_root
            )

        env_runtime_cfg = merged_config.get("env_config")
        if isinstance(env_runtime_cfg, dict) and env_runtime_cfg.get("config_path"):
            env_runtime_cfg["config_path"] = _resolve_runtime_path(
                env_runtime_cfg["config_path"],
                repo_root,
                shared_root,
                prefer_shared=True,
            )

        retrieval_cfg = merged_config.get("retrieval")
        if isinstance(retrieval_cfg, dict) and retrieval_cfg.get(
            "no_retrieval_runs_file"
        ):
            retrieval_cfg["no_retrieval_runs_file"] = _resolve_runtime_path(
                retrieval_cfg["no_retrieval_runs_file"], repo_root, shared_root
            )

        search_cfg = merged_config.get("search_config")
        if isinstance(search_cfg, dict):
            reranker_cfg = search_cfg.get("reranker")
            if isinstance(reranker_cfg, dict):
                for key, prefer_shared in [
                    ("feature_tsv", False),
                    ("ltr_model_file", False),
                    ("tfidf_base_dir", False),
                    ("model_base_dir", False),
                    ("model_base_dir_llm_features", False),
                    ("debug_log_path", False),
                ]:
                    if reranker_cfg.get(key):
                        reranker_cfg[key] = _resolve_runtime_path(
                            reranker_cfg[key],
                            repo_root,
                            shared_root,
                            prefer_shared=prefer_shared,
                        )

        # Log which environment config was loaded
        if env_config:
            print(f"📦 Loaded {environment_name} environment settings:")
            for key in env_config.keys():
                value = env_config[key]
                # Truncate long paths for display
                if isinstance(value, str) and len(value) > 60:
                    display_value = value[:57] + "..."
                else:
                    display_value = value
                print(f"   - {key}: {display_value}")

        # Return both merged config and original environment-specific config
        return merged_config, env_config

    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML configuration file: {e}")


async def main():
    parser = argparse.ArgumentParser(
        description="Run multi-environment evaluation experiments with YAML configuration"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="traj_retrieval/run_evaluation_config.yaml",
        help="Path to YAML configuration file (default: run_evaluation_config.yaml)",
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

    # Load configuration from YAML (pass environment_name override if provided)
    config, environment_specific_config = load_config(
        args.config, environment_name_override=args.environment_name
    )

    # Override config values with command line arguments if provided
    overrides_applied = []

    # Note: environment_name override was already applied during config loading
    if args.environment_name is not None:
        overrides_applied.append(
            f"environment_name: {args.environment_name} (applied during config load)"
        )

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
        overrides_applied.append(
            f"search_config.candidate_generation.rank_retrieve: {args.rank_retrieve}"
        )
    if args.out_dir is not None:
        config["out_dir"] = args.out_dir
        overrides_applied.append(f"out_dir: {args.out_dir}")
    # Note: environment_name override is already applied during config loading
    # (so that the correct environment-specific settings are loaded)
    # We don't need to apply it again here

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

    # Validate retrieval config exists
    if "retrieval" not in config:
        raise ValueError(
            "Configuration must contain 'retrieval' section with 'frequency_strategy' and 'retrieval_type'"
        )

    if "frequency_strategy" not in config["retrieval"]:
        raise ValueError("retrieval.frequency_strategy is required in configuration")

    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frequency_strategy = config["retrieval"]["frequency_strategy"]
    output_filename = f"evaluation_results_{frequency_strategy}_{timestamp}.json"

    # Show environment name from config
    env_name = config.get("environment_name", "Multi-Environment").title()
    print(f"🧪 {env_name} Evaluation Runner")
    print("=" * 60)

    try:
        # Create and run evaluation using config values
        # Note: environment-specific configs (task, evaluation_set, indices_dir, etc.)
        # have been merged into root config by load_config()
        runner = FlatEvaluationRunner(
            evaluation_set_path=config[
                "evaluation_set"
            ],  # From environment-specific config
            retrieval_config=config["retrieval"],  # New nested retrieval config
            evaluation_run_id=config["evaluation_run_id"],
            max_concurrent=config["max_concurrent"],
            max_steps=config.get(
                "max_steps"
            ),  # Can be integer or "N_given" (e.g., "1_given", "2_given")
            model=config["model"],
            specific_task=config.get(
                "task"
            ),  # From environment-specific config (optional)
            rps=config["rps"],
            history_window=config["history_window"],
            indices_dir=config["indices_dir"],  # From environment-specific config
            out_dir=config["out_dir"],
            experiment_type=config.get("experiment_type", "action_id_mapping"),
            remove_actions_after_use=config.get(
                "remove_actions_after_use", ["look around"]
            ),  # From environment-specific config
            environment_name=config.get("environment_name", "alfworld"),
            ultimate_fallback_action=config.get(
                "ultimate_fallback_action", "look around"
            ),  # From environment-specific config
            agent_call_policy=config.get(
                "agent_call_policy", "normal_then_structured"
            ),  # From environment-specific config
            env_config=config.get(
                "env_config"
            ),  # Environment-specific configuration
            search_config=config.get(
                "search_config"
            ),  # NEW: search configuration for LanceDB filters (from environment config)
            start_idx=config.get(
                "start_idx"
            ),  # Starting index for evaluation set slicing
            end_idx=config.get("end_idx"),  # Ending index for evaluation set slicing
            environment_specific_config=environment_specific_config,  # Complete environment config from YAML for logging
            global_python_cut_off_step=config.get(
                "global_python_cut_off_step"
            ),  # NEW: Global Python step cutoff
        )

        # Run all episodes
        results = await runner.run_all_episodes()

        # Save results first (before printing summary in case it fails)
        results_saved = runner.save_results(results, output_filename)

        # Print summary
        runner.print_summary(results)

        # Check if results were saved
        if results_saved:
            print(f"\n🎉 Flat evaluation completed successfully!")
            print(f"   Results saved to: {runner.evaluation_run_dir / output_filename}")
            print(f"   Individual episode results are organized in task folders")
            return 0
        else:
            print(f"\n⚠️  Evaluation completed but failed to save results")
            return 1

    except Exception as e:
        print(f"\n💥 Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
