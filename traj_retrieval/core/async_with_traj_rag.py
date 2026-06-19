# traj_retrieval/core/async_with_traj_rag.py
# Core functionality for trajectory retrieval

import os
import sys

# Set environment variables BEFORE any other imports to prevent PyTorch threading issues
# These are primarily for CPU stability (especially on macOS) and won't hurt GPU performance
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Only set restrictive threading limits on macOS/CPU to avoid deadlocks
# On GPU systems, these limits can reduce performance, so we use more permissive defaults
if sys.platform == "darwin":  # macOS
    # macOS needs strict limits to avoid threading deadlocks with MPS/CPU
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
else:
    # On Linux/Windows with GPU, allow more threads for better CPU fallback performance
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")

import json
import random
import asyncio
from datetime import datetime
from collections import deque
from typing import Awaitable, Callable, Tuple, TypeVar, List, Optional, Deque, Dict, Any
from openai import AsyncOpenAI

# Import environment handlers
from .base_handler import EnvironmentHandler, TrajectoryNode, StepResult
from .environment_factory import EnvironmentFactory
from .retrieval_strategy import RetrievalManager, RetrievalResult
from .retrieval_handler import RetrievalPlanner, RetrievalHandler

# Import utilities from new locations
from ..utils.logging_util import (
    analyze_non_admissible_cause,
    create_step_debug_info,
    update_step_debug_with_retrieval,
    update_step_debug_with_results,
    create_detailed_debug_structure,
    create_llm_debug_info,
    create_planner_debug_info,
    write_llm_debug_file,
)
from ..strategies.factory import get_experiment_strategy

# -----------------------------
# Async rate limiter + backoff
# -----------------------------
T = TypeVar("T")


class AsyncRateLimiter:
    """Simple per-process limiter: at most `rps` requests per second."""

    def __init__(self, rps: float = 1 / 3.0):
        # e.g., rps=1/3 -> 1 request every 3 seconds
        self.min_interval = 1.0 / max(rps, 1e-9)
        self._lock = asyncio.Lock()
        self._next_ok = 0.0

    async def wait(self):
        async with self._lock:
            now = asyncio.get_running_loop().time()
            if now < self._next_ok:
                await asyncio.sleep(self._next_ok - now)
            self._next_ok = asyncio.get_running_loop().time() + self.min_interval

    async def __aenter__(self):
        await self.wait()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


async def retry_with_backoff_async(
    func: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 6,
    base_delay: float = 1.5,
    max_delay: float = 30.0,
    jitter=(0.2, 0.6),
    context: str = "LLM call",
) -> T:
    """Retry an awaitable on 429/transient errors with exponential backoff."""
    attempt = 0
    while True:
        try:
            if attempt > 0:
                print(
                    f"[RETRY] {context} attempt {attempt + 1}/{max_retries + 1}",
                    flush=True,
                )
            return await func()
        except Exception as e:
            status = getattr(e, "status_code", None)
            msg = str(e).lower()
            is_429 = (status == 429) or ("rate limit" in msg) or ("429" in msg)
            is_transient = any(
                s in msg
                for s in (
                    "timeout",
                    "temporar",
                    "reset by peer",
                    "unavailable",
                    "connection",
                )
            )
            if not (is_429 or is_transient) or attempt >= max_retries:
                if attempt >= max_retries:
                    print(
                        f"[RETRY] {context} failed after {attempt + 1} attempts: {e}",
                        flush=True,
                    )
                else:
                    print(
                        f"[RETRY] {context} failed with non-retryable error: {e}",
                        flush=True,
                    )
                raise
            retry_after = None
            resp = getattr(e, "response", None)
            if resp is not None:
                try:
                    retry_after = float(resp.headers.get("Retry-After", ""))
                except Exception:
                    retry_after = None
            if retry_after is None:
                delay = min(base_delay * (2**attempt), max_delay)
                delay *= 1.0 + random.uniform(*jitter)
            else:
                delay = retry_after
            print(
                f"[BACKOFF] {context} retrying in {delay:.2f}s due to: {e}", flush=True
            )
            await asyncio.sleep(delay)
            attempt += 1


# -----------------------------
# Note: TrajectoryRetriever has been replaced by retrieval_strategy.py
# All retrieval logic is now handled by RetrievalManager and RetrievalStrategy classes
# RetrievalPlanner and RetrievalHandler are now in retrieval_handler.py
# -----------------------------


# Note: These helper functions have been replaced by RetrievalManager
# - load_trajectory_retrieval_results() -> retrieval_manager.retrieve()
# - format_trajectory_context() -> env_handler.format_retrieval_result()
# - get_trajectory_length() -> retrieval_result.metadata['trajectory_length']


def compact_history_str(
    history: List[Dict], env_handler: EnvironmentHandler, max_chars: int = 1500
) -> str:
    """
    Build a compact, model-readable history string from recent steps.
    Delegates formatting to the environment handler for environment-specific customization.
    """
    parts = []
    for h in history:
        step_num = h.get("step", "?")
        obs = h.get("observation", "")
        action = h.get("action", "")
        reward = h.get("reward", 0)
        reasoning = h.get("reasoning", "")
        inventory = h.get("inventory", "")
        url = h.get("url", "")

        # Delegate formatting to handler
        entry = env_handler.format_history_entry(
            step=step_num,
            observation=obs,
            action=action,
            reward=reward,
            reasoning=reasoning,
            inventory=inventory,
            url=url,
        )

        parts.append(entry)

    s = "\n".join(parts)
    return s


# -----------------------------
# LLM call (with history + trajectory RAG)
# -----------------------------
async def llm_choose_action_async(
    aclient: AsyncOpenAI,
    limiter: AsyncRateLimiter,
    requests_sem: asyncio.Semaphore,
    model: str,
    goal_text: str,
    observation: str,
    inventory: str,
    admissible_actions: List[str],
    recent_history_str: str,
    trajectory_context: str = "",
    experiment_type: str = "action_id_mapping",
    env_handler=None,
    ultimate_fallback_action: str = "look around",
    episode_id: str = "unknown",
    current_step: int = 0,
    max_steps: int = 0,
) -> Tuple[str, str, Dict]:  # Now returns action, reasoning, debug_info
    """Ask the model for a short plan and one valid action (JSON), with memory context and trajectory RAG."""

    # Get the experiment strategy (async version)
    try:
        strategy = get_experiment_strategy(experiment_type, use_sync=False)
    except ValueError as e:
        print(
            f"[WARNING] Invalid experiment type '{experiment_type}': {e}. Falling back to 'action_string_direct'",
            flush=True,
        )
        strategy = get_experiment_strategy("action_string_direct", use_sync=False)

    # Prepare user payload for debug info
    user_payload = {
        "goal": goal_text,
        "current_observation": observation,
        "inventory": inventory,
        "admissible_actions": admissible_actions,
        "recent_history": recent_history_str,
    }

    # Get current URL from handler if available (for web environments like WebArena)
    current_url = env_handler.get_current_url() if env_handler else ""

    # Prepare simplified debug info using utility
    debug_info = create_llm_debug_info(
        user_payload, trajectory_context, model, url=current_url
    )

    # Delegate to strategy's execute method
    try:
        action, strategy_debug = await strategy.execute(
            aclient=aclient,
            limiter=limiter,
            requests_sem=requests_sem,
            model=model,
            goal_text=goal_text,
            observation=observation,
            inventory=inventory,
            admissible_actions=admissible_actions,
            recent_history_str=recent_history_str,
            trajectory_context=trajectory_context,
            debug_info=debug_info,
            env_handler=env_handler,
            ultimate_fallback_action=ultimate_fallback_action,
            episode_id=episode_id,
            current_step=current_step,
            max_steps=max_steps,
        )

        # Extract reasoning from strategy debug info
        reasoning = strategy_debug.get("reasoning", "")

        # Log success or warning based on whether reasoning was found
        if reasoning.strip():
            print(
                f"[DEBUG] Reasoning extracted successfully - Length: {len(reasoning)} chars",
                flush=True,
            )
        else:
            print(
                f"[WARNING] No reasoning found in strategy debug info for {experiment_type}. Debug keys: {list(strategy_debug.keys())}",
                flush=True,
            )
            # Provide fallback reasoning
            reasoning = "Agent could not decide the action so falling back to look-around action"
            # Add warning to debug info
            strategy_debug.setdefault("warnings", []).append(
                {
                    "stage": "reasoning_extraction",
                    "message": "No reasoning found in strategy response",
                    "strategy": experiment_type,
                    "fallback_reasoning_provided": True,
                }
            )
            strategy_debug["reasoning"] = reasoning

        # Merge debug info
        debug_info.update(strategy_debug)

        return action, reasoning, debug_info
    except Exception as strategy_error:
        print(f"[ERROR] Strategy execution failed: {strategy_error}", flush=True)

        # Write detailed debug file for strategy errors
        step_context = {
            "goal": goal_text,
            "observation": observation,
            "inventory": inventory,
            "admissible_actions_count": len(admissible_actions),
            "recent_history_length": len(recent_history_str),
            "trajectory_context_length": len(trajectory_context),
            "strategy_name": strategy.get_strategy_name(),
        }

        try:
            debug_file_path = write_llm_debug_file(
                error=strategy_error,
                model=model,
                messages=[],  # Strategy handles its own messages
                response_text="STRATEGY_EXECUTION_ERROR",
                action_mapping=None,
                step_info=step_context,
            )
            if debug_file_path:
                print(
                    f"[DEBUG] Strategy error debug info saved to: {debug_file_path}",
                    flush=True,
                )
        except Exception as debug_error:
            print(
                f"[WARNING] Failed to write strategy error debug file: {debug_error}",
                flush=True,
            )

        # Use strategy fallback as ultimate fallback
        reasoning, action = strategy.get_fallback_action(
            admissible_actions=admissible_actions,
            ultimate_fallback_action=ultimate_fallback_action,
        )
        print(
            f"[DEBUG] Ultimate fallback action: '{action}' with reasoning: '{reasoning}'",
            flush=True,
        )

        debug_info["errors"] = debug_info.get("errors", [])
        debug_info["errors"].append(f"Strategy execution failed: {str(strategy_error)}")
        debug_info.update(
            {
                "parsed_result": {
                    "action": action,
                    "reasoning": reasoning,  # Use the reasoning from strategy fallback
                    "success": False,
                    "fallback_used": True,
                    "strategy_error": True,
                    "experiment_type": experiment_type,
                }
            }
        )
        return action, reasoning, debug_info  # Return reasoning from strategy fallback


# -----------------------------
# Episode runner (Generic - works with any EnvironmentHandler)
# -----------------------------
async def run_episode_async(
    env_handler: EnvironmentHandler,
    aclient: AsyncOpenAI,
    limiter: AsyncRateLimiter,
    requests_sem: asyncio.Semaphore,
    model_slug: str,
    retrieval_manager: RetrievalManager,
    frequency_strategy: str = "t0",
    history_window: int = 10,
    experiment_type: str = "action_id_mapping",
    remove_actions_after_use: List[str] = None,
    ultimate_fallback_action: str = "look around",
    simulation_loader=None,  # NEW: SimulationActionsLoader for simulate_till_tk
    simulate_step_k: int = None,  # NEW: Per-episode simulate_step_k (required for simulate_till_tk)
    agentic_until_step_k: int = None,
    retrieve_once_step_k: int = None,
    global_python_cut_off_step: int = None,  # NEW: Global hard limit on Python steps (overrides max_steps)
):
    """
    Generic episode runner that works with any EnvironmentHandler.

    The handler must be already initialized before calling this function.

    Args:
        env_handler: Environment handler (already initialized) implementing EnvironmentHandler interface
        aclient: AsyncOpenAI client for LLM calls
        limiter: Rate limiter for API calls
        requests_sem: Semaphore for concurrent request control
        model_slug: Model identifier for LLM
        retrieval_manager: RetrievalManager for memory-augmented retrieval
        frequency_strategy: When to retrieve ("none", "t0", "every_10", "agentic", "agentic_until_tk", "retrieve_once_at_tk", "simulate_till_tk")
        history_window: Number of previous steps to include in LLM history
        experiment_type: Type of experiment strategy to use
        remove_actions_after_use: Actions to prevent consecutive use
        ultimate_fallback_action: Action to use when all retry attempts fail
        simulation_loader: SimulationActionsLoader instance (required for simulate_till_tk)
        agentic_until_step_k: For agentic_until_tk, use agentic retrieval only while python_step < k
        retrieve_once_step_k: For retrieve_once_at_tk, force exactly one retrieval when python_step == k
    """
    # Handler should already be initialized by caller

    # Get initial state
    current_node = env_handler.reset()

    # Get metadata for logging
    metadata = env_handler.get_task_metadata()
    task_display = f"{metadata['task_name']}:{metadata['variation']}"
    is_unlimited_mode = metadata.get("is_unlimited", False)
    max_steps = metadata.get("max_steps", 50)

    print(f"\n{'='*100}", flush=True)
    print(f"🎯 EPISODE START: [{task_display}]", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"[{task_display}] Goal: {current_node.goal}", flush=True)
    print(
        f"[{task_display}] Max Steps: {'unlimited' if is_unlimited_mode else max_steps}",
        flush=True,
    )
    print(f"[{task_display}] Frequency Strategy: {frequency_strategy}", flush=True)
    print(
        f"[{task_display}] Example Initial Actions: {random.sample(current_node.admissible_actions, min(3, len(current_node.admissible_actions)))}",
        flush=True,
    )
    print(f"{'='*100}\n", flush=True)

    # Sliding window memory + loop-avoidance buffers
    history: Deque[Dict] = deque(maxlen=history_window)

    # Action removal tracking - prevent consecutive use
    if remove_actions_after_use is None:
        remove_actions_after_use = ["look around", "look"]
    last_action = None  # Track the last action to prevent consecutive use

    # NEW: Load simulation actions for simulate_till_tk strategy
    simulation_steps = None
    simulation_metadata = None
    if frequency_strategy == "simulate_till_tk":
        if simulation_loader is None:
            raise ValueError(
                "simulate_till_tk strategy requires simulation_loader parameter"
            )

        if simulate_step_k is None:
            raise ValueError(
                "simulate_till_tk strategy requires simulate_step_k parameter (should be in episode descriptor)"
            )

        # Get variation_id from metadata (handle both variation and variation_id keys)
        variation_id = metadata.get("variation_id") or metadata.get("variation")

        # Check if simulation data exists for this episode before trying to load it
        if not simulation_loader.has_data(
            task_name=metadata["task_name"], variation_id=variation_id
        ):
            # No simulation data available - return early with a skip result
            print(f"\n{'='*80}")
            print(f"⚠️  WARNING: NO SIMULATION DATA AVAILABLE")
            print(f"{'='*80}")
            print(f"Task: {metadata['task_name']}")
            print(f"Variation: {variation_id}")
            print(f"Strategy: simulate_till_tk")
            print(f"Simulation file: {simulation_loader.json_file_path}")
            print(
                f"\nThis episode will be SKIPPED because simulate_till_tk requires pre-recorded actions."
            )
            print(f"{'='*80}\n")

            # Return a "skipped" trajectory
            return {
                "goal_text": metadata.get("goal_text", ""),
                "steps": [],
                "done": False,
                "final_score": 0,
                "total_python_steps": 0,
                "skipped": True,
                "skip_reason": "no_simulation_data",
            }, {
                "skipped": True,
                "skip_reason": "no_simulation_data",
                "task_name": metadata["task_name"],
                "variation_id": variation_id,
            }

        # Load simulation steps (action + reasoning) (will raise RuntimeError if missing - fail-fast)
        simulation_steps = simulation_loader.get_steps(
            task_name=metadata["task_name"], variation_id=variation_id
        )

        print(
            f"[{task_display}] 🎬 SIMULATION MODE: Loaded {len(simulation_steps)} pre-recorded steps (action + reasoning)"
        )
        print(
            f"[{task_display}] 🎬 Will simulate steps 0-{simulate_step_k-1}, retrieve at step {simulate_step_k}, then use LLM from step {simulate_step_k+1}"
        )

        # Create simulation metadata for debug logging
        simulation_metadata = {
            "simulate_step_k": simulate_step_k,
            "simulation_file": str(simulation_loader.json_file_path),
            "total_simulation_steps_loaded": len(simulation_steps),
            "environment_name": simulation_loader.environment_name,
        }

    # Initialize retrieval handler (centralized retrieval logic)
    retrieval_handler = RetrievalHandler(
        frequency_strategy=frequency_strategy,
        retrieval_manager=retrieval_manager,
        aclient=aclient,
        limiter=limiter,
        requests_sem=requests_sem,
        model_slug=model_slug,
        simulation_loader=simulation_loader,  # NEW: Pass simulation_loader
        simulate_step_k=simulate_step_k,  # NEW: Pass per-episode simulate_step_k
        agentic_until_step_k=agentic_until_step_k,
        retrieve_once_step_k=retrieve_once_step_k,
    )

    # Trajectory context and tracking
    trajectory_context = ""
    retrieved_trajectory_lengths = []

    traj = {
        "task": metadata["task_name"],
        "variation": metadata["variation"],
        "max_variations": metadata["max_variations"],
        "steps": [],
    }

    # Extract ALFWorld-specific metadata if this is an ALFWorld environment
    alfworld_metadata = None
    if hasattr(env_handler, "episode_metadata") and env_handler.episode_metadata:
        # Extract ALFWorld-specific fields from episode metadata
        alfworld_metadata = {}
        if "game_file" in env_handler.episode_metadata:
            alfworld_metadata["game_file"] = env_handler.episode_metadata["game_file"]
        if "task_id" in env_handler.episode_metadata:
            alfworld_metadata["task_id"] = env_handler.episode_metadata["task_id"]
        if "floor_plan" in env_handler.episode_metadata:
            alfworld_metadata["floor_plan"] = env_handler.episode_metadata["floor_plan"]

    # Detailed debugging information using utility
    detailed_debug = create_detailed_debug_structure(
        metadata["task_name"],
        metadata["variation"],
        metadata["max_variations"],
        current_node.goal,
        frequency_strategy,
        simulation_metadata,
        alfworld_metadata,
    )

    episode_done = False
    last_score = None
    python_step = 0  # Python action counter (for logging only)

    while not episode_done:
        # Check if we've reached the internal step limit (skip check for unlimited mode)
        if not is_unlimited_mode and current_node.internal_step_count >= max_steps:
            print(f"\n{'='*100}", flush=True)
            print(f"⛔ EPISODE LIMIT REACHED: [{task_display}]", flush=True)
            print(
                f"[{task_display}] Internal step count: {current_node.internal_step_count}/{max_steps}",
                flush=True,
            )
            print(f"{'='*100}\n", flush=True)
            break

        # NEW: Check if we've reached the global Python step cutoff limit (hard limit)
        if (
            global_python_cut_off_step is not None
            and python_step >= global_python_cut_off_step
        ):
            print(f"\n{'='*100}", flush=True)
            print(f"🛑 GLOBAL PYTHON STEP CUTOFF REACHED: [{task_display}]", flush=True)
            print(
                f"[{task_display}] Python step count: {python_step}/{global_python_cut_off_step}",
                flush=True,
            )
            print(
                f"[{task_display}] This cutoff overrides max_steps limit ({max_steps})",
                flush=True,
            )
            print(f"{'='*100}\n", flush=True)
            break

        # Step header
        max_steps_display = "unlimited" if is_unlimited_mode else str(max_steps)
        print(f"\n{'▬'*100}", flush=True)
        print(
            f"🔄 STEP {python_step}: [{task_display}] (Internal: {current_node.internal_step_count}/{max_steps_display})",
            flush=True,
        )
        print(f"{'▬'*100}", flush=True)
        print(
            f"[{task_display}][Step {python_step}] 👁  Observation: '{current_node.observation}'",
            flush=True,
        )
        print(
            f"[{task_display}][Step {python_step}] 🎒 Inventory: '{current_node.inventory}'",
            flush=True,
        )

        # Get current admissible actions and handle consecutive action removal
        current_admissible = current_node.admissible_actions.copy()

        # Remove actions that would be consecutive (same as last action)
        if last_action and remove_actions_after_use:
            original_count = len(current_admissible)
            current_admissible = [
                a
                for a in current_admissible
                if not (a in remove_actions_after_use and a == last_action)
            ]
            if len(current_admissible) < original_count:
                removed_count = original_count - len(current_admissible)
                print(
                    f"[{task_display}][Step {python_step}] 🚫 Action Removal: Prevented consecutive use of '{last_action}' - removed {removed_count} instances",
                    flush=True,
                )

        print(
            f"[{task_display}][Step {python_step}] 🎯 Admissible Actions: {len(current_admissible)} total | Sample: {current_admissible[:3]}",
            flush=True,
        )

        # Build history (sliding window memory) string for the LLM
        recent_history_str = compact_history_str(list(history), env_handler)

        # Get current URL from handler (will be empty for most environments)
        current_url = env_handler.get_current_url()

        # Initialize simplified step info for debugging using utility
        step_debug_info = create_step_debug_info(
            python_step,
            current_node.observation,
            current_node.inventory,
            current_admissible,
            recent_history_str,
            trajectory_context,
            url=current_url,
        )

        # Unified retrieval handling - single call handles all frequency strategies
        retrieval_result = await retrieval_handler.retrieve_if_needed(
            python_step=python_step,
            current_node=current_node,
            metadata=metadata,
            recent_history_str=recent_history_str,
            trajectory_context=trajectory_context,
            task_display=task_display,
            step_debug_info=step_debug_info,
            env_handler=env_handler,
        )

        # Update trajectory context if retrieval was successful
        if retrieval_result:
            trajectory_context = retrieval_result.formatted_context
            traj_length = retrieval_result.metadata.get("trajectory_length", 0)
            retrieved_trajectory_lengths.append(traj_length)

        # NEW: Check if we should use simulation mode (simulate_till_tk strategy)
        use_simulation = False
        if simulation_steps is not None and simulate_step_k is not None:
            use_simulation = python_step < simulate_step_k and python_step < len(
                simulation_steps
            )

        if use_simulation:
            # SIMULATION MODE: Use pre-recorded action + reasoning from no-retrieval run
            sim_step = simulation_steps[python_step]
            action = sim_step["action"]
            reasoning = sim_step[
                "reasoning"
            ]  # Use actual reasoning from original trajectory
            step_debug_info["llm_call"] = {
                "mode": "simulation",
                "step": python_step,
                "simulate_step_k": simulate_step_k,
                "total_simulation_steps": len(simulation_steps),
                "note": "Using actual reasoning from original trajectory",
            }

            print(f"\n{'┈'*100}", flush=True)
            print(
                f"🎬 SIMULATION MODE: [{task_display}][Step {python_step}/{simulate_step_k-1}]",
                flush=True,
            )
            print(f"{'┈'*100}", flush=True)
            print(
                f"[{task_display}][Step {python_step}] 🎬 Using pre-recorded action: '{action}'",
                flush=True,
            )
            print(
                f"[{task_display}][Step {python_step}] 💭 Using original reasoning: '{reasoning[:80]}...'",
                flush=True,
            )
            print(
                f"[{task_display}][Step {python_step}] ℹ️  Will switch to LLM mode at step {simulate_step_k} (after retrieval)",
                flush=True,
            )
            print(f"{'┈'*100}\n", flush=True)

            # Log mode transition at the step before k
            if python_step == simulate_step_k - 1:
                print(
                    f"[{task_display}] ⚠️  NEXT STEP: Will perform retrieval and switch to LLM mode",
                    flush=True,
                )
        else:
            # LLM MODE: Ask the LLM (with memory + soft avoid list + trajectory RAG context)
            print(f"\n{'┈'*100}", flush=True)
            print(
                f"🧠 LLM ACTION SELECTION: [{task_display}][Step {python_step}]",
                flush=True,
            )
            if simulation_steps is not None and simulate_step_k is not None:
                if python_step == simulate_step_k:
                    print(
                        f"🎯 SWITCHED TO LLM MODE (after retrieval at step k={simulate_step_k})",
                        flush=True,
                    )
            print(f"{'┈'*100}", flush=True)
            try:
                action, reasoning, llm_debug = await llm_choose_action_async(
                    aclient,
                    limiter,
                    requests_sem,
                    model_slug,
                    current_node.goal,
                    current_node.observation,
                    current_node.inventory,
                    current_admissible,
                    recent_history_str=recent_history_str,
                    trajectory_context=trajectory_context,
                    experiment_type=experiment_type,
                    env_handler=env_handler,
                    ultimate_fallback_action=ultimate_fallback_action,
                    episode_id=task_display,
                    current_step=python_step,
                    max_steps=max_steps,
                )
                step_debug_info["llm_call"] = llm_debug
                # Enhanced reasoning output
                if reasoning and reasoning.strip():
                    print(
                        f"[{task_display}][Step {python_step}] 💭 Reasoning: {reasoning[:200]}{'...' if len(reasoning) > 200 else ''}",
                        flush=True,
                    )
                print(
                    f"[{task_display}][Step {python_step}] ✅ Selected Action: '{action}'",
                    flush=True,
                )
                print(f"{'┈'*100}\n", flush=True)
            except Exception as e:
                print(
                    f"[{task_display}][Step {python_step}] ❌ LLM Error: {type(e).__name__} - {str(e)[:100]}",
                    flush=True,
                )

                # Check if this is a JSON decode error (API-level issue) or other error
                error_type = type(e).__name__
                if "JSONDecodeError" in error_type:
                    print(
                        f"[{task_display}][Step {python_step}] ℹ️  API-level JSON error detected - debug captured in llm_choose_action_async",
                        flush=True,
                    )
                else:
                    # Only write debug file for non-JSON errors (to avoid duplicates)
                    step_context = {
                        "step": python_step,
                        "internal_step": current_node.internal_step_count,
                        "goal": current_node.goal,
                        "observation": current_node.observation,
                        "inventory": current_node.inventory,
                        "admissible_actions_count": len(current_admissible),
                        "recent_history_length": len(recent_history_str),
                        "trajectory_context_length": len(trajectory_context),
                    }

                    try:
                        debug_file_path = write_llm_debug_file(
                            error=e,
                            model=model_slug,
                            messages=[],  # We don't have access to messages here
                            response_text="EPISODE_LEVEL_ERROR",
                            action_mapping=None,
                            step_info=step_context,
                        )
                        if debug_file_path:
                            print(
                                f"[{task_display}][Step {python_step}] 📝 Debug file saved: {debug_file_path}",
                                flush=True,
                            )
                    except Exception as debug_error:
                        print(
                            f"[{task_display}][Step {python_step}] ⚠️  Failed to write debug file: {debug_error}",
                            flush=True,
                        )

                # Fallback to a safe action using configured ultimate_fallback_action
                print(f"\n{'╌'*100}", flush=True)
                print(
                    f"🔄 FALLBACK ACTION: [{task_display}][Step {python_step}]",
                    flush=True,
                )
                print(f"{'╌'*100}", flush=True)
                # Always use configured ultimate fallback action
                action = ultimate_fallback_action
                reasoning = f"Agent could not decide the action so falling back to {ultimate_fallback_action} action"

                # Log warning if fallback is not in admissible actions (for debugging)
                if (
                    current_admissible
                    and ultimate_fallback_action not in current_admissible
                ):
                    print(
                        f"[{task_display}][Step {python_step}] ⚠️  WARNING: Configured fallback '{ultimate_fallback_action}' not in admissible actions (count: {len(current_admissible)})",
                        flush=True,
                    )
                    print(
                        f"[{task_display}][Step {python_step}] Using it anyway - validation will happen in environment handler if needed",
                        flush=True,
                    )
                else:
                    print(
                        f"[{task_display}][Step {python_step}] Using configured fallback: '{action}'",
                        flush=True,
                    )
                step_debug_info["llm_call"] = {
                    "errors": [f"LLM call failed: {str(e)}"],
                    "warnings": [],
                    "parsed_result": {
                        "action": action,
                        "reasoning": reasoning,
                        "success": False,
                        "fallback_used": True,
                    },
                }
                print(
                    f"[{task_display}][Step {python_step}] 💭 Fallback Reasoning: {reasoning[:200]}{'...' if len(reasoning) > 200 else ''}",
                    flush=True,
                )
                print(
                    f"[{task_display}][Step {python_step}] ⚠️  Fallback Action: '{action}'",
                    flush=True,
                )
                print(f"{'╌'*100}\n", flush=True)

        # Step environment using handler interface
        print(f"{'·'*100}", flush=True)
        print(f"⚙️  ENVIRONMENT STEP: [{task_display}][Step {python_step}]", flush=True)
        print(f"{'·'*100}", flush=True)
        step_result = env_handler.step(action, current_node)

        # Track last action to prevent consecutive use
        if action in remove_actions_after_use:
            print(
                f"[{task_display}][Step {python_step}] 🔒 Action '{action}' marked for consecutive-use prevention",
                flush=True,
            )
        last_action = action  # Always update last action regardless of whether it's in removal list

        print(
            f"[{task_display}][Step {python_step}] 📊 Reward: {step_result.reward} | Score: {step_result.score} | Done: {step_result.done} | Internal Steps: {step_result.internal_steps_consumed}",
            flush=True,
        )
        print(f"{'·'*100}\n", flush=True)

        # Complete the step debug info with results using utility
        # Get next URL after step (handler state updated after step)
        next_url = env_handler.get_current_url()

        update_step_debug_with_results(
            step_debug_info,
            reasoning,
            action,
            step_result.reward,
            step_result.score,
            step_result.done,
            step_result.observation,
            next_url=next_url,
        )

        # Log & update memory
        traj["steps"].append(
            {
                "t": python_step,
                "internal_step_before": current_node.internal_step_count,
                "internal_step_after": current_node.internal_step_count
                + step_result.internal_steps_consumed,
                "internal_steps_consumed": step_result.internal_steps_consumed,
                "observation": current_node.observation,
                "inventory": current_node.inventory,
                "reasoning": reasoning,
                "action": action,
                "reward": step_result.reward,
                "score": step_result.score,
                "episodeDone": bool(step_result.done),
                "url": current_url,  # Add URL to trajectory steps
            }
        )

        # Add detailed debug info
        detailed_debug["steps"].append(step_debug_info)

        history.append(
            {
                "step": python_step,
                "action": action,
                "observation": current_node.observation,
                "inventory": current_node.inventory,
                "reward": step_result.reward,
                "reasoning": reasoning,
                "url": current_url,  # Add URL to history
            }
        )

        last_score = step_result.score
        episode_done = step_result.done or episode_done

        # Increment Python action counter (for logging only)
        python_step += 1

        if step_result.done:
            max_steps_display = "unlimited" if is_unlimited_mode else str(max_steps)
            print(f"\n{'='*100}", flush=True)
            print(f"✅ EPISODE COMPLETE: [{task_display}]", flush=True)
            print(f"{'='*100}", flush=True)
            print(
                f"[{task_display}] Completed at Python Step: {python_step}", flush=True
            )
            print(f"[{task_display}] Final Score: {step_result.score}", flush=True)
            print(
                f"[{task_display}] Total Internal Steps: {current_node.internal_step_count + step_result.internal_steps_consumed}",
                flush=True,
            )
            print(f"{'='*100}\n", flush=True)
            break

        # Get next state from handler
        current_node = env_handler.get_current_node(python_step)

    # Get final metadata
    final_metadata = env_handler.get_task_metadata()

    # Attach episode completion metadata (kept in-memory; final aggregation happens in main)
    traj["done"] = bool(episode_done)
    traj["final_score"] = last_score if last_score is not None else 0
    traj["goal_text"] = current_node.goal  # Include goal text for JSON output
    traj["total_python_steps"] = python_step
    traj["final_internal_step"] = current_node.internal_step_count
    traj["max_steps"] = max_steps

    # Complete detailed debug information
    avg_retrieved_length = (
        sum(retrieved_trajectory_lengths) / len(retrieved_trajectory_lengths)
        if retrieved_trajectory_lengths
        else 0
    )
    detailed_debug.update(
        {
            "done": bool(episode_done),
            "final_score": last_score if last_score is not None else 0,
            "total_steps": len(detailed_debug["steps"]),
            "retrieved_trajectory_lengths": retrieved_trajectory_lengths,
            "avg_retrieved_length": avg_retrieved_length,
        }
    )

    # Final episode summary
    print(f"\n{'='*100}", flush=True)
    print(f"📋 EPISODE SUMMARY: [{task_display}]", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"[{task_display}] Episode Done: {bool(episode_done)}", flush=True)
    print(
        f"[{task_display}] Final Score: {last_score if last_score is not None else 0}",
        flush=True,
    )
    print(f"[{task_display}] Total Python Steps: {python_step}", flush=True)
    print(
        f"[{task_display}] Final Internal Steps: {current_node.internal_step_count}",
        flush=True,
    )
    print(f"[{task_display}] Frequency Strategy: {frequency_strategy}", flush=True)
    if retrieved_trajectory_lengths:
        print(
            f"[{task_display}] Trajectories Retrieved: {len(retrieved_trajectory_lengths)} | Avg Length: {avg_retrieved_length:.1f} steps",
            flush=True,
        )
    print(f"{'='*100}\n", flush=True)

    # Close environment
    env_handler.close()

    return traj, detailed_debug
