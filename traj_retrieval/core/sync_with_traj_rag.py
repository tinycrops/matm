# traj_retrieval/core/sync_with_traj_rag.py
# Synchronous core functionality for trajectory retrieval
# This module is completely synchronous and does NOT import asyncio
# Created specifically for environments like WebArena that use Playwright's sync API

import os
import sys

# Set environment variables BEFORE any other imports
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if sys.platform == "darwin":  # macOS
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
else:
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")

import json
import random
from datetime import datetime
from collections import deque
from typing import Tuple, List, Optional, Deque, Dict, Any

# CRITICAL: Do NOT import asyncio anywhere in this module
# Playwright checks for asyncio at the process level

# IMPORTANT: Cannot use openai library (even sync client) because it imports asyncio
# This pollutes the process and breaks Playwright sync API
# Instead, use requests library for direct HTTP calls
import requests

# CRITICAL: Close any existing asyncio event loop before using Playwright sync API
# Even if asyncio is imported by dependencies, we must ensure no event loop exists
try:
    import asyncio

    loop = asyncio.get_event_loop()
    if loop and not loop.is_closed():
        loop.close()
    asyncio.set_event_loop(None)
except Exception:
    pass  # If asyncio not imported yet, that's fine

# Import environment handlers
from .base_handler import EnvironmentHandler, TrajectoryNode, StepResult

# Import utilities (SYNC versions only)
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
from .retrieval_strategy import RetrievalManager, RetrievalResult
from .retrieval_handler_sync import SyncRetrievalHandler


class SimpleSyncHTTPClient:
    """
    Simple synchronous HTTP client for OpenRouter API.

    This mimics the OpenAI client interface but uses requests library
    to avoid importing asyncio (which breaks Playwright sync API).

    Note: This class avoids circular references to ensure JSON serializability
    of debug info structures.
    """

    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.base_url = base_url
        # Pass api_key and base_url directly to avoid circular reference
        self.chat = self._ChatNamespace(api_key, base_url)

    class _ChatNamespace:
        def __init__(self, api_key: str, base_url: str):
            self.api_key = api_key
            self.base_url = base_url
            self.completions = self._CompletionsNamespace(api_key, base_url)

        class _CompletionsNamespace:
            def __init__(self, api_key: str, base_url: str):
                self.api_key = api_key
                self.base_url = base_url

            def create(
                self,
                model: str,
                messages: List[Dict],
                temperature: float = 0.0,
                max_tokens: int = 2000,
                response_format: Optional[Dict] = None,
            ):
                """Make a synchronous API call to OpenRouter."""
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }

                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }

                if response_format:
                    payload["response_format"] = response_format

                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120,
                )

                if response.status_code != 200:
                    raise Exception(
                        f"API error {response.status_code}: {response.text}"
                    )

                data = response.json()

                # Create a simple response object that mimics OpenAI's structure
                class SimpleResponse:
                    def __init__(self, data):
                        self.choices = [self._Choice(data["choices"][0])]

                    class _Choice:
                        def __init__(self, choice_data):
                            self.message = self._Message(choice_data["message"])

                        class _Message:
                            def __init__(self, message_data):
                                self.content = message_data.get("content", "")

                return SimpleResponse(data)


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
# Synchronous Episode Runner (analogous to run_episode_async but completely sync)
# -----------------------------
def run_episode_sync(
    env_handler: EnvironmentHandler,
    api_key: str,
    model_slug: str,
    retrieval_manager: RetrievalManager,
    frequency_strategy: str = "t0",
    history_window: int = 10,
    experiment_type: str = "action_id_mapping",
    remove_actions_after_use: Optional[List[str]] = None,
    ultimate_fallback_action: str = "look around",
    simulation_loader=None,  # NEW: SimulationActionsLoader for simulate_till_tk
    simulate_step_k: int = None,  # NEW: Per-episode simulate_step_k (required for simulate_till_tk)
    agentic_until_step_k: int = None,
    retrieve_once_step_k: int = None,
    global_python_cut_off_step: int = None,  # NEW: Global hard limit on Python steps (overrides max_steps)
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Synchronous episode runner (no asyncio).

    This is the synchronous analog of run_episode_async() from async_with_traj_rag.py.
    It follows the same structure but executes everything synchronously.

    The handler must be already initialized before calling this function.

    IMPORTANT: This module does NOT import asyncio anywhere. It's designed for
    environments like WebArena that use Playwright's sync API, which conflicts
    with asyncio at the process level.

    Args:
        env_handler: Environment handler (already initialized) implementing EnvironmentHandler interface
        api_key: OpenAI API key (synchronous version uses direct HTTP instead of async client)
        model_slug: Model identifier
        retrieval_manager: RetrievalManager for memory-augmented retrieval
        frequency_strategy: When to retrieve ("none", "t0", "every_10", "agentic", "agentic_until_tk", "retrieve_once_at_tk", "simulate_till_tk")
        history_window: Number of previous steps in history
        experiment_type: Experiment strategy type
        remove_actions_after_use: Actions to prevent consecutive use
        ultimate_fallback_action: Fallback when LLM fails
        simulation_loader: SimulationActionsLoader instance (required for simulate_till_tk)
        agentic_until_step_k: For agentic_until_tk, use agentic retrieval only while python_step < k
        retrieve_once_step_k: For retrieve_once_at_tk, force exactly one retrieval when python_step == k

    Returns:
        Tuple of (trajectory_dict, detailed_debug_dict)
    """
    # Handler is already initialized by caller (like run_episode_async)

    # Get initial state
    current_node = env_handler.reset()

    # Get metadata for logging
    metadata = env_handler.get_task_metadata()
    task_display = f"{metadata['task_name']}:{metadata['variation']}"
    is_unlimited_mode = metadata.get("is_unlimited", False)
    max_steps = metadata.get("max_steps", 50)

    # Initialize simple synchronous HTTP client (avoids asyncio import from openai library)
    sync_client = SimpleSyncHTTPClient(
        api_key=api_key, base_url="https://openrouter.ai/api/v1"
    )

    # Get strategy for action selection (SYNC version)
    strategy = get_experiment_strategy(experiment_type, use_sync=True)

    print(f"\n{'='*100}", flush=True)
    print(f"🎯 EPISODE START (SYNC MODE): [{task_display}]", flush=True)
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
    last_action = None

    # NEW: Load simulation steps (action + reasoning) for simulate_till_tk strategy
    simulation_steps = None
    simulation_metadata = None
    target_state_info = (
        None  # NEW: For pseudo_simulate_till_closest_tk (contains observation + URL)
    )

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
            "frequency_strategy": "simulate_till_tk",
            "simulate_step_k": simulate_step_k,
            "simulation_file": str(simulation_loader.json_file_path),
            "total_simulation_steps_loaded": len(simulation_steps),
            "environment_name": simulation_loader.environment_name,
        }

    elif frequency_strategy == "pseudo_simulate_till_closest_tk":
        # NEW: pseudo_simulate_till_closest_tk strategy
        if simulation_loader is None:
            raise ValueError(
                "pseudo_simulate_till_closest_tk strategy requires simulation_loader parameter"
            )

        if simulate_step_k is None:
            raise ValueError(
                "pseudo_simulate_till_closest_tk strategy requires simulate_step_k parameter (should be in episode descriptor)"
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
            print(f"Strategy: pseudo_simulate_till_closest_tk")
            print(f"Simulation file: {simulation_loader.json_file_path}")
            print(
                f"\nThis episode will be SKIPPED because pseudo_simulate_till_closest_tk requires pre-recorded data."
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

        # Load the target state info (observation + URL) at step k from simulation data
        target_state_info = simulation_loader.get_target_state_info(
            task_name=metadata["task_name"],
            variation_id=variation_id,
            target_step_k=simulate_step_k,
        )

        print(
            f"[{task_display}] 🔍 PSEUDO-SIMULATION MODE: Monitoring for state similarity"
        )
        print(f"[{task_display}] 🔍 Target step k: {simulate_step_k}")
        print(f"[{task_display}] 🔍 Target state info:")
        print(
            f"[{task_display}]     - observation (first 100 chars): {target_state_info['observation'][:100] if target_state_info['observation'] else 'None'}..."
        )
        print(
            f"[{task_display}]     - url: {target_state_info['url'] if target_state_info['url'] else '(empty - will not be used for matching)'}"
        )
        print(
            f"[{task_display}] 🔍 Will check EVERY step (starting from step 0) for state similarity"
        )
        print(
            f"[{task_display}] 🔍 When current state matches target: trigger ONE-TIME retrieval"
        )
        print(
            f"[{task_display}] 🔍 After retrieval: use LLM with retrieved trajectory for rest of episode"
        )

        # Create simulation metadata for debug logging
        simulation_metadata = {
            "frequency_strategy": "pseudo_simulate_till_closest_tk",
            "simulate_step_k": simulate_step_k,
            "target_state_info": {
                "observation_preview": target_state_info["observation"][:100]
                if target_state_info["observation"]
                else None,
                "observation_length": len(target_state_info["observation"])
                if target_state_info["observation"]
                else 0,
                "url": target_state_info["url"],
                "url_used_for_matching": bool(
                    target_state_info["url"]
                ),  # True if URL will be used
            },
            "simulation_file": str(simulation_loader.json_file_path),
            "environment_name": simulation_loader.environment_name,
        }

    # Initialize retrieval handler (centralized retrieval logic - SYNC version)
    sync_retrieval_handler = SyncRetrievalHandler(
        frequency_strategy=frequency_strategy,
        retrieval_manager=retrieval_manager,
        sync_client=sync_client,
        model_slug=model_slug,
        simulation_loader=simulation_loader,  # NEW: Pass simulation_loader
        simulate_step_k=simulate_step_k,  # NEW: Pass per-episode simulate_step_k
        target_state_info=target_state_info,  # NEW: Pass target state info (observation + URL) for pseudo_simulate_till_closest_tk
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

    # Detailed debugging information
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

        # Build history (sliding window memory) string
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
        retrieval_result = sync_retrieval_handler.retrieve_if_needed(
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
            # LLM MODE: Ask the strategy to select an action (synchronous LLM calls)
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
                # Create a NEW debug_info for the strategy (to avoid circular reference)
                # This matches the async version which creates a fresh debug_info in llm_choose_action_async
                strategy_debug_info = {}

                # Call strategy to select action (synchronous)
                action, strategy_debug = strategy.execute(
                    sclient=sync_client,
                    limiter=None,  # No rate limiting in sync mode
                    requests_sem=None,  # No semaphore in sync mode
                    model=model_slug,
                    goal_text=current_node.goal,
                    observation=current_node.observation,
                    inventory=current_node.inventory,
                    admissible_actions=current_admissible,
                    recent_history_str=recent_history_str,
                    trajectory_context=trajectory_context,
                    debug_info=strategy_debug_info,  # Pass NEW dict to avoid circular reference
                    env_handler=env_handler,
                    ultimate_fallback_action=ultimate_fallback_action,
                    episode_id=f"{task_display}",
                    current_step=python_step,
                    max_steps=max_steps,
                )

                # Store strategy debug info into step_debug_info (now no circular reference)
                step_debug_info["llm_call"] = strategy_debug

                reasoning = strategy_debug.get("reasoning", "")
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
                    f"[{task_display}][Step {python_step}] ❌ Strategy Error: {type(e).__name__} - {str(e)[:100]}",
                    flush=True,
                )

                # Fallback to a safe action using configured ultimate_fallback_action
                print(f"\n{'╌'*100}", flush=True)
                print(
                    f"🔄 FALLBACK ACTION: [{task_display}][Step {python_step}]",
                    flush=True,
                )
                print(f"{'╌'*100}", flush=True)

                action = ultimate_fallback_action
                reasoning = f"Agent could not decide the action so falling back to {ultimate_fallback_action} action"
                step_debug_info["llm_call"] = {
                    "errors": [f"Strategy execution failed: {str(e)}"],
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
        last_action = action  # Always update last action

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

        # Add detailed debug info
        detailed_debug["steps"].append(step_debug_info)

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

    # Attach episode completion metadata
    traj["done"] = bool(episode_done)
    traj["final_score"] = last_score if last_score is not None else 0
    traj["goal_text"] = current_node.goal
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
            "total_steps": len(detailed_debug["steps"])
            if "steps" in detailed_debug
            else python_step,
            "retrieved_trajectory_lengths": retrieved_trajectory_lengths,
            "avg_retrieved_length": avg_retrieved_length,
        }
    )

    # NEW: Add pseudo_simulate_till_closest_tk metadata to detailed_debug
    if frequency_strategy == "pseudo_simulate_till_closest_tk":
        detailed_debug["pseudo_simulate_metadata"] = {
            "retrieval_triggered": sync_retrieval_handler.get_retrieval_triggered_status(),
            "target_step_k": simulate_step_k,
            "total_steps_executed": python_step,
            "observation_matched": sync_retrieval_handler.get_retrieval_triggered_status(),
        }

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

    # NEW: Warning for pseudo_simulate_till_closest_tk if retrieval never triggered
    if frequency_strategy == "pseudo_simulate_till_closest_tk":
        if not sync_retrieval_handler.get_retrieval_triggered_status():
            print(f"\n{'⚠'*100}", flush=True)
            print(f"⚠️  WARNING: RETRIEVAL NEVER TRIGGERED", flush=True)
            print(f"{'⚠'*100}", flush=True)
            print(
                f"[{task_display}] Strategy: pseudo_simulate_till_closest_tk",
                flush=True,
            )
            print(
                f"[{task_display}] Target step k (from simulation data): {simulate_step_k}",
                flush=True,
            )
            print(f"[{task_display}] Total steps executed: {python_step}", flush=True)
            print(
                f"[{task_display}] Observation NEVER matched target observation",
                flush=True,
            )
            print(
                f"[{task_display}] Agent ran entire episode WITHOUT retrieval",
                flush=True,
            )
            print(f"[{task_display}] This may indicate:", flush=True)
            print(
                f"[{task_display}]   1. Agent took different path than simulation",
                flush=True,
            )
            print(
                f"[{task_display}]   2. Target observation at step {simulate_step_k} was not reached",
                flush=True,
            )
            print(
                f"[{task_display}]   3. Observation similarity threshold (1.0) is too strict",
                flush=True,
            )
            print(f"{'⚠'*100}\n", flush=True)
        else:
            print(
                f"[{task_display}] ✅ Retrieval successfully triggered during episode",
                flush=True,
            )

    print(f"{'='*100}\n", flush=True)

    # Close environment
    env_handler.close()

    return traj, detailed_debug
