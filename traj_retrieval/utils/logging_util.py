# traj_retrieval/utils/logging_util.py
# Logging utilities for detailed debugging and error analysis

import json
import traceback
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


# Global variable to store current run_id for debug file organization
_current_run_id: Optional[str] = None


def set_global_run_id(run_id: str):
    """Set the global run_id for debug file organization."""
    global _current_run_id
    _current_run_id = run_id


def get_current_run_id() -> Optional[str]:
    """Get the current run_id for debug file organization."""
    global _current_run_id
    return _current_run_id


def analyze_non_admissible_cause(
    action: str, admissible_actions: List[str], trajectory_context: str
) -> str:
    """
    Analyze the probable cause of why the model returned a non-admissible action.
    """
    # Check if action appears in retrieved trajectory context
    if trajectory_context and action.lower() in trajectory_context.lower():
        # Find the position in trajectory context
        step_position = "unknown"
        lines = trajectory_context.split("\n")
        for i, line in enumerate(lines):
            if action.lower() in line.lower() and "step" in line.lower():
                step_position = line.strip()
                break
        return f"Action found in retrieved trajectory context but not in current admissible actions - trajectory may contain future/different state actions (found at: {step_position})"

    # Check if action is a partial match of admissible actions
    partial_matches = [
        a
        for a in admissible_actions
        if action.lower() in a.lower() or a.lower().startswith(action.lower())
    ]
    if partial_matches:
        return f"Action appears to be incomplete/partial - similar admissible actions exist: {partial_matches}"

    # Check if action is similar to admissible actions (fuzzy matching)
    action_words = set(action.lower().split())
    similar_actions = []
    for admissible in admissible_actions:
        admissible_words = set(admissible.lower().split())
        if (
            len(action_words.intersection(admissible_words)) >= 2
        ):  # At least 2 words in common
            similar_actions.append(admissible)

    if similar_actions:
        return f"Action has similar words to admissible actions - possible confusion with large action space. Similar actions: {similar_actions[:3]}"

    return "Not sure - action doesn't match patterns in trajectory context or admissible actions"


def create_step_debug_info(
    t: int,
    observation: str,
    inventory: str,
    current_admissible: List[str],
    recent_history_str: str,
    trajectory_context: str,
    url: str = "",
) -> Dict:
    """
    Initialize detailed step info for debugging.

    IMPORTANT: Admissible actions are now only shown in LLM prompt section to avoid duplication.
    Environment section only shows count and top 10 actions for brevity.

    Args:
        url: Current URL (empty string for non-web environments like WebArena)
    """
    return {
        "step_number": t,
        "environment": {
            "observation": observation,
            "inventory": inventory,
            "admissible_actions_top_10": current_admissible[
                :10
            ],  # Show only top 10 for brevity
            "total_admissible_actions": len(current_admissible),
            "url": url,  # URL for web environments (WebArena), empty string otherwise
        },
        # Removed "context" field as it's not needed in debug output
        "retrieval": {},
        "planner": {},
        "llm_call": {},
        "result": {},
    }


def update_step_debug_with_retrieval(
    step_debug_info: Dict,
    strategy: str,
    tag: str,
    action_obs_pairs: List[Dict],
    trajectory_context: str,
    triggered_by_planner: bool = False,
    retrieval_metadata: Optional[Dict] = None,
) -> None:
    """
    Update step debug info with retrieval information.

    Args:
        step_debug_info: The step debug info dictionary to update
        strategy: Retrieval strategy name
        tag: Planner decision tag (if triggered by planner)
        action_obs_pairs: List of action-observation pairs from retrieved trajectory
        trajectory_context: The formatted trajectory context string
        triggered_by_planner: Whether retrieval was triggered by planner
        retrieval_metadata: Extended metadata from RetrievalResult (includes thought_id, similarity scores, etc.)
    """
    base_retrieval_info = {
        "retrieved_actions_observations": action_obs_pairs,
        "context_provided": bool(trajectory_context.strip()),
    }

    # Add extended metadata if provided
    if retrieval_metadata:
        base_retrieval_info.update(
            {
                "thought_id": retrieval_metadata.get("thought_id", ""),
                "task_name": retrieval_metadata.get("task_name", ""),
                "variation_idx": retrieval_metadata.get("variation_idx", ""),
                "similarity_score": retrieval_metadata.get("similarity_score", 0.0),
                "trajectory_length": retrieval_metadata.get("trajectory_length", 0),
                "vector_store": retrieval_metadata.get("vector_store", ""),
                # Retrieval ranking information
                "rank_retrieve": retrieval_metadata.get(
                    "rank_retrieve", 1
                ),  # Which rank was selected (1-indexed)
                "top_k": retrieval_metadata.get(
                    "top_k", 100
                ),  # How many results were retrieved for stats
                # Top 100 statistics
                "top_100_mean_distance": retrieval_metadata.get(
                    "top_100_mean_distance", 0.0
                ),
                "top_100_std_distance": retrieval_metadata.get(
                    "top_100_std_distance", 0.0
                ),
                "top_100_count": retrieval_metadata.get("top_100_count", 0),
                # Query debug info (NEW: for debugging filters)
                "query_debug": retrieval_metadata.get("query_debug", {}),
                # Query text (NEW: exact query string used for embedding and LanceDB search)
                "query": retrieval_metadata.get("query", ""),
            }
        )

    if triggered_by_planner:
        step_debug_info["retrieval"] = {
            "triggered_by_planner": True,
            "planner_decision": tag,
            **base_retrieval_info,
        }
    else:
        step_debug_info["retrieval"] = {"strategy": strategy, **base_retrieval_info}

    # Note: Removed context field update since context was removed from step_debug_info structure


def update_step_debug_with_results(
    step_debug_info: Dict,
    reasoning: str,
    action: str,
    reward: int,
    step_score: int,
    done: bool,
    next_obs: str,
    next_url: str = "",
) -> None:
    """
    Complete the step debug info with results.

    Args:
        next_url: Next URL after action (empty string for non-web environments)
    """
    step_debug_info["result"] = {
        "reasoning": reasoning,
        "action": action,
        "reward": reward,
        "score": step_score,
        "episode_done": bool(done),
        "next_observation": str(next_obs),
        "next_url": next_url,  # URL after action execution (for WebArena)
    }


def create_detailed_debug_structure(
    task_name: str,
    rand_variation_idx: int,
    max_variations: int,
    goal_text: str,
    retrieval_strategy: str,
    simulation_metadata: Optional[Dict] = None,
    alfworld_metadata: Optional[Dict] = None,
) -> Dict:
    """
    Create the initial detailed debugging information structure.

    Args:
        task_name: Task name
        rand_variation_idx: Variation index
        max_variations: Maximum variations
        goal_text: Goal text
        retrieval_strategy: Retrieval strategy name
        simulation_metadata: Optional metadata for simulate_till_tk strategy
        alfworld_metadata: Optional ALFWorld-specific metadata (game_file, task_id, floor_plan)
    """
    debug_structure = {
        "task": task_name,
        "variation": rand_variation_idx,
        "max_variations": max_variations,
        "goal_text": goal_text,
        "retrieval_strategy": retrieval_strategy,
        "steps": [],
    }

    # Add simulation metadata if provided (for simulate_till_tk strategy)
    if simulation_metadata:
        debug_structure["simulation_metadata"] = simulation_metadata

    # Add ALFWorld-specific metadata if provided (only for ALFWorld environment)
    if alfworld_metadata:
        if "game_file" in alfworld_metadata:
            debug_structure["game_file"] = alfworld_metadata["game_file"]
        if "task_id" in alfworld_metadata:
            debug_structure["task_id"] = alfworld_metadata["task_id"]
        if "floor_plan" in alfworld_metadata:
            debug_structure["floor_plan"] = alfworld_metadata["floor_plan"]

    return debug_structure


def create_llm_debug_info(
    user_payload: Dict, trajectory_context: str, model: str, url: str = ""
) -> Dict:
    """
    Create simplified debug info for LLM calls.

    IMPORTANT: This captures the FULL user payload information for debugging.
    Captures all components that go into the LLM prompt.

    Note: Reasoning is NOT included in prompt_components as it's the LLM's response, not part of the prompt.

    Args:
        url: Current URL (empty string for non-web environments)
    """
    return {
        "prompt_components": {
            "goal": user_payload.get("goal", ""),
            "observation": user_payload.get("current_observation", ""),
            "inventory": user_payload.get("inventory", ""),
            "admissible_actions": user_payload.get("admissible_actions", []),
            "recent_history": user_payload.get("recent_history", ""),
            "trajectory_context": trajectory_context
            if trajectory_context
            else "",  # Full content instead of just boolean
            "trajectory_context_present": bool(
                trajectory_context.strip()
            ),  # Keep boolean for backward compatibility
            "url": url  # URL for web environments (WebArena), empty string otherwise
            # Note: reasoning is NOT here - it's the LLM response, stored in result.reasoning
        },
        "model": model,
        "errors": [],
        "warnings": [],
        "system_message": None,  # Will be set by strategy
        "user_message": None,  # Will be set by strategy
    }


def create_planner_debug_info(user_payload: Dict, model: str) -> Dict:
    """
    Create simplified debug info for planner calls.
    """
    return {
        "prompt_components": {
            "goal": user_payload.get("goal", ""),
            "current_observation": user_payload.get("current_observation", ""),
            "recent_history": user_payload.get("recent_history", ""),
            "current_retrieved_context": user_payload.get(
                "current_retrieved_context", ""
            ),
        },
        "model": model,
        "errors": [],
        "warnings": [],
    }


def write_llm_debug_file(
    error: Exception,
    model: str,
    messages: List[Dict],
    response_text: str,
    action_mapping: Optional[Dict[int, str]] = None,
    step_info: Optional[Dict] = None,
    output_dir: str = "llm_debug_logs",
    run_id: Optional[str] = None,
) -> str:
    """
    Write detailed debug information for failed LLM calls to a separate file.

    Args:
        error: The exception that occurred
        model: The model name used
        messages: The messages sent to the LLM
        response_text: The raw response text from the LLM
        action_mapping: Optional mapping of action IDs to action strings
        step_info: Optional step context information
        output_dir: Directory to save debug files
        run_id: Optional run ID for organizing files

    Returns:
        Path to the created debug file
    """
    # Create organized directory structure
    debug_base_dir = Path(output_dir)
    debug_base_dir.mkdir(parents=True, exist_ok=True)

    # Use provided run_id or global run_id
    if run_id is None:
        run_id = get_current_run_id()

    # Organize by run_id if available
    if run_id:
        debug_dir = debug_base_dir / f"run_{run_id}"
        debug_dir.mkdir(parents=True, exist_ok=True)
    else:
        debug_dir = debug_base_dir

    # Further organize by error type
    error_type = type(error).__name__
    error_type_dir = debug_dir / error_type.lower()
    error_type_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds

    # Include strategy info in filename if available
    strategy_name = ""
    if step_info and "strategy" in step_info:
        strategy_name = f"_{step_info['strategy']}"
    elif step_info and "stage" in step_info:
        strategy_name = f"_{step_info['stage']}"

    debug_file = (
        error_type_dir / f"llm_error_{timestamp}{strategy_name}_{error_type}.json"
    )

    # Prepare debug data
    debug_data = {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "error_info": {
            "error_type": error_type,
            "error_message": str(error),
            "error_traceback": traceback.format_exc(),
        },
        "llm_request": {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "action_mapping_size": len(action_mapping) if action_mapping else 0,
            "action_mapping_sample": dict(list(action_mapping.items())[:10])
            if action_mapping
            else {},
        },
        "llm_response": {
            "raw_text": response_text,
            "text_length": len(response_text),
            "starts_with": response_text[:100] if response_text else "",
            "ends_with": response_text[-100:]
            if len(response_text) > 100
            else response_text,
        },
        "step_context": step_info or {},
        "action_mapping_full": action_mapping or {},
    }

    # Write debug file
    try:
        with debug_file.open("w", encoding="utf-8") as f:
            json.dump(debug_data, f, indent=2, ensure_ascii=False)
        print(f"[DEBUG] Wrote LLM error debug file: {debug_file}")
        return str(debug_file)
    except Exception as write_error:
        print(f"[WARNING] Failed to write LLM debug file: {write_error}")
        return ""
