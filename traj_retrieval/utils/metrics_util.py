# traj_retrieval/utils/metrics_util.py
# Metrics utilities for error analysis and comprehensive metrics collection

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict


def analyze_episode_errors(detailed_episodes: List[Dict]) -> Dict:
    """
    Analyze errors and warnings from detailed episode data.
    Returns comprehensive error statistics.
    """
    total_warnings = 0
    total_errors = 0
    warning_types = {}
    error_types = {}
    probable_causes = {
        "retrieval_trajectory_mismatch": 0,
        "partial_incomplete_action": 0,
        "similar_word_confusion": 0,
        "not_sure": 0,
    }

    episode_details = []

    for ep_idx, episode in enumerate(detailed_episodes):
        episode_warnings = 0
        episode_errors = 0
        episode_warning_types = {}
        episode_error_types = {}
        episode_probable_causes = {
            "retrieval_trajectory_mismatch": 0,
            "partial_incomplete_action": 0,
            "similar_word_confusion": 0,
            "not_sure": 0,
        }

        steps = episode.get("steps", [])
        for step in steps:
            # Analyze LLM call warnings and errors
            llm_call = step.get("llm_call", {})

            # Count warnings (only those where ultimate fallback was triggered)
            warnings = llm_call.get("warnings", [])

            for warning in warnings:
                if isinstance(warning, dict):
                    # Only count warnings where ultimate fallback was triggered
                    if not warning.get("ultimate_fallback_triggered", False):
                        continue

                    episode_warnings += 1  # Count only ultimate fallback warnings

                    msg = warning.get("message", "")
                    cause = warning.get("probable_cause", "")

                    # Categorize warning type
                    if "non-admissible action" in msg.lower():
                        warning_type = "non_admissible_action"
                    else:
                        warning_type = "other"

                    episode_warning_types[warning_type] = (
                        episode_warning_types.get(warning_type, 0) + 1
                    )

                    # Categorize probable cause
                    if "trajectory context" in cause.lower():
                        episode_probable_causes["retrieval_trajectory_mismatch"] += 1
                    elif "incomplete/partial" in cause.lower():
                        episode_probable_causes["partial_incomplete_action"] += 1
                    elif "similar words" in cause.lower():
                        episode_probable_causes["similar_word_confusion"] += 1
                    else:
                        episode_probable_causes["not_sure"] += 1
                else:
                    # Handle old string format warnings - assume they are ultimate fallback for backward compatibility
                    episode_warnings += 1
                    episode_warning_types["other"] = (
                        episode_warning_types.get("other", 0) + 1
                    )
                    episode_probable_causes["not_sure"] += 1

            # Count errors
            errors = llm_call.get("errors", [])
            episode_errors += len(errors)

            for error in errors:
                if "json parsing failed" in error.lower():
                    error_type = "json_parsing_failed"
                elif "llm call failed" in error.lower():
                    error_type = "llm_call_failed"
                else:
                    error_type = "other"

                episode_error_types[error_type] = (
                    episode_error_types.get(error_type, 0) + 1
                )

            # Check planner errors
            planner = step.get("planner", {})
            planner_errors = planner.get("errors", [])
            episode_errors += len(planner_errors)
            for error in planner_errors:
                episode_error_types["planner_failed"] = (
                    episode_error_types.get("planner_failed", 0) + 1
                )

        # Aggregate episode stats
        total_warnings += episode_warnings
        total_errors += episode_errors

        # Merge episode warning/error types into global counts
        for wtype, count in episode_warning_types.items():
            warning_types[wtype] = warning_types.get(wtype, 0) + count

        for etype, count in episode_error_types.items():
            error_types[etype] = error_types.get(etype, 0) + count

        for cause, count in episode_probable_causes.items():
            probable_causes[cause] += count

        # Store per-episode details
        episode_details.append(
            {
                "episode_index": ep_idx,
                "total_warnings": episode_warnings,
                "total_errors": episode_errors,
                "warning_types": episode_warning_types,
                "error_types": episode_error_types,
                "probable_causes": episode_probable_causes,
                "total_steps": len(steps),
                "final_score": episode.get("final_score", 0),
                "avg_retrieved_trajectory_length": episode.get(
                    "avg_retrieved_length", 0
                ),
            }
        )

    return {
        "summary": {
            "total_warnings": total_warnings,
            "total_errors": total_errors,
            "warning_types": warning_types,
            "error_types": error_types,
            "probable_causes": probable_causes,
        },
        "per_episode": episode_details,
    }


def create_enhanced_metrics(
    task: str,
    strategy: str,
    episodes: List[Dict],
    detailed_episodes: List[Dict],
    environment_name: str = "alfworld",
) -> Dict:
    """
    Create enhanced metrics with error analysis.
    Uses environment-specific success criteria.
    """
    from ..core.environment_factory import EnvironmentFactory

    # Get environment handler to use environment-specific success determination
    env_handler = EnvironmentFactory.create_handler(environment_name)

    final_scores = [ep.get("finalScore", 0) for ep in episodes]
    max_final_score = max(final_scores) if final_scores else 0
    min_final_score = min(final_scores) if final_scores else 0
    avg_final_score = sum(final_scores) / max(len(final_scores), 1)

    # Use environment-specific success determination instead of hardcoded threshold
    success_rate = sum(
        1 for s in final_scores if env_handler.is_successful_episode(s)
    ) / max(len(final_scores), 1)

    # Analyze errors from detailed episodes
    error_analysis = analyze_episode_errors(detailed_episodes)

    # Calculate trajectory length metrics
    all_avg_lengths = [ep.get("avg_retrieved_length", 0) for ep in detailed_episodes]
    global_avg_retrieved_length = (
        sum(all_avg_lengths) / max(len(all_avg_lengths), 1) if all_avg_lengths else 0
    )

    return {
        "task": task,
        "strategy": strategy,
        "episodes": len(final_scores),
        "max_final_score": max_final_score,
        "min_final_score": min_final_score,
        "avg_final_score": avg_final_score,
        "success_rate": success_rate,
        "avg_retrieved_trajectory_length": global_avg_retrieved_length,
        "error_analysis": error_analysis,
    }


def _find_circular_references(obj, path="root", seen=None, depth=0, max_depth=50):
    """
    Debug function to find circular references and non-serializable objects in a data structure.

    Args:
        obj: Object to analyze
        path: Current path in the data structure (for debugging)
        seen: Set of object IDs already visited
        depth: Current recursion depth
        max_depth: Maximum depth to prevent infinite recursion

    Returns:
        List of (path, issue_description) tuples
    """
    if seen is None:
        seen = set()

    issues = []

    # Check depth limit
    if depth > max_depth:
        issues.append((path, f"Max depth {max_depth} exceeded"))
        return issues

    # Handle primitives (JSON-serializable)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return issues

    # Get object ID
    obj_id = id(obj)
    obj_type = type(obj).__name__

    # Check for circular reference
    if obj_id in seen:
        issues.append((path, f"Circular reference detected: {obj_type} (id={obj_id})"))
        return issues

    # Mark as seen
    seen.add(obj_id)

    try:
        # Handle lists and tuples
        if isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                item_issues = _find_circular_references(
                    item, f"{path}[{i}]", seen.copy(), depth + 1, max_depth
                )
                issues.extend(item_issues)
            return issues

        # Handle dictionaries
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_str = str(key) if not isinstance(key, str) else key
                value_issues = _find_circular_references(
                    value, f"{path}.{key_str}", seen.copy(), depth + 1, max_depth
                )
                issues.extend(value_issues)
            return issues

        # Non-primitive, non-container types are potentially problematic
        if not isinstance(obj, (list, tuple, dict, str, int, float, bool, type(None))):
            # Check if it has circular references in its attributes
            if hasattr(obj, "__dict__"):
                issues.append(
                    (path, f"Non-serializable object: {obj_type} with __dict__")
                )
                # Try to inspect its attributes
                try:
                    for attr_name, attr_value in obj.__dict__.items():
                        attr_issues = _find_circular_references(
                            attr_value,
                            f"{path}.{attr_name}",
                            seen.copy(),
                            depth + 1,
                            max_depth,
                        )
                        issues.extend(attr_issues)
                except Exception as e:
                    issues.append((path, f"Error inspecting {obj_type}.__dict__: {e}"))
            else:
                issues.append(
                    (path, f"Non-serializable object: {obj_type} (no __dict__)")
                )
    finally:
        seen.discard(obj_id)

    return issues


def write_detailed_debug_json(
    output_dir: Path,
    task: str,
    strategy: str,
    model: str,
    detailed_episodes: List[Dict],
    max_steps_mode,
    hist: int,
    indices_dir: str,
) -> None:
    """
    Write detailed debugging information to JSON file.

    Args:
        max_steps_mode: Can be an integer (override) or "given" (from evaluation set)
    """
    detailed_out_path = output_dir / f"{task}_trajectories_rag_detailed_path.json"

    # Create debug log path
    debug_log_path = output_dir / f"{task}_debug_circular_reference.log"

    try:
        detailed_data = {
            "taskName": task,
            "retrieval_strategy": strategy,
            "model": model,
            "episodes": detailed_episodes,
            "metadata": {
                "total_episodes": len(detailed_episodes),
                "max_steps_mode": max_steps_mode
                if max_steps_mode is not None
                else "given",
                "history_window": hist,
                "indices_dir": indices_dir,
                "timestamp": datetime.now().isoformat(),
            },
        }

        # Try to serialize directly first
        with detailed_out_path.open("w", encoding="utf-8") as f:
            json.dump(detailed_data, f, ensure_ascii=False, indent=2)
        print(f"[IO] Wrote detailed debugging information: {detailed_out_path}")

    except (TypeError, ValueError) as e:
        error_msg = str(e)
        print(f"[IO] Failed to write detailed debugging information: {error_msg}")

        # Run circular reference detection
        print(
            f"[DEBUG] Analyzing data structure for circular references and non-serializable objects..."
        )
        issues = _find_circular_references(detailed_data)

        # Write debug log
        with debug_log_path.open("w", encoding="utf-8") as f:
            f.write(f"=== Circular Reference / Non-Serializable Object Debug Log ===\n")
            f.write(f"Task: {task}\n")
            f.write(f"Strategy: {strategy}\n")
            f.write(f"Error: {error_msg}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"\n{'='*80}\n")
            f.write(f"ISSUES FOUND: {len(issues)}\n")
            f.write(f"{'='*80}\n\n")

            if issues:
                for i, (path, issue) in enumerate(issues, 1):
                    f.write(f"{i}. Path: {path}\n")
                    f.write(f"   Issue: {issue}\n\n")
            else:
                f.write("No obvious circular references detected.\n")
                f.write("This might be a different serialization issue.\n")

            # Add structure overview
            f.write(f"\n{'='*80}\n")
            f.write(f"DATA STRUCTURE OVERVIEW\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Number of episodes: {len(detailed_episodes)}\n")

            if detailed_episodes:
                f.write(f"\nFirst episode structure:\n")
                first_ep = detailed_episodes[0]
                f.write(f"  Type: {type(first_ep)}\n")
                if isinstance(first_ep, dict):
                    f.write(f"  Keys: {list(first_ep.keys())}\n")
                    for key in first_ep.keys():
                        value = first_ep[key]
                        f.write(f"    {key}: {type(value).__name__}")
                        if isinstance(value, (list, dict)):
                            f.write(f" (len={len(value)})")
                        f.write(f"\n")

                    # Check steps if present
                    if (
                        "steps" in first_ep
                        and isinstance(first_ep["steps"], list)
                        and first_ep["steps"]
                    ):
                        f.write(f"\n  First step structure:\n")
                        first_step = first_ep["steps"][0]
                        f.write(f"    Type: {type(first_step)}\n")
                        if isinstance(first_step, dict):
                            f.write(f"    Keys: {list(first_step.keys())}\n")
                            for key in first_step.keys():
                                value = first_step[key]
                                f.write(f"      {key}: {type(value).__name__}")
                                if isinstance(value, (list, dict)):
                                    f.write(f" (len={len(value)})")
                                elif not isinstance(
                                    value, (str, int, float, bool, type(None))
                                ):
                                    # Non-primitive type - could be an issue
                                    f.write(f" ⚠️  NON-PRIMITIVE")
                                    if hasattr(value, "__dict__"):
                                        f.write(
                                            f" (has __dict__ with keys: {list(value.__dict__.keys()) if hasattr(value, '__dict__') else 'N/A'})"
                                        )
                                f.write(f"\n")

        print(f"[DEBUG] Wrote circular reference debug log: {debug_log_path}")
        print(f"[DEBUG] Please check the log file to identify problematic objects")

        # Re-raise the exception so the caller knows it failed
        raise

    except Exception as e:
        print(f"[IO] Failed to write detailed debugging information: {e}")
        import traceback

        traceback.print_exc()


def write_metrics_json(
    output_dir: Path, task: str, strategy: str, metrics: Dict
) -> None:
    """
    Write metrics to JSON file.
    """
    metrics_json_path = output_dir / f"metrics_{task}_{strategy}.json"
    try:
        with metrics_json_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[IO] Wrote metrics JSON: {metrics_json_path}")
    except Exception as e:
        print(f"[IO] Failed to write metrics JSON: {e}")


def write_metrics_csv(base_output_dir: Path, run_id: str, metrics: Dict) -> None:
    """
    Write metrics to CSV file with run_id tracking.
    """
    try:
        metrics_csv_path = base_output_dir / "metrics_log.csv"
        write_header = not metrics_csv_path.exists()

        with metrics_csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    [
                        "run_id",
                        "task",
                        "strategy",
                        "episodes",
                        "max_final_score",
                        "min_final_score",
                        "avg_final_score",
                        "success_rate",
                        "total_warnings",
                        "total_errors",
                        "non_admissible_warnings",
                        "retrieval_mismatch_count",
                        "partial_action_count",
                        "similar_word_confusion_count",
                        "not_sure_count",
                        "json_parsing_errors",
                        "llm_call_errors",
                        "planner_errors",
                    ]
                )

            error_summary = metrics["error_analysis"]["summary"]
            writer.writerow(
                [
                    run_id,
                    metrics["task"],
                    metrics["strategy"],
                    metrics["episodes"],
                    f"{metrics['max_final_score']:.4f}",
                    f"{metrics['min_final_score']:.4f}",
                    f"{metrics['avg_final_score']:.4f}",
                    f"{metrics['success_rate']:.4f}",
                    error_summary["total_warnings"],
                    error_summary["total_errors"],
                    error_summary["warning_types"].get("non_admissible_action", 0),
                    error_summary["probable_causes"]["retrieval_trajectory_mismatch"],
                    error_summary["probable_causes"]["partial_incomplete_action"],
                    error_summary["probable_causes"]["similar_word_confusion"],
                    error_summary["probable_causes"]["not_sure"],
                    error_summary["error_types"].get("json_parsing_failed", 0),
                    error_summary["error_types"].get("llm_call_failed", 0),
                    error_summary["error_types"].get("planner_failed", 0),
                ]
            )

        print(f"[IO] Appended metrics CSV: {metrics_csv_path}")
    except Exception as e:
        print(f"[METRICS] Failed to write CSV metrics: {e}")
