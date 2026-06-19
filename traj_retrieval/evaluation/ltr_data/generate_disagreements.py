#!/usr/bin/env python3
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import statistics
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Try faster JSON parser
try:
    import ujson as json_parser
except ImportError:
    import json as json_parser

try:
    # Use RapidFuzz when available
    from rapidfuzz import fuzz

    def compute_query_similarity(query1: str, query2: str) -> float:
        return fuzz.ratio(query1, query2) / 100.0

except ImportError:
    from difflib import SequenceMatcher

    def compute_query_similarity(query1: str, query2: str) -> float:
        return SequenceMatcher(None, query1, query2).ratio()


ENVIRONMENT_FILTER = os.environ.get("DISAGREEMENTS_ENV", "alfworld")
MAX_COMPARISONS_TO_PROCESS = float("inf")

# Use all available CPUs for maximum speed, or override with NUM_WORKERS env var
# MEMORY WARNING: Trajectory files can be 100-800MB each. With N workers, peak memory
# can reach N * 800MB + 15GB (for main process). Recommended: 32 workers with 128GB RAM.
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", mp.cpu_count()))  # Parallel file I/O

ENVIRONMENT_CONFIGS = {
    "alfworld": {
        "base_path": os.environ.get(
            "ALFWORLD_BASE_PATH",
            os.path.join(os.environ.get("MATM_DATA_ROOT", "environments"), "alfworld"),
        ),
        "traj_logs": os.environ.get(
            "ALFWORLD_TRAJ_LOGS",
            os.path.join(
                os.environ.get(
                    "ALFWORLD_BASE_PATH",
                    os.path.join(os.environ.get("MATM_DATA_ROOT", "environments"), "alfworld"),
                ),
                "traj_logs",
            ),
        ),
        "simulation_traj_logs": os.environ.get(
            "ALFWORLD_SIMULATION_TRAJ_LOGS",
            os.path.join(
                os.environ.get(
                    "ALFWORLD_BASE_PATH",
                    os.path.join(os.environ.get("MATM_DATA_ROOT", "environments"), "alfworld"),
                ),
                "simulation_traj_logs",
            ),
        ),
        "query_similarity_threshold": 0.99,
        "comparison_file_pattern": "simulation_compare_{env}_e0_vs_e6_ranks.json",
        "output_single_query_file": "processed_disagreements_single_query_t99.json",
        "output_multiple_queries_file": "processed_disagreements_multiple_queries_t99.json",
        "output_stats_file": "processed_disagreements_stats_t99.json",
    },
    "webarena": {
        "base_path": os.environ.get(
            "WEBARENA_BASE_PATH",
            os.path.join(os.environ.get("MATM_DATA_ROOT", "environments"), "webarena"),
        ),
        "traj_logs": os.environ.get(
            "WEBARENA_TRAJ_LOGS",
            os.path.join(
                os.environ.get(
                    "WEBARENA_BASE_PATH",
                    os.path.join(os.environ.get("MATM_DATA_ROOT", "environments"), "webarena"),
                ),
                "traj_logs",
            ),
        ),
        "simulation_traj_logs": os.environ.get(
            "WEBARENA_SIMULATION_TRAJ_LOGS",
            os.path.join(
                os.environ.get(
                    "WEBARENA_BASE_PATH",
                    os.path.join(os.environ.get("MATM_DATA_ROOT", "environments"), "webarena"),
                ),
                "simulation_traj_logs",
            ),
        ),
        "query_similarity_threshold": 0.99,
        "comparison_file_pattern": "simulation_compare_{env}_e0_vs_e7_ranks.json",
        "output_single_query_file": "processed_disagreements_single_query_t99.json",
        "output_multiple_queries_file": "processed_disagreements_multiple_queries_t99.json",
        "output_stats_file": "processed_disagreements_stats_t99.json",
    },
}

QUERY_SIMILARITY_THRESHOLD = ENVIRONMENT_CONFIGS.get(ENVIRONMENT_FILTER, {}).get(
    "query_similarity_threshold", 0.99
)

# Global cache for trajectory files
trajectory_cache = {}


def clean_webarena_query(query: str) -> str:
    """
    Remove numbers in square brackets from WebArena queries.
    Example: "[1] RootWebArea" -> " RootWebArea"
    """
    # Remove patterns like [1], [123], etc.
    cleaned = re.sub(r"\[\d+\]", "", query)
    return cleaned


def preprocess_query_for_clustering(query: str, environment: str) -> str:
    """
    Preprocess query based on environment-specific rules.
    Returns the cleaned query to be used for clustering.
    """
    if environment == "webarena":
        return clean_webarena_query(query)
    return query


def cluster_queries(queries: List[str]) -> Dict[int, List[int]]:
    """
    Cluster queries with progress tracking.
    Uses greedy clustering to maintain order-dependent consistency.
    """
    if len(queries) <= 1:
        return {0: [0]} if queries else {}

    clusters = {}
    assigned = [False] * len(queries)
    cluster_id = 0

    # Add progress bar for clustering
    for i in tqdm(range(len(queries)), desc="Clustering queries", unit="query"):
        if assigned[i]:
            continue

        clusters[cluster_id] = [i]
        assigned[i] = True

        # Compare with remaining unassigned queries
        for j in range(i + 1, len(queries)):
            if assigned[j]:
                continue

            similarity = compute_query_similarity(queries[i], queries[j])
            if similarity >= QUERY_SIMILARITY_THRESHOLD:
                clusters[cluster_id].append(j)
                assigned[j] = True

        cluster_id += 1

    return clusters


def find_episode_in_trajectory(
    traj_data: Dict,
    task_name: str,
    variation: str,
    goal_text: str,
    simulate_step_k: Optional[int],
) -> Optional[Dict]:
    if "episodes" not in traj_data:
        return None

    normalized_goal = " ".join(goal_text.split())

    for episode in traj_data["episodes"]:
        ep_variation = str(episode.get("variation", ""))
        ep_goal = " ".join(
            episode.get("goal_text", episode.get("goalText", "")).split()
        )

        if ep_variation != str(variation):
            continue

        if ep_goal != normalized_goal:
            continue

        if simulate_step_k is not None:
            sim_metadata = episode.get("simulation_metadata", {})
            ep_step_k = sim_metadata.get("simulate_step_k")
            if ep_step_k != simulate_step_k:
                continue

        return episode

    return None


def extract_retrieval_from_episode(episode: Dict) -> Optional[Dict]:
    if "steps" not in episode:
        return None

    # Early exit when retrieval found
    for step in episode["steps"]:
        if "retrieval" in step and step["retrieval"]:
            retrieval = step["retrieval"]
            if (
                "retrieved_actions_observations" in retrieval
                and retrieval["retrieved_actions_observations"]
            ):
                return retrieval

    return None


def get_retrieval_info_single(args: Tuple) -> Optional[Tuple[str, str, Dict]]:
    """
    Extract retrieval info from a single file.
    Used for parallel processing.
    Returns (original_query, cleaned_query, full_run_data) or None
    """
    (
        run_id,
        task_name,
        variation,
        goal_text,
        simulate_step_k,
        simulation_logs_path,
        run_type,
        success,
        comp_key,
        environment,
    ) = args

    if run_type == "baseline":
        return None  # Baselines handled separately

    try:
        run_path = Path(simulation_logs_path) / run_id / f"run_{task_name}"
        traj_file = run_path / f"{task_name}_trajectories_rag_detailed_path.json"
        print(traj_file)
        if not traj_file.exists():
            return None

        # Read file
        with open(traj_file, "r") as f:
            traj_data = json_parser.load(f)

        episode = find_episode_in_trajectory(
            traj_data, task_name, variation, goal_text, simulate_step_k
        )
        if episode is None:
            return None

        retrieval = extract_retrieval_from_episode(episode)
        if retrieval is None:
            return None

        original_query = retrieval.get("query", "")
        if not original_query:
            return None

        # Preprocess query for clustering
        cleaned_query = preprocess_query_for_clustering(original_query, environment)

        run_data = {
            "run_id": run_id,
            "run_type": run_type,
            "success": success,
            "query": original_query,  # Keep original query as main query
            "retrieved_chunk": retrieval.get("retrieved_actions_observations", []),
            "metadata": {
                "task_name": task_name,
                "variation": variation,
                "goal_text": goal_text,
                "thought_id": retrieval.get("thought_id", ""),
                "retrieved_task_name": retrieval.get("task_name", ""),
                "retrieved_variation": retrieval.get("variation_idx", ""),
                "similarity_score": retrieval.get("similarity_score", 0.0),
                "rank_retrieve": retrieval.get("rank_retrieve", None),
                "cleaned_query": cleaned_query,  # Store cleaned query in metadata
            },
            "comp_key": comp_key,
        }

        return (original_query, cleaned_query, run_data)
    except Exception as e:
        return None


def compute_stats(single_query_data: Dict, multiple_queries_data: Dict) -> Dict:
    single_lengths = [1] * len(single_query_data)
    multiple_lengths = [len(entries) for entries in multiple_queries_data.values()]
    all_lengths = single_lengths + multiple_lengths

    if not all_lengths:
        return {
            "single_query": {"total_queries": 0, "total_entries": 0},
            "multiple_queries": {
                "total_queries": 0,
                "total_entries": 0,
                "avg_entries_per_query": 0,
                "min_entries_per_query": 0,
                "max_entries_per_query": 0,
                "std_entries_per_query": 0,
            },
            "overall": {
                "total_queries": 0,
                "total_entries": 0,
                "avg_entries_per_query": 0,
            },
        }

    return {
        "single_query": {
            "total_queries": len(single_query_data),
            "total_entries": len(single_query_data),
        },
        "multiple_queries": {
            "total_queries": len(multiple_queries_data),
            "total_entries": sum(multiple_lengths),
            "avg_entries_per_query": statistics.mean(multiple_lengths)
            if multiple_lengths
            else 0,
            "min_entries_per_query": min(multiple_lengths) if multiple_lengths else 0,
            "max_entries_per_query": max(multiple_lengths) if multiple_lengths else 0,
            "std_entries_per_query": statistics.stdev(multiple_lengths)
            if len(multiple_lengths) > 1
            else 0,
        },
        "overall": {
            "total_queries": len(single_query_data) + len(multiple_queries_data),
            "total_entries": sum(all_lengths),
            "avg_entries_per_query": statistics.mean(all_lengths),
        },
    }


def process_comparison_file(
    comparison_file: Path, environment: str
) -> Tuple[Dict, Dict]:
    print(f"{'='*70}")
    print(f"PROCESSING COMPARISON FILE")
    print(f"{'='*70}")
    print(f"File: {comparison_file}")
    print(f"Environment: {environment}")
    print(f"Loading comparison data...")

    with open(comparison_file, "r") as f:
        data = json_parser.load(f)

    comparisons = data.get("comparisons", [])
    print(f"Total comparisons in file: {len(comparisons)}")

    # Pre-filter valid comparisons
    print("Filtering valid comparisons...")
    valid_comparisons = []
    for comparison in tqdm(comparisons, desc="Filtering", unit="comp"):
        results = comparison.get("results", [])
        if not results:
            continue
        has_simulation = any(r["run_type"] == "simulation" for r in results)
        if not has_simulation:
            continue
        valid_comparisons.append(comparison)

    comparisons_to_process = (
        valid_comparisons[: int(MAX_COMPARISONS_TO_PROCESS)]
        if MAX_COMPARISONS_TO_PROCESS != float("inf")
        else valid_comparisons
    )
    print(f"Valid comparisons to process: {len(comparisons_to_process)}")

    config = ENVIRONMENT_CONFIGS[environment]
    simulation_logs_path = config["simulation_traj_logs"]

    # Prepare tasks for parallel processing
    print(f"\nPreparing extraction tasks...")
    extraction_tasks = []
    baseline_runs = []

    for comparison in comparisons_to_process:
        results = comparison.get("results", [])
        task_name = comparison["task_name"]
        variation = comparison["variation"]
        goal_text = comparison["goal_text"]
        simulate_step_k = comparison.get("simulate_step_k")
        comp_key = f"{task_name}|||{variation}|||{simulate_step_k}"

        for result in results:
            run_id = result["run_id"]
            run_type = result["run_type"]
            success = result["success"]

            if run_type == "baseline":
                baseline_runs.append(
                    {
                        "run_id": run_id,
                        "run_type": run_type,
                        "success": success,
                        "query": None,
                        "retrieved_chunk": [],
                        "metadata": {
                            "task_name": task_name,
                            "variation": variation,
                            "goal_text": goal_text,
                            "rank_retrieve": None,
                            "cleaned_query": None,
                        },
                        "comp_key": comp_key,
                    }
                )
            else:
                extraction_tasks.append(
                    (
                        run_id,
                        task_name,
                        variation,
                        goal_text,
                        simulate_step_k,
                        simulation_logs_path,
                        run_type,
                        success,
                        comp_key,
                        environment,
                    )
                )

    print(f"Total baseline runs: {len(baseline_runs)}")
    print(f"Total simulation files to process: {len(extraction_tasks)}")

    # Process files in parallel
    all_runs_by_original_query = defaultdict(list)
    all_runs_by_cleaned_query = defaultdict(list)
    all_runs_by_original_query[None] = baseline_runs

    print(f"\nExtracting retrieval info using {NUM_WORKERS} parallel workers...")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(get_retrieval_info_single, task): task
            for task in extraction_tasks
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing files",
            unit="file",
        ):
            result = future.result()
            if result is not None:
                original_query, cleaned_query, run_data = result
                all_runs_by_original_query[original_query].append(run_data)
                all_runs_by_cleaned_query[cleaned_query].append(run_data)

    unique_original_query_count = len(
        [q for q in all_runs_by_original_query.keys() if q is not None]
    )
    unique_cleaned_query_count = len(
        [q for q in all_runs_by_cleaned_query.keys() if q is not None]
    )

    print(f"\n{'='*70}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"Total unique original queries: {unique_original_query_count}")
    print(f"Total unique cleaned queries: {unique_cleaned_query_count}")

    # Use cleaned queries for clustering
    unique_cleaned_queries = [
        q for q in all_runs_by_cleaned_query.keys() if q is not None
    ]

    # Clustering phase
    print(f"\n{'='*70}")
    print(f"CLUSTERING PHASE")
    print(f"{'='*70}")

    cleaned_query_to_cluster = {}
    if len(unique_cleaned_queries) > 1:
        print(f"Clustering {len(unique_cleaned_queries)} unique cleaned queries...")
        print(f"Similarity threshold: {QUERY_SIMILARITY_THRESHOLD}")

        query_clusters = cluster_queries(unique_cleaned_queries)

        print(f"Created {len(query_clusters)} clusters")

        for cluster_id, query_indices in query_clusters.items():
            cluster_head_query = unique_cleaned_queries[query_indices[0]]
            for idx in query_indices:
                cleaned_query_to_cluster[
                    unique_cleaned_queries[idx]
                ] = cluster_head_query
    else:
        for q in unique_cleaned_queries:
            cleaned_query_to_cluster[q] = q

    # Group runs by cluster (using cleaned queries)
    print("\nGrouping runs by cluster...")
    clustered_runs = defaultdict(list)
    for cleaned_query, runs in tqdm(
        all_runs_by_cleaned_query.items(), desc="Grouping runs"
    ):
        if cleaned_query is None:
            continue
        cluster_head = cleaned_query_to_cluster.get(cleaned_query, cleaned_query)
        # Add cluster head to each run for later use
        for run in runs:
            run["cluster_head_query"] = cluster_head
        clustered_runs[cluster_head].extend(runs)

    baseline_runs = all_runs_by_original_query[None]

    # Compute disagreements
    print(f"\n{'='*70}")
    print(f"COMPUTING DISAGREEMENTS")
    print(f"{'='*70}")

    # NOTE: Disagreement entries are keyed by cluster_query (the cluster head)
    # This ensures that runs with similar queries (e.g., differing only in element IDs)
    # are grouped together under the same cluster head query.
    # The original query for each run is preserved in metadata.original_query
    disagreement_entries = defaultdict(
        lambda: {"cluster_query": "", "retrieved_chunk": [], "metadata": {}, "score": 0}
    )

    for cluster_query, cluster_runs in tqdm(
        clustered_runs.items(), desc="Processing clusters", unit="cluster"
    ):
        runs_by_comp = defaultdict(list)
        for run in cluster_runs:
            runs_by_comp[run["comp_key"]].append(run)

        for comp_key, comp_runs in runs_by_comp.items():
            matching_baselines = [b for b in baseline_runs if b["comp_key"] == comp_key]
            all_comp_runs = comp_runs + matching_baselines

            # Pairwise disagreement detection
            for i in range(len(all_comp_runs)):
                for j in range(i + 1, len(all_comp_runs)):
                    run_i = all_comp_runs[i]
                    run_j = all_comp_runs[j]

                    if run_i["success"] != run_j["success"]:
                        if run_i["run_type"] == "simulation":
                            key = f"{run_i['run_id']}|||{run_i['metadata']['task_name']}|||{run_i['metadata']['variation']}|||{comp_key.split('|||')[-1]}"

                            if disagreement_entries[key]["cluster_query"] == "":
                                # Use cluster head query for grouping
                                disagreement_entries[key]["cluster_query"] = run_i.get(
                                    "cluster_head_query",
                                    run_i["metadata"]["cleaned_query"],
                                )
                                disagreement_entries[key]["retrieved_chunk"] = run_i[
                                    "retrieved_chunk"
                                ]
                                disagreement_entries[key]["metadata"] = run_i[
                                    "metadata"
                                ].copy()
                                # Add original query to metadata
                                disagreement_entries[key]["metadata"][
                                    "original_query"
                                ] = run_i["query"]

                            if run_i["success"]:
                                disagreement_entries[key]["score"] += 1
                            else:
                                disagreement_entries[key]["score"] -= 1

                        if run_j["run_type"] == "simulation":
                            key = f"{run_j['run_id']}|||{run_j['metadata']['task_name']}|||{run_j['metadata']['variation']}|||{comp_key.split('|||')[-1]}"

                            if disagreement_entries[key]["cluster_query"] == "":
                                # Use cluster head query for grouping
                                disagreement_entries[key]["cluster_query"] = run_j.get(
                                    "cluster_head_query",
                                    run_j["metadata"]["cleaned_query"],
                                )
                                disagreement_entries[key]["retrieved_chunk"] = run_j[
                                    "retrieved_chunk"
                                ]
                                disagreement_entries[key]["metadata"] = run_j[
                                    "metadata"
                                ].copy()
                                # Add original query to metadata
                                disagreement_entries[key]["metadata"][
                                    "original_query"
                                ] = run_j["query"]

                            if run_j["success"]:
                                disagreement_entries[key]["score"] += 1
                            else:
                                disagreement_entries[key]["score"] -= 1

    print(f"Total disagreement entries: {len(disagreement_entries)}")

    # Organize by cluster query (using cluster head queries)
    print("\nOrganizing entries by cluster query...")
    query_entries = defaultdict(list)
    for key, entry in tqdm(disagreement_entries.items(), desc="Organizing"):
        if entry["cluster_query"]:
            query_entries[entry["cluster_query"]].append(
                {
                    "run_key": key,
                    "query": entry["cluster_query"],  # Use cluster head as the query
                    "retrieved_chunk": entry["retrieved_chunk"],
                    "metadata": entry["metadata"],
                    "score": entry["score"],
                }
            )

    single_query_data = {}
    multiple_queries_data = {}

    for query, entries in query_entries.items():
        if len(entries) == 1:
            single_query_data[query] = entries[0]
        else:
            multiple_queries_data[query] = entries

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Single query entries: {len(single_query_data)}")
    print(f"Multiple query entries: {len(multiple_queries_data)}")

    return single_query_data, multiple_queries_data


def main():
    print(f"\n{'='*70}")
    print(f"DISAGREEMENT PROCESSOR - OPTIMIZED")
    print(f"{'='*70}")
    print(f"Environment: {ENVIRONMENT_FILTER}")
    print(f"Query similarity threshold: {QUERY_SIMILARITY_THRESHOLD}")
    print(f"Parallel workers: {NUM_WORKERS}")
    print(f"{'='*70}\n")

    compare_dir = Path(__file__).parent
    env_dir = compare_dir / ENVIRONMENT_FILTER

    config = ENVIRONMENT_CONFIGS[ENVIRONMENT_FILTER]
    comparison_file_override = os.environ.get("DISAGREEMENTS_COMPARISON_FILE")
    if comparison_file_override:
        comparison_file = Path(comparison_file_override)
    else:
        comparison_file_pattern = config["comparison_file_pattern"]
        comparison_filename = comparison_file_pattern.format(env=ENVIRONMENT_FILTER)
        comparison_file = env_dir / comparison_filename

    if not comparison_file.exists():
        print(f"❌ Comparison file not found: {comparison_file}")
        return

    single_query_data, multiple_queries_data = process_comparison_file(
        comparison_file, ENVIRONMENT_FILTER
    )

    stats = compute_stats(single_query_data, multiple_queries_data)

    print(f"\n{'='*70}")
    print(f"STATISTICS")
    print(f"{'='*70}")
    print(f"\nSingle Query:")
    print(f"  Total queries: {stats['single_query']['total_queries']}")
    print(f"  Total entries: {stats['single_query']['total_entries']}")
    print(f"\nMultiple Queries:")
    print(f"  Total queries: {stats['multiple_queries']['total_queries']}")
    print(f"  Total entries: {stats['multiple_queries']['total_entries']}")
    print(
        f"  Average entries per query: {stats['multiple_queries']['avg_entries_per_query']:.2f}"
    )
    print(
        f"  Min entries per query: {stats['multiple_queries']['min_entries_per_query']}"
    )
    print(
        f"  Max entries per query: {stats['multiple_queries']['max_entries_per_query']}"
    )
    print(
        f"  Std entries per query: {stats['multiple_queries']['std_entries_per_query']:.2f}"
    )
    print(f"\nOverall:")
    print(f"  Total queries: {stats['overall']['total_queries']}")
    print(f"  Total entries: {stats['overall']['total_entries']}")
    print(
        f"  Average entries per query: {stats['overall']['avg_entries_per_query']:.2f}"
    )

    print(f"\n{'='*70}")
    print(f"SAVING RESULTS")
    print(f"{'='*70}")

    # Use complete output filenames from config unless overridden
    output_single = env_dir / os.environ.get(
        "DISAGREEMENTS_OUTPUT_SINGLE",
        config["output_single_query_file"],
    )
    output_multiple = env_dir / os.environ.get(
        "DISAGREEMENTS_OUTPUT_MULTIPLE",
        config["output_multiple_queries_file"],
    )
    output_stats = env_dir / os.environ.get(
        "DISAGREEMENTS_OUTPUT_STATS",
        config["output_stats_file"],
    )

    with open(output_single, "w") as f:
        json.dump(single_query_data, f, indent=2)
    print(f"✓ {output_single}")

    with open(output_multiple, "w") as f:
        json.dump(multiple_queries_data, f, indent=2)
    print(f"✓ {output_multiple}")

    with open(output_stats, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✓ {output_stats}")

    print(f"\n{'='*70}")
    print(f"✅ COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
