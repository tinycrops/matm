#!/usr/bin/env python3
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm
import lancedb
import numpy as np
from sentence_transformers import SentenceTransformer
from multiprocessing import Pool
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tfidf_utils import load_tfidf_vectorizers, compute_all_tfidf_features  # noqa: E402

# Environment to process
ENVIRONMENT_FILTER = os.environ.get(
    "ENRICH_ENV", os.environ.get("ENVIRONMENT_FILTER", "alfworld")
)

# Number of parallel workers (can be overridden by NUM_WORKERS env var)
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "32"))
SEQ2SEQ_SOURCE_FALLBACK_MODE = (
    os.environ.get(
        "LTR_SEQ2SEQ_SOURCE_FALLBACK",
        "disabled",
    )
    .strip()
    .lower()
)
VALID_SEQ2SEQ_SOURCE_FALLBACK_MODES = {
    "disabled",
    "max_tsv",
    "mean_tsv",
}

# Environment configurations
ENVIRONMENT_CONFIGS = {
    "alfworld": {
        "base_path": os.environ.get(
            "ALFWORLD_BASE_PATH",
            os.path.join(os.environ.get("MATM_DATA_ROOT", "environments"), "alfworld"),
        ),
        "lancedb_path": os.environ.get(
            "ALFWORLD_LANCEDB_PATH",
            os.path.join(
                os.environ.get("MATM_DATA_ROOT", "environments"),
                "train_only_lancedb/alfworld/lancedb_indices",
            ),
        ),
        "table_name": "alfworld",
        "input_single_query_file": os.environ.get(
            "ENRICH_INPUT_SINGLE",
            "processed_disagreements_single_query_t99.json",
        ),
        "input_multiple_queries_file": os.environ.get(
            "ENRICH_INPUT_MULTIPLE",
            "processed_disagreements_multiple_queries_t99.json",
        ),
        "output_single_query_file": os.environ.get(
            "ENRICH_OUTPUT_SINGLE",
            "1enriched_disagreements_single_query_t99.json",
        ),
        "output_multiple_queries_file": os.environ.get(
            "ENRICH_OUTPUT_MULTIPLE",
            "1enriched_disagreements_multiple_queries_t99.json",
        ),
    },
    "webarena": {
        "base_path": os.environ.get(
            "WEBARENA_BASE_PATH",
            os.path.join(os.environ.get("MATM_DATA_ROOT", "environments"), "webarena"),
        ),
        "lancedb_path": os.environ.get(
            "WEBARENA_LANCEDB_PATH",
            os.path.join(
                os.environ.get("MATM_DATA_ROOT", "environments"),
                "train_only_lancedb/webarena/lancedb_indices",
            ),
        ),
        "table_name": "webarena",
        "input_single_query_file": os.environ.get(
            "ENRICH_INPUT_SINGLE",
            "processed_disagreements_single_query_t99.json",
        ),
        "input_multiple_queries_file": os.environ.get(
            "ENRICH_INPUT_MULTIPLE",
            "processed_disagreements_multiple_queries_t99.json",
        ),
        "output_single_query_file": os.environ.get(
            "ENRICH_OUTPUT_SINGLE",
            "1enriched_disagreements_single_query_t99.json",
        ),
        "output_multiple_queries_file": os.environ.get(
            "ENRICH_OUTPUT_MULTIPLE",
            "1enriched_disagreements_multiple_queries_t99.json",
        ),
    },
}

if SEQ2SEQ_SOURCE_FALLBACK_MODE not in VALID_SEQ2SEQ_SOURCE_FALLBACK_MODES:
    print(
        f"⚠️  Unknown LTR_SEQ2SEQ_SOURCE_FALLBACK={SEQ2SEQ_SOURCE_FALLBACK_MODE!r}; "
        "falling back to 'disabled'"
    )
    SEQ2SEQ_SOURCE_FALLBACK_MODE = "disabled"

# Global embedding model
EMBEDDING_MODEL = None
# Global LanceDB table
LANCEDB_TABLE = None
# Global TF-IDF vectorizers
TFIDF_VECTORIZERS = None
# Global source-model feature table
CONSUMER_MODEL_FEATURES = None
# Global environment name (for worker initialization)
_GLOBAL_ENVIRONMENT = None

CONSUMER_FEATURES_TSV = Path(
    os.environ.get(
        "LTR_CONSUMER_FEATURES_TSV",
        str(Path(__file__).resolve().parents[1] / "LTRConsumerFeatures.tsv"),
    )
)

# WebArena benchmark model mapping (directory name -> (model name, score))
WEBARENA_BENCHMARK_MAP = {
    "agentoccam-judge": ("AgentOccam-Judge", 45.7),
    "2405_all_tasks_step_webarena_bugfix": ("SteP", 33.5),
    "webarena_clean_trajectory": ("Learn-by-Interact", 48.0),
}


def safe_slug(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return safe or "unknown"


def normalize_feature_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def parse_feature_value(raw: str):
    raw = raw.strip()
    if raw == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        return raw
    if value.is_integer():
        return int(value)
    return value


def load_consumer_model_features() -> Dict[str, Dict]:
    if not CONSUMER_FEATURES_TSV.exists():
        print(f"⚠️  Consumer feature TSV not found: {CONSUMER_FEATURES_TSV}")
        return {
            "by_model": {},
            "by_safe_model": {},
            "feature_columns": [],
            "feature_maxima": {},
            "feature_means": {},
        }

    with CONSUMER_FEATURES_TSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            return {
                "by_model": {},
                "by_safe_model": {},
                "feature_columns": [],
                "feature_maxima": {},
                "feature_means": {},
            }

        model_col = fieldnames[0]
        feature_columns = fieldnames[1:]
        by_model: Dict[str, Dict] = {}
        by_safe_model: Dict[str, Dict] = {}
        feature_values: Dict[str, List[float]] = defaultdict(list)

        for row in reader:
            model_slug = (row.get(model_col) or "").strip()
            if not model_slug:
                continue
            parsed = {
                normalize_feature_name(col): parse_feature_value(row.get(col, ""))
                for col in feature_columns
            }
            for feature_name, value in parsed.items():
                if isinstance(value, (int, float)):
                    feature_values[feature_name].append(float(value))
            by_model[model_slug] = parsed
            by_safe_model[safe_slug(model_slug)] = {
                "model_slug": model_slug,
                "features": parsed,
            }

    print(
        f"✓ Loaded consumer model feature table: {len(by_model)} models from {CONSUMER_FEATURES_TSV}"
    )
    return {
        "by_model": by_model,
        "by_safe_model": by_safe_model,
        "feature_columns": [normalize_feature_name(col) for col in feature_columns],
        "feature_maxima": {
            feature_name: max(values)
            for feature_name, values in feature_values.items()
            if values
        },
        "feature_means": {
            feature_name: sum(values) / len(values)
            for feature_name, values in feature_values.items()
            if values
        },
    }


def maybe_apply_seq2seq_source_fallback(
    agent_type: Optional[str],
    source_feature_map: Optional[Dict],
) -> Optional[Dict]:
    if source_feature_map is not None:
        return source_feature_map
    if agent_type != "seq_2_seq":
        return source_feature_map
    if SEQ2SEQ_SOURCE_FALLBACK_MODE == "disabled":
        return source_feature_map

    feature_table = CONSUMER_MODEL_FEATURES or {}
    if SEQ2SEQ_SOURCE_FALLBACK_MODE == "max_tsv":
        fallback = dict(feature_table.get("feature_maxima", {}))
    else:
        fallback = dict(feature_table.get("feature_means", {}))
    return fallback or source_feature_map


def build_source_feature_bundle(feature_map: Optional[Dict]) -> Dict[str, object]:
    bundle: Dict[str, object] = {}
    feature_columns = (CONSUMER_MODEL_FEATURES or {}).get("feature_columns", [])
    for feature_name in feature_columns:
        bundle[f"source_{feature_name}"] = (
            feature_map.get(feature_name) if feature_map else None
        )
    return bundle


def extract_benchmark_info_from_source_file(
    source_file: str,
) -> Tuple[Optional[str], Optional[float]]:
    """
    Extract benchmark model name and score from WebArena source file path.

    Args:
        source_file: Full path to source file (e.g.,
                     "/data/.../webarena/data/agentoccam-judge/1.json")

    Returns:
        Tuple of (model_name, score) or (None, None) if not found
    """
    if not source_file:
        return None, None

    # Extract the directory name from the path
    # Path format: .../webarena/data/{directory_name}/filename.json
    path_parts = source_file.split("/")

    # Find the directory name after "webarena/data"
    # Look for "webarena" first, then find "data" after it
    try:
        webarena_idx = path_parts.index("webarena")
        # Look for "data" after webarena
        if (
            webarena_idx + 1 < len(path_parts)
            and path_parts[webarena_idx + 1] == "data"
        ):
            # The directory we want is after "data"
            if webarena_idx + 2 < len(path_parts):
                directory_name = path_parts[webarena_idx + 2]

                # Look up in the benchmark map
                if directory_name in WEBARENA_BENCHMARK_MAP:
                    model_name, score = WEBARENA_BENCHMARK_MAP[directory_name]
                    return model_name, score
    except (ValueError, IndexError):
        pass

    return None, None


def get_source_file_from_gold_repo(
    thought_id: str, task_name: str, variation_idx: str, environment: str
) -> Optional[str]:
    """
    Look up the source_file from the gold_repo JSON for the given trajectory.

    Args:
        thought_id: Trajectory ID (used as filename in gold_repo)
        task_name: Task name (e.g., "intent_template_id_14")
        variation_idx: Variation index (e.g., "task_id_42")
        environment: Environment name (e.g., "webarena")

    Returns:
        Source file path string, or None if not found
    """
    if environment != "webarena":
        return None

    config = ENVIRONMENT_CONFIGS.get(environment)
    if not config:
        return None

    base_path = config["base_path"]
    gold_repo_path = (
        Path(base_path)
        / "gold_repo"
        / task_name
        / str(variation_idx)
        / f"{thought_id}.json"
    )

    if not gold_repo_path.exists():
        return None

    try:
        with open(gold_repo_path, "r") as f:
            data = json.load(f)
        return data.get("source_file")
    except (json.JSONDecodeError, IOError, KeyError):
        return None


def init_globals(environment: str):
    """Initialize global model and database connection."""
    global EMBEDDING_MODEL, LANCEDB_TABLE, TFIDF_VECTORIZERS, CONSUMER_MODEL_FEATURES, _GLOBAL_ENVIRONMENT

    _GLOBAL_ENVIRONMENT = environment

    print(f"Initializing embedding model (intfloat/e5-base)...")
    EMBEDDING_MODEL = SentenceTransformer("intfloat/e5-base")

    config = ENVIRONMENT_CONFIGS[environment]
    lancedb_path = config["lancedb_path"]
    table_name = config["table_name"]

    print(f"Connecting to LanceDB at {lancedb_path}...")
    db = lancedb.connect(lancedb_path)
    LANCEDB_TABLE = db.open_table(table_name)
    print(f"✓ Connected to table: {table_name}")

    print(f"Loading TF-IDF vectorizers...")
    TFIDF_VECTORIZERS = load_tfidf_vectorizers(environment)
    print(f"✓ Loaded TF-IDF vectorizers")

    CONSUMER_MODEL_FEATURES = load_consumer_model_features()


def init_worker(environment: str):
    """Initialize worker process with its own model and database connection."""
    global EMBEDDING_MODEL, LANCEDB_TABLE, TFIDF_VECTORIZERS, CONSUMER_MODEL_FEATURES, _GLOBAL_ENVIRONMENT

    # Set the environment for this worker
    _GLOBAL_ENVIRONMENT = environment

    # Each worker gets its own embedding model and database connection
    EMBEDDING_MODEL = SentenceTransformer("intfloat/e5-base")

    config = ENVIRONMENT_CONFIGS[environment]
    lancedb_path = config["lancedb_path"]
    table_name = config["table_name"]

    db = lancedb.connect(lancedb_path)
    LANCEDB_TABLE = db.open_table(table_name)

    # Load TF-IDF vectorizers for this worker
    TFIDF_VECTORIZERS = load_tfidf_vectorizers(environment)

    CONSUMER_MODEL_FEATURES = load_consumer_model_features()


def parse_query_components(query: str) -> Dict[str, str]:
    """Parse query text into goal, state, context, progress components.

    Note: Uses re.DOTALL flag to match newlines in multi-line fields like context.
    """
    parts = {}

    # Extract goal (everything from "goal:" to "| state:")
    # Use DOTALL to match newlines in goal text
    goal_match = re.search(r"goal: (.*?) \| state:", query, re.DOTALL)
    if goal_match:
        parts["goal"] = goal_match.group(1).strip()

    # Extract state (from "state:" to "| context:" or "| progress:")
    # Use DOTALL to match newlines in state text
    state_match = re.search(r"state: (.*?) \| (?:context|progress):", query, re.DOTALL)
    if state_match:
        parts["state"] = state_match.group(1).strip()

    # Extract context (from "context:" to "| progress:")
    # Use DOTALL to match newlines in context text (CRITICAL: context often has multiple lines!)
    context_match = re.search(r"context: (.*?) \| progress:", query, re.DOTALL)
    if context_match:
        parts["context"] = context_match.group(1).strip()
    else:
        parts["context"] = ""

    # Extract progress (from "progress:" to end)
    progress_match = re.search(r"progress: (.*?)$", query, re.DOTALL)
    if progress_match:
        parts["progress"] = progress_match.group(1).strip()

    return parts


def extract_step_index(progress: str) -> Optional[int]:
    """Extract step index from progress string like 'step_till_now: 0 | current_reward: 0.0'."""
    match = re.search(r"step_till_now:\s*(\d+)", progress)
    if match:
        return int(match.group(1))
    return None


def find_lancedb_row(thought_id: str, retrieved_chunk: List[Dict]) -> Optional[Dict]:
    """
    Find the LanceDB row by thought_id and matching retrieved_chunk with guidance.

    Args:
        thought_id: Trajectory ID to search for
        retrieved_chunk: List of action-observation pairs to match

    Returns:
        Dictionary with row data or None if not found
    """
    # Query LanceDB for rows with matching thought_id
    results = (
        LANCEDB_TABLE.search()
        .where(f"thought_id = '{thought_id}'")
        .limit(1000)
        .to_list()
    )

    if not results:
        return None

    # Match retrieved_chunk with guidance
    for row in results:
        guidance_str = row.get("guidance", "[]")
        guidance = json.loads(guidance_str)

        # Check if guidance matches retrieved_chunk
        if len(guidance) != len(retrieved_chunk):
            continue

        # Compare action-observation pairs
        match = True
        for i, (chunk_step, guide_step) in enumerate(zip(retrieved_chunk, guidance)):
            if chunk_step.get("action") != guide_step.get("action"):
                match = False
                break
            if chunk_step.get("observation") != guide_step.get("observation"):
                match = False
                break

        if match:
            return row

    return None


def compute_lexical_overlap(text1: str, text2: str) -> float:
    """
    Compute normalized token overlap between two texts.

    Jaccard similarity = |intersection| / |union|
    Tokenization: lowercase and split on whitespace

    Returns:
        Jaccard similarity (intersection over union of tokens)
        Range: [0, 1], where 1 means identical token sets
    """
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    return len(intersection) / len(union) if union else 0.0


def compute_query_overlap_ratio(query_text: str, retrieved_text: str) -> float:
    """
    Compute query overlap ratio: what fraction of query words appear in retrieved text.

    Formula: |intersection| / |query_words|
    Tokenization: lowercase and split on whitespace

    Returns:
        Query overlap ratio (intersection over query size)
        Range: [0, 1], where 1 means all query words appear in retrieved text
    """
    query_tokens = set(query_text.lower().split())
    retrieved_tokens = set(retrieved_text.lower().split())

    if not query_tokens:
        return 0.0

    intersection = query_tokens.intersection(retrieved_tokens)

    return len(intersection) / len(query_tokens)


def compute_embedding_similarity(embed1: List[float], embed2: List[float]) -> float:
    """
    Compute positive L2 distance between two embeddings.

    L2 distance (Euclidean distance) = sqrt(sum((a_i - b_i)^2))
    Returns positive distance to match retriever_score interpretation.

    Returns:
        Positive L2 distance (LOWER is more similar, 0 is identical)
        Range: [0, inf), where 0 means identical embeddings

    Note: This matches the interpretation of retriever_score from LanceDB.
    """
    arr1 = np.array(embed1)
    arr2 = np.array(embed2)
    l2_dist = np.linalg.norm(arr1 - arr2)
    return float(l2_dist)


def count_tokens(text: str) -> int:
    """Simple token count by splitting on whitespace."""
    return len(text.split())


def enrich_entry(entry: Dict) -> Dict:
    """
    Enrich a single disagreement entry with all features.

    IMPORTANT NOTE ON SIMILARITY INTERPRETATIONS:
    ALL similarity features use POSITIVE L2 distance for consistency:
    - embedding_similarity, goal_only_similarity, state_only_similarity,
      context_only_similarity, retriever_score:
      * Range: [0, inf)
      * Interpretation: LOWER = MORE SIMILAR (closer to 0 is more similar)
      * Example: 0.096 is MORE similar than 0.150

    This ensures all similarity metrics have the same interpretation.

    Args:
        entry: Dictionary with query, retrieved_chunk, metadata, score

    Returns:
        Enriched entry dictionary
    """
    query = entry["query"]
    retrieved_chunk = entry["retrieved_chunk"]
    metadata = entry["metadata"]

    # Parse query into components
    query_parts = parse_query_components(query)

    # Find matching LanceDB row
    thought_id = metadata.get("thought_id", "")
    lancedb_row = find_lancedb_row(thought_id, retrieved_chunk)

    if lancedb_row is None:
        # Return entry with None features if not found
        enriched = entry.copy()
        enriched["enrichment_failed"] = True
        enriched["enrichment_reason"] = "lancedb_row_not_found"
        return enriched

    # Extract LanceDB embeddings
    retrieved_goal_embed = lancedb_row.get("goal_only", [])
    retrieved_state_embed = lancedb_row.get("state_only", [])
    retrieved_context_embed = lancedb_row.get("context_only", [])
    retrieved_key_embed = lancedb_row.get("key_embed", [])

    # Extract LanceDB text components
    retrieved_goal = lancedb_row.get("key_raw_goal", "")
    retrieved_state = lancedb_row.get("key_raw_state", "")
    retrieved_context = lancedb_row.get("key_raw_context", "")
    retrieved_progress = lancedb_row.get("key_raw_progress", "")

    # Reconstruct retrieved raw key
    retrieved_raw_key = f"goal: {retrieved_goal} | state: {retrieved_state} | context: {retrieved_context} | progress: {retrieved_progress}"

    # Compute query embeddings
    query_goal_embed = EMBEDDING_MODEL.encode(query_parts.get("goal", "")).tolist()
    query_state_embed = EMBEDDING_MODEL.encode(query_parts.get("state", "")).tolist()
    query_context_text = query_parts.get("context", "").strip()
    query_context_embed = (
        EMBEDDING_MODEL.encode(query_context_text).tolist()
        if query_context_text
        else [0.0] * len(query_goal_embed)
    )
    query_key_embed = EMBEDDING_MODEL.encode(query).tolist()

    # =========================================================================
    # (1) Query-trajectory matching features
    # =========================================================================

    # Embedding similarity (positive L2 distance, LOWER = MORE SIMILAR)
    embedding_similarity = compute_embedding_similarity(
        query_key_embed, retrieved_key_embed
    )

    # Goal-only similarity (positive L2 distance, LOWER = MORE SIMILAR)
    goal_only_similarity = compute_embedding_similarity(
        query_goal_embed, retrieved_goal_embed
    )

    # State-only similarity (positive L2 distance, LOWER = MORE SIMILAR)
    state_only_similarity = compute_embedding_similarity(
        query_state_embed, retrieved_state_embed
    )

    # Context-only similarity (positive L2 distance, LOWER = MORE SIMILAR)
    # Check if context exists and is non-empty (after stripping whitespace)
    retrieved_context_text = retrieved_context.strip() if retrieved_context else ""

    if query_context_text and retrieved_context_text:
        context_only_similarity = compute_embedding_similarity(
            query_context_embed, retrieved_context_embed
        )
    else:
        context_only_similarity = None

    # =========================================================================
    # Lexical overlap features (Jaccard and query overlap ratio)
    # =========================================================================

    # Full text Jaccard similarity
    text_overlap_jaccard = compute_lexical_overlap(query, retrieved_raw_key)

    # Component-wise Jaccard similarity
    goal_only_jaccard = compute_lexical_overlap(
        query_parts.get("goal", ""), retrieved_goal
    )

    state_only_jaccard = compute_lexical_overlap(
        query_parts.get("state", ""), retrieved_state
    )

    # Context Jaccard (None if either context is empty)
    query_context_text = query_parts.get("context", "").strip()
    retrieved_context_text = retrieved_context.strip() if retrieved_context else ""

    if query_context_text and retrieved_context_text:
        context_only_jaccard = compute_lexical_overlap(
            query_context_text, retrieved_context_text
        )
    else:
        context_only_jaccard = None

    # Query overlap ratio (what fraction of query words appear in retrieved text)
    query_overlap_ratio = compute_query_overlap_ratio(query, retrieved_raw_key)

    # Component-wise query overlap ratio
    goal_only_query_overlap = compute_query_overlap_ratio(
        query_parts.get("goal", ""), retrieved_goal
    )

    state_only_query_overlap = compute_query_overlap_ratio(
        query_parts.get("state", ""), retrieved_state
    )

    # Context query overlap (None if either context is empty)
    if query_context_text and retrieved_context_text:
        context_only_query_overlap = compute_query_overlap_ratio(
            query_context_text, retrieved_context_text
        )
    else:
        context_only_query_overlap = None

    # TF-IDF and bigram overlap features
    tfidf_bigram_features = compute_all_tfidf_features(
        query=query,
        retrieved_key=retrieved_raw_key,
        query_goal=query_parts.get("goal", ""),
        retrieved_goal=retrieved_goal,
        query_state=query_parts.get("state", ""),
        retrieved_state=retrieved_state,
        query_context=query_parts.get("context", ""),
        retrieved_context=retrieved_context,
        vectorizers=TFIDF_VECTORIZERS,
    )

    # Task match (binary: 1 if task names match, 0 otherwise)
    query_task_name = metadata.get("task_name")
    retrieved_task_name = lancedb_row.get("task_name")
    task_match = 1 if query_task_name == retrieved_task_name else 0

    # Task-variation match (binary: 1 if BOTH task AND variation match, 0 otherwise)
    query_variation = str(metadata.get("variation"))
    retrieved_variation = str(lancedb_row.get("variation_idx"))
    task_variation_match = (
        1
        if (
            query_task_name == retrieved_task_name
            and query_variation == retrieved_variation
        )
        else 0
    )

    # =========================================================================
    # (2) Trajectory-intrinsic features
    # =========================================================================

    # Trajectory length (number of steps in retrieved_chunk)
    trajectory_length = len(retrieved_chunk)

    # Retrieved text length (token count)
    retrieved_text = json.dumps(retrieved_chunk)
    retrieved_text_length = count_tokens(retrieved_text)

    # Success flag (from LanceDB)
    success_flag = 1 if lancedb_row.get("success", False) else 0

    # Agent type (from LanceDB)
    agent_type = lancedb_row.get("agent_type", None)

    # Source model features
    source_feature_map = None
    if CONSUMER_MODEL_FEATURES:
        source_feature_map = CONSUMER_MODEL_FEATURES["by_model"].get(agent_type)
        source_feature_map = maybe_apply_seq2seq_source_fallback(
            agent_type, source_feature_map
        )

    # Total steps (from LanceDB metadata JSON)
    lancedb_metadata_str = lancedb_row.get("metadata", "{}")
    try:
        lancedb_metadata = json.loads(lancedb_metadata_str)
        total_steps = lancedb_metadata.get("total_steps", None)
    except (json.JSONDecodeError, TypeError):
        total_steps = None

    # Source file and benchmark info (from gold_repo JSON, WebArena only)
    source_file = None
    source_benchmark_model = None
    source_benchmark_score = None

    if _GLOBAL_ENVIRONMENT == "webarena":
        source_file = get_source_file_from_gold_repo(
            thought_id=thought_id,
            task_name=retrieved_task_name,
            variation_idx=retrieved_variation,
            environment=_GLOBAL_ENVIRONMENT,
        )

        # Extract benchmark model and score from source file path
        if source_file:
            (
                source_benchmark_model,
                source_benchmark_score,
            ) = extract_benchmark_info_from_source_file(source_file)

    # =========================================================================
    # (3) Query features
    # =========================================================================

    # Query length (token count)
    query_length = count_tokens(query)

    # Step index (extract from progress)
    step_index = extract_step_index(query_parts.get("progress", ""))

    # Progress alignment (absolute difference between query step and retrieved trajectory step)
    retrieved_step_index = extract_step_index(retrieved_progress)

    if step_index is not None and retrieved_step_index is not None:
        # Absolute difference: 0 means same step, higher means more misaligned
        progress_alignment_abs_diff = abs(step_index - retrieved_step_index)
    else:
        progress_alignment_abs_diff = None

    # =========================================================================
    # (4) Retrieval-process features
    # =========================================================================

    # Retriever score (from metadata, same as similarity_score from LanceDB)
    # IMPORTANT: This is POSITIVE L2 distance where LOWER = MORE SIMILAR (range: [0, ∞))
    # Now matches the interpretation of all computed similarity features for consistency
    retriever_score = metadata.get("similarity_score")

    # =========================================================================
    # Build enriched entry
    # =========================================================================

    enriched = entry.copy()
    enriched["features"] = {
        # Query-trajectory matching
        # NOTE: All similarity features use POSITIVE L2 distance (range: [0, inf), LOWER = MORE SIMILAR)
        "embedding_similarity": embedding_similarity,
        "goal_only_similarity": goal_only_similarity,
        "state_only_similarity": state_only_similarity,
        "context_only_similarity": context_only_similarity,
        # Jaccard similarity features (range: [0, 1], HIGHER = MORE SIMILAR)
        "text_overlap_jaccard": text_overlap_jaccard,
        "goal_only_jaccard": goal_only_jaccard,
        "state_only_jaccard": state_only_jaccard,
        "context_only_jaccard": context_only_jaccard,
        # Query overlap ratio features (range: [0, 1], HIGHER = MORE OVERLAP)
        "query_overlap_ratio": query_overlap_ratio,
        "goal_only_query_overlap": goal_only_query_overlap,
        "state_only_query_overlap": state_only_query_overlap,
        "context_only_query_overlap": context_only_query_overlap,
        # TF-IDF and bigram overlap features (range: [0, 1], HIGHER = MORE SIMILAR)
        "text_overlap_tfidf": tfidf_bigram_features["text_overlap_tfidf"],
        "text_overlap_bigram": tfidf_bigram_features["text_overlap_bigram"],
        "goal_only_overlap_tfidf": tfidf_bigram_features["goal_only_overlap_tfidf"],
        "goal_only_overlap_bigram": tfidf_bigram_features["goal_only_overlap_bigram"],
        "state_only_overlap_tfidf": tfidf_bigram_features["state_only_overlap_tfidf"],
        "state_only_overlap_bigram": tfidf_bigram_features["state_only_overlap_bigram"],
        "context_only_overlap_tfidf": tfidf_bigram_features[
            "context_only_overlap_tfidf"
        ],
        "context_only_overlap_bigram": tfidf_bigram_features[
            "context_only_overlap_bigram"
        ],
        "task_match": task_match,
        "task_variation_match": task_variation_match,
        # Trajectory-intrinsic
        "trajectory_length": trajectory_length,
        "retrieved_text_length": retrieved_text_length,
        "success_flag": success_flag,
        "agent_type": agent_type,
        "total_steps": total_steps,
        "source_file": source_file,  # WebArena only: original source file path
        "source_benchmark_model": source_benchmark_model,  # WebArena only: benchmark model name
        "source_benchmark_score": source_benchmark_score,  # WebArena only: benchmark score
        # Query features
        "query_length": query_length,
        "step_index": step_index,
        "progress_alignment_abs_diff": progress_alignment_abs_diff,
        # Retrieval-process
        # NOTE: retriever_score uses POSITIVE L2 distance (same interpretation as computed similarities)
        # NOTE: initial_rank_position removed - it's redundant with metadata.rank_retrieve
        "retriever_score": retriever_score,
    }

    enriched["features"].update(build_source_feature_bundle(source_feature_map))

    enriched["enrichment_failed"] = False

    return enriched


def process_single_query_file(input_file: Path, output_file: Path):
    """Process single query disagreements file with parallel processing."""
    print(f"\n{'='*70}")
    print(f"Processing: {input_file.name}")
    print(f"{'='*70}")

    with open(input_file, "r") as f:
        data = json.load(f)

    print(f"Total entries: {len(data)}")
    print(f"Using parallel processing with {NUM_WORKERS} workers...")

    # Prepare items for parallel processing
    items = list(data.items())

    # Create pool with worker initialization
    with Pool(
        processes=NUM_WORKERS, initializer=init_worker, initargs=(_GLOBAL_ENVIRONMENT,)
    ) as pool:
        # Process in parallel with progress bar
        results = list(
            tqdm(
                pool.imap(enrich_entry_wrapper, items),
                total=len(items),
                desc="Enriching entries",
                unit="entry",
            )
        )

    # Reconstruct dictionary
    enriched_data = dict(results)

    failed_count = sum(
        1 for entry in enriched_data.values() if entry.get("enrichment_failed", False)
    )

    print(f"\n✓ Enriched: {len(enriched_data) - failed_count}/{len(enriched_data)}")
    print(f"✗ Failed: {failed_count}/{len(enriched_data)}")

    with open(output_file, "w") as f:
        json.dump(enriched_data, f, indent=2)

    print(f"✓ Saved: {output_file}")


def enrich_entry_wrapper(item: Tuple[str, Dict]) -> Tuple[str, Dict]:
    """Wrapper function for parallel processing of single entries."""
    query_key, entry = item
    enriched_entry = enrich_entry(entry)
    return (query_key, enriched_entry)


def process_multiple_queries_file(input_file: Path, output_file: Path):
    """Process multiple queries disagreements file with parallel processing."""
    print(f"\n{'='*70}")
    print(f"Processing: {input_file.name}")
    print(f"{'='*70}")

    with open(input_file, "r") as f:
        data = json.load(f)

    total_entries = sum(len(entries) for entries in data.values())
    print(f"Total query groups: {len(data)}")
    print(f"Total entries: {total_entries}")
    print(f"Using parallel processing with {NUM_WORKERS} workers...")

    # Flatten all entries with their query keys for parallel processing
    flat_items = []
    for query_key, entries in data.items():
        for entry in entries:
            flat_items.append((query_key, entry))

    # Create pool with worker initialization
    with Pool(
        processes=NUM_WORKERS, initializer=init_worker, initargs=(_GLOBAL_ENVIRONMENT,)
    ) as pool:
        # Process in parallel with progress bar
        results = list(
            tqdm(
                pool.imap(enrich_entry_flat_wrapper, flat_items, chunksize=10),
                total=len(flat_items),
                desc="Enriching entries",
                unit="entry",
            )
        )

    # Reconstruct nested dictionary structure
    enriched_data = {}
    for query_key, enriched_entry in results:
        if query_key not in enriched_data:
            enriched_data[query_key] = []
        enriched_data[query_key].append(enriched_entry)

    failed_count = sum(
        1
        for entries in enriched_data.values()
        for entry in entries
        if entry.get("enrichment_failed", False)
    )

    print(f"\n✓ Enriched: {total_entries - failed_count}/{total_entries}")
    print(f"✗ Failed: {failed_count}/{total_entries}")

    with open(output_file, "w") as f:
        json.dump(enriched_data, f, indent=2)

    print(f"✓ Saved: {output_file}")


def enrich_entry_flat_wrapper(item: Tuple[str, Dict]) -> Tuple[str, Dict]:
    """Wrapper function for parallel processing of flattened entries."""
    query_key, entry = item
    enriched_entry = enrich_entry(entry)
    return (query_key, enriched_entry)


def main():
    print(f"\n{'='*70}")
    print(f"DISAGREEMENT ENRICHMENT")
    print(f"{'='*70}")
    print(f"Environment: {ENVIRONMENT_FILTER}")
    print(f"Parallel processing: {NUM_WORKERS} workers")
    print(f"seq_2_seq source fallback: {SEQ2SEQ_SOURCE_FALLBACK_MODE}")
    print(f"{'='*70}\n")

    # Initialize globals (main process)
    init_globals(ENVIRONMENT_FILTER)

    # Setup paths
    compare_dir = Path(__file__).parent
    env_dir = compare_dir / ENVIRONMENT_FILTER

    # Get config for environment
    config = ENVIRONMENT_CONFIGS[ENVIRONMENT_FILTER]

    # Input files (complete file names from config)
    single_query_input = env_dir / config["input_single_query_file"]
    multiple_queries_input = env_dir / config["input_multiple_queries_file"]

    # Output files (complete file names from config)
    single_query_output = env_dir / config["output_single_query_file"]
    multiple_queries_output = env_dir / config["output_multiple_queries_file"]

    # Process files
    if single_query_input.exists():
        process_single_query_file(single_query_input, single_query_output)
    else:
        print(f"⚠️  File not found: {single_query_input}")

    if multiple_queries_input.exists():
        process_multiple_queries_file(multiple_queries_input, multiple_queries_output)
    else:
        print(f"⚠️  File not found: {multiple_queries_input}")

    print(f"\n{'='*70}")
    print(f"✅ ENRICHMENT COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Required for multiprocessing on Windows and to avoid issues on Linux
    mp.set_start_method("spawn", force=True)
    main()
