"""Runtime feature construction for LTR reranking.

This module builds query-doc features online from:
- query text + query metadata
- LanceDB retrieved rows
- optional TF-IDF vectorizers

The feature names are aligned to training TSV columns consumed by LTR models.
"""

from __future__ import annotations

import csv
import json
import os
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


VALID_SEQ2SEQ_SOURCE_FALLBACK_MODES = {
    "disabled",
    "max_tsv",
    "mean_tsv",
}


def safe_slug(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return safe or "unknown"


def normalize_feature_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def parse_feature_value(raw: Any) -> Any:
    raw_text = str(raw).strip()
    if raw_text == "":
        return None
    try:
        value = float(raw_text)
    except ValueError:
        return raw_text
    if value.is_integer():
        return int(value)
    return value


def _default_consumer_features_tsv() -> Path:
    env_path = os.environ.get("LTR_CONSUMER_FEATURES_TSV")
    if env_path:
        return Path(env_path)
    return (
        Path(__file__).resolve().parents[1]
        / "traj_retrieval"
        / "evaluation"
        / "LTRConsumerFeatures.tsv"
    )


_CONSUMER_FEATURES_TSV = _default_consumer_features_tsv()
_SEQ2SEQ_SOURCE_FALLBACK_MODE = (
    os.environ.get(
        "LTR_SEQ2SEQ_SOURCE_FALLBACK",
        "disabled",
    )
    .strip()
    .lower()
)
if _SEQ2SEQ_SOURCE_FALLBACK_MODE not in VALID_SEQ2SEQ_SOURCE_FALLBACK_MODES:
    _SEQ2SEQ_SOURCE_FALLBACK_MODE = "disabled"

_MODEL_FEATURE_TABLE: Optional[Dict[str, Any]] = None


def load_consumer_model_features() -> Dict[str, Any]:
    global _MODEL_FEATURE_TABLE
    if _MODEL_FEATURE_TABLE is not None:
        return _MODEL_FEATURE_TABLE

    if not _CONSUMER_FEATURES_TSV.exists():
        _MODEL_FEATURE_TABLE = {
            "by_model": {},
            "by_safe_model": {},
            "feature_columns": [],
            "feature_maxima": {},
            "feature_means": {},
        }
        return _MODEL_FEATURE_TABLE

    with _CONSUMER_FEATURES_TSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            _MODEL_FEATURE_TABLE = {
                "by_model": {},
                "by_safe_model": {},
                "feature_columns": [],
                "feature_maxima": {},
                "feature_means": {},
            }
            return _MODEL_FEATURE_TABLE

        model_col = fieldnames[0]
        feature_columns = [normalize_feature_name(col) for col in fieldnames[1:]]
        by_model: Dict[str, Dict[str, Any]] = {}
        by_safe_model: Dict[str, Dict[str, Any]] = {}
        feature_values: Dict[str, List[float]] = {name: [] for name in feature_columns}

        for row in reader:
            model_slug = (row.get(model_col) or "").strip()
            if not model_slug:
                continue
            parsed = {
                normalize_feature_name(col): parse_feature_value(row.get(col, ""))
                for col in fieldnames[1:]
            }
            for feature_name, value in parsed.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    feature_values.setdefault(feature_name, []).append(float(value))
            by_model[model_slug] = parsed
            by_safe_model[safe_slug(model_slug)] = {
                "model_slug": model_slug,
                "features": parsed,
            }

    _MODEL_FEATURE_TABLE = {
        "by_model": by_model,
        "by_safe_model": by_safe_model,
        "feature_columns": feature_columns,
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
    return _MODEL_FEATURE_TABLE


def resolve_model_feature_map(model_slug: Any) -> Optional[Dict[str, Any]]:
    if model_slug is None:
        return None
    slug = str(model_slug).strip()
    if not slug:
        return None
    feature_table = load_consumer_model_features()
    feature_map = feature_table["by_model"].get(slug)
    if feature_map is not None:
        return feature_map
    safe_match = feature_table["by_safe_model"].get(safe_slug(slug))
    if isinstance(safe_match, dict):
        features = safe_match.get("features")
        if isinstance(features, dict):
            return features
    return None


def maybe_apply_seq2seq_source_fallback(
    agent_type: Any,
    source_feature_map: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if source_feature_map is not None:
        return source_feature_map
    if str(agent_type or "") != "seq_2_seq":
        return source_feature_map
    feature_table = load_consumer_model_features()
    if _SEQ2SEQ_SOURCE_FALLBACK_MODE == "max_tsv":
        fallback = dict(feature_table.get("feature_maxima", {}))
    elif _SEQ2SEQ_SOURCE_FALLBACK_MODE == "mean_tsv":
        fallback = dict(feature_table.get("feature_means", {}))
    else:
        fallback = {}
    return fallback or source_feature_map


def build_model_feature_bundle(
    prefix: str,
    feature_map: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    bundle: Dict[str, Any] = {}
    feature_columns = load_consumer_model_features().get("feature_columns", [])
    for feature_name in feature_columns:
        bundle[f"{prefix}_{feature_name}"] = (
            feature_map.get(feature_name) if feature_map else None
        )
    return bundle


def build_delta_feature_bundle(
    left_prefix: str,
    left_map: Optional[Dict[str, Any]],
    right_prefix: str,
    right_map: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    bundle: Dict[str, Any] = {}
    feature_columns = load_consumer_model_features().get("feature_columns", [])
    for feature_name in feature_columns:
        left_value = left_map.get(feature_name) if left_map else None
        right_value = right_map.get(feature_name) if right_map else None
        if isinstance(left_value, (int, float, np.integer, np.floating)) and isinstance(
            right_value, (int, float, np.integer, np.floating)
        ):
            bundle[f"{left_prefix}_minus_{right_prefix}_{feature_name}"] = float(
                left_value
            ) - float(right_value)
        else:
            bundle[f"{left_prefix}_minus_{right_prefix}_{feature_name}"] = None
    return bundle


def parse_query_components(query: str) -> Dict[str, str]:
    """Parse query text into goal/state/context/progress fields."""
    parts: Dict[str, str] = {}

    goal_match = re.search(r"goal: (.*?) \| state:", query, re.DOTALL)
    if goal_match:
        parts["goal"] = goal_match.group(1).strip()

    state_match = re.search(r"state: (.*?) \| (?:context|progress):", query, re.DOTALL)
    if state_match:
        parts["state"] = state_match.group(1).strip()

    context_match = re.search(r"context: (.*?) \| progress:", query, re.DOTALL)
    if context_match:
        parts["context"] = context_match.group(1).strip()
    else:
        parts["context"] = ""

    progress_match = re.search(r"progress: (.*?)$", query, re.DOTALL)
    if progress_match:
        parts["progress"] = progress_match.group(1).strip()

    return parts


def extract_step_index(progress: str) -> Optional[int]:
    """Extract step index from progress text like 'step_till_now: 3 | ...'."""
    match = re.search(r"step_till_now:\s*(\d+)", progress)
    if match:
        return int(match.group(1))
    return None


def compute_embedding_similarity(embed1: List[float], embed2: List[float]) -> float:
    """Compute positive L2 distance between two embeddings."""
    arr1 = np.asarray(embed1, dtype=float)
    arr2 = np.asarray(embed2, dtype=float)
    return float(np.linalg.norm(arr1 - arr2))


def compute_lexical_overlap(text1: str, text2: str) -> float:
    """Compute token Jaccard similarity."""
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return float(len(intersection) / len(union)) if union else 0.0


def compute_query_overlap_ratio(query_text: str, retrieved_text: str) -> float:
    """Compute fraction of query tokens appearing in retrieved text."""
    query_tokens = set(query_text.lower().split())
    retrieved_tokens = set(retrieved_text.lower().split())
    if not query_tokens:
        return 0.0
    intersection = query_tokens.intersection(retrieved_tokens)
    return float(len(intersection) / len(query_tokens))


def count_tokens(text: str) -> int:
    """Simple whitespace token count."""
    return len(text.split())


def load_tfidf_vectorizers(
    environment: str,
    base_dir: Optional[Path] = None,
) -> Dict[str, TfidfVectorizer]:
    """Load TF-IDF vectorizers for one environment."""
    valid_environments = {"alfworld", "webarena"}
    if environment not in valid_environments:
        raise ValueError(
            f"Invalid environment '{environment}'. Must be one of {sorted(valid_environments)}."
        )

    if base_dir is None:
        base_dir = Path(__file__).resolve().parent

    env_dir = base_dir / environment
    if not env_dir.exists():
        raise FileNotFoundError(f"TF-IDF environment directory not found: {env_dir}")

    vectorizers: Dict[str, TfidfVectorizer] = {}
    for component in ("full", "goal", "state", "context"):
        path = env_dir / f"tfidf_vectorizer_{component}.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing TF-IDF vectorizer: {path}. "
                "Run TF-IDF vectorizer build first."
            )
        with path.open("rb") as file_obj:
            try:
                vectorizers[component] = pickle.load(file_obj)
            except ModuleNotFoundError as error:
                # Compatibility shim for pickles referencing numpy._core.* symbols.
                if "numpy._core" not in str(error):
                    raise
                import numpy.core as numpy_core
                import numpy.core.numeric as numpy_core_numeric

                sys.modules.setdefault("numpy._core", numpy_core)
                sys.modules.setdefault("numpy._core.numeric", numpy_core_numeric)
                file_obj.seek(0)
                vectorizers[component] = pickle.load(file_obj)
    return vectorizers


def compute_tfidf_similarity(
    text1: str,
    text2: str,
    vectorizer: TfidfVectorizer,
) -> float:
    """Compute TF-IDF cosine similarity in [0, 1]."""
    if not text1.strip() or not text2.strip():
        return 0.0
    try:
        vec1 = vectorizer.transform([text1])
        vec2 = vectorizer.transform([text2])
        similarity = cosine_similarity(vec1, vec2)[0, 0]
        return float(np.clip(similarity, 0.0, 1.0))
    except Exception:
        return 0.0


def extract_bigrams(text: str) -> Set[Tuple[str, str]]:
    """Extract bigram set from text."""
    tokens = text.lower().split()
    if len(tokens) < 2:
        return set()
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


def compute_bigram_overlap(text1: str, text2: str) -> float:
    """Compute bigram Jaccard similarity."""
    bigrams1 = extract_bigrams(text1)
    bigrams2 = extract_bigrams(text2)
    if not bigrams1 or not bigrams2:
        return 0.0
    intersection = bigrams1.intersection(bigrams2)
    union = bigrams1.union(bigrams2)
    if not union:
        return 0.0
    return float(len(intersection) / len(union))


def compute_all_tfidf_features(
    query: str,
    retrieved_key: str,
    query_goal: str,
    retrieved_goal: str,
    query_state: str,
    retrieved_state: str,
    query_context: str,
    retrieved_context: str,
    vectorizers: Dict[str, TfidfVectorizer],
) -> Dict[str, Optional[float]]:
    """Compute TF-IDF + bigram overlap features."""
    features: Dict[str, Optional[float]] = {}
    features["text_overlap_tfidf"] = compute_tfidf_similarity(
        query, retrieved_key, vectorizers["full"]
    )
    features["text_overlap_bigram"] = compute_bigram_overlap(query, retrieved_key)
    features["goal_only_overlap_tfidf"] = compute_tfidf_similarity(
        query_goal, retrieved_goal, vectorizers["goal"]
    )
    features["goal_only_overlap_bigram"] = compute_bigram_overlap(
        query_goal, retrieved_goal
    )
    features["state_only_overlap_tfidf"] = compute_tfidf_similarity(
        query_state, retrieved_state, vectorizers["state"]
    )
    features["state_only_overlap_bigram"] = compute_bigram_overlap(
        query_state, retrieved_state
    )

    query_context_stripped = query_context.strip()
    retrieved_context_stripped = retrieved_context.strip()
    if query_context_stripped and retrieved_context_stripped:
        features["context_only_overlap_tfidf"] = compute_tfidf_similarity(
            query_context_stripped,
            retrieved_context_stripped,
            vectorizers["context"],
        )
        features["context_only_overlap_bigram"] = compute_bigram_overlap(
            query_context_stripped,
            retrieved_context_stripped,
        )
    else:
        features["context_only_overlap_tfidf"] = None
        features["context_only_overlap_bigram"] = None

    return features


def sanitize_agent_type(value: Any) -> str:
    """Sanitize agent type for one-hot feature column matching."""
    text = str(value) if value is not None else "unknown"
    chars: List[str] = []
    for ch in text:
        chars.append(ch if (ch.isalnum() or ch == "_") else "_")
    cleaned = "".join(chars).strip("_")
    return cleaned or "unknown"


def project_features_to_vector(
    features: Dict[str, Any],
    feature_columns: List[str],
) -> Tuple[List[float], Dict[str, Any]]:
    """Project feature dict to vector by TSV column order with 0.0 fallback."""
    vector: List[float] = []
    missing_count = 0
    agent_type = sanitize_agent_type(features.get("agent_type"))

    for column in feature_columns:
        if column.startswith("agent_type__"):
            expected = column.split("agent_type__", 1)[1]
            vector.append(1.0 if agent_type == expected else 0.0)
            continue

        value = features.get(column)
        if isinstance(value, (int, float, np.integer, np.floating)):
            vector.append(float(value))
        else:
            vector.append(0.0)
            missing_count += 1

    diagnostics = {
        "feature_missing_count": int(missing_count),
    }
    return vector, diagnostics


def _coerce_embedding(value: Any) -> Optional[List[float]]:
    if isinstance(value, np.ndarray):
        arr = value.astype(float).reshape(-1)
        return arr.tolist() if arr.size > 0 else None
    if isinstance(value, (list, tuple)):
        try:
            arr = [float(v) for v in value]
        except Exception:
            return None
        return arr if arr else None
    return None


def _safe_embedding_distance(
    embed1: Optional[List[float]], embed2: Optional[List[float]]
) -> Optional[float]:
    if embed1 is None or embed2 is None:
        return None
    if len(embed1) == 0 or len(embed2) == 0:
        return None
    if len(embed1) != len(embed2):
        return None
    return compute_embedding_similarity(embed1, embed2)


def _parse_json_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _parse_guidance(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return []
    return []


def _canonicalize_steps_for_text_length(
    steps: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Normalize trajectory steps for text-length parity with legacy enriched features.

    Legacy datasets are closest to action/observation-focused chunks. Keep that
    stable subset first, and fallback to score-stripped full steps otherwise.
    """
    cleaned: List[Dict[str, Any]] = []
    for step in steps:
        if isinstance(step, dict):
            if "action" in step or "observation" in step:
                cleaned.append(
                    {
                        "action": step.get("action"),
                        "observation": step.get("observation"),
                    }
                )
            else:
                cleaned.append(
                    {key: value for key, value in step.items() if key != "score"}
                )
        else:
            cleaned.append(step)
    return cleaned


def _encode_to_list(embed_model: Any, text: str) -> List[float]:
    encoded = embed_model.encode(text)
    if isinstance(encoded, np.ndarray):
        return encoded.astype(float).reshape(-1).tolist()
    if isinstance(encoded, list):
        return [float(v) for v in encoded]
    return np.asarray(encoded, dtype=float).reshape(-1).tolist()


@dataclass
class QueryContext:
    query_text: str
    query_meta: Dict[str, Any]
    query_parts: Dict[str, str]
    query_goal_embed: List[float]
    query_state_embed: List[float]
    query_context_embed: List[float]
    query_key_embed: List[float]
    query_length: int
    step_index: Optional[int]


def build_query_context(
    query_text: str,
    query_meta: Dict[str, Any],
    embed_model: Any,
) -> QueryContext:
    """Build query-side context and embeddings once per query."""
    query_parts = parse_query_components(query_text)
    goal_text = query_parts.get("goal", "")
    state_text = query_parts.get("state", "")
    context_text = query_parts.get("context", "").strip()

    query_goal_embed = _encode_to_list(embed_model, goal_text)
    query_state_embed = _encode_to_list(embed_model, state_text)
    query_key_embed = _encode_to_list(embed_model, query_text)
    if context_text:
        query_context_embed = _encode_to_list(embed_model, context_text)
    else:
        query_context_embed = [0.0] * len(query_key_embed)

    step_index = extract_step_index(query_parts.get("progress", ""))
    return QueryContext(
        query_text=query_text,
        query_meta=dict(query_meta) if isinstance(query_meta, dict) else {},
        query_parts=query_parts,
        query_goal_embed=query_goal_embed,
        query_state_embed=query_state_embed,
        query_context_embed=query_context_embed,
        query_key_embed=query_key_embed,
        query_length=count_tokens(query_text),
        step_index=step_index,
    )


def build_candidate_features(
    query_ctx: QueryContext,
    lancedb_row: Dict[str, Any],
    retrieved_rank: int,
    distance: Optional[float],
    feature_columns: List[str],
    tfidf_vecs: Optional[Dict[str, TfidfVectorizer]],
    query_meta_override: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[float], Dict[str, Any]]:
    """Build runtime features and aligned feature vector for one candidate."""
    retrieved_goal = str(lancedb_row.get("key_raw_goal", "") or "")
    retrieved_state = str(lancedb_row.get("key_raw_state", "") or "")
    retrieved_context = str(lancedb_row.get("key_raw_context", "") or "")
    retrieved_progress = str(lancedb_row.get("key_raw_progress", "") or "")
    retrieved_raw_key = (
        f"goal: {retrieved_goal} | state: {retrieved_state} | "
        f"context: {retrieved_context} | progress: {retrieved_progress}"
    )

    query_context_text = query_ctx.query_parts.get("context", "").strip()
    retrieved_context_text = retrieved_context.strip()

    retrieved_key_embed = _coerce_embedding(lancedb_row.get("key_embed"))
    retrieved_goal_embed = _coerce_embedding(lancedb_row.get("goal_only"))
    retrieved_state_embed = _coerce_embedding(lancedb_row.get("state_only"))
    retrieved_context_embed = _coerce_embedding(lancedb_row.get("context_only"))

    embedding_similarity = _safe_embedding_distance(
        query_ctx.query_key_embed,
        retrieved_key_embed,
    )
    goal_only_similarity = _safe_embedding_distance(
        query_ctx.query_goal_embed,
        retrieved_goal_embed,
    )
    state_only_similarity = _safe_embedding_distance(
        query_ctx.query_state_embed,
        retrieved_state_embed,
    )
    if query_context_text and retrieved_context_text:
        context_only_similarity = _safe_embedding_distance(
            query_ctx.query_context_embed,
            retrieved_context_embed,
        )
    else:
        context_only_similarity = None

    text_overlap_jaccard = compute_lexical_overlap(
        query_ctx.query_text, retrieved_raw_key
    )
    goal_only_jaccard = compute_lexical_overlap(
        query_ctx.query_parts.get("goal", ""),
        retrieved_goal,
    )
    state_only_jaccard = compute_lexical_overlap(
        query_ctx.query_parts.get("state", ""),
        retrieved_state,
    )
    if query_context_text and retrieved_context_text:
        context_only_jaccard = compute_lexical_overlap(
            query_context_text, retrieved_context_text
        )
    else:
        context_only_jaccard = None

    query_overlap_ratio = compute_query_overlap_ratio(
        query_ctx.query_text, retrieved_raw_key
    )
    goal_only_query_overlap = compute_query_overlap_ratio(
        query_ctx.query_parts.get("goal", ""),
        retrieved_goal,
    )
    state_only_query_overlap = compute_query_overlap_ratio(
        query_ctx.query_parts.get("state", ""),
        retrieved_state,
    )
    if query_context_text and retrieved_context_text:
        context_only_query_overlap = compute_query_overlap_ratio(
            query_context_text,
            retrieved_context_text,
        )
    else:
        context_only_query_overlap = None

    tfidf_features: Dict[str, Optional[float]] = {
        "text_overlap_tfidf": None,
        "text_overlap_bigram": None,
        "goal_only_overlap_tfidf": None,
        "goal_only_overlap_bigram": None,
        "state_only_overlap_tfidf": None,
        "state_only_overlap_bigram": None,
        "context_only_overlap_tfidf": None,
        "context_only_overlap_bigram": None,
    }
    if tfidf_vecs is not None:
        tfidf_features = compute_all_tfidf_features(
            query=query_ctx.query_text,
            retrieved_key=retrieved_raw_key,
            query_goal=query_ctx.query_parts.get("goal", ""),
            retrieved_goal=retrieved_goal,
            query_state=query_ctx.query_parts.get("state", ""),
            retrieved_state=retrieved_state,
            query_context=query_ctx.query_parts.get("context", ""),
            retrieved_context=retrieved_context,
            vectorizers=tfidf_vecs,
        )

    task_query_meta = query_ctx.query_meta
    if isinstance(query_meta_override, dict):
        task_query_meta = query_meta_override

    query_task_name = task_query_meta.get("task_name")
    query_variation = task_query_meta.get("variation")
    if query_task_name is None:
        query_task_name = query_ctx.query_meta.get("task_name")
    if query_variation is None:
        query_variation = query_ctx.query_meta.get("variation")
    retrieved_task_name = lancedb_row.get("task_name")
    retrieved_variation = lancedb_row.get("variation_idx")
    task_match = 1.0 if str(query_task_name) == str(retrieved_task_name) else 0.0
    task_variation_match = (
        1.0
        if str(query_task_name) == str(retrieved_task_name)
        and str(query_variation) == str(retrieved_variation)
        else 0.0
    )

    guidance = _parse_guidance(lancedb_row.get("guidance"))
    trajectory_length = float(len(guidance)) if guidance else 0.0
    # Keep parity with legacy enriched features using action/observation-centric text.
    guidance_for_length = _canonicalize_steps_for_text_length(guidance)
    retrieved_text_length = (
        float(count_tokens(json.dumps(guidance_for_length)))
        if guidance
        else float(count_tokens(retrieved_raw_key))
    )
    success_flag = 1.0 if bool(lancedb_row.get("success", False)) else 0.0
    agent_type = lancedb_row.get("agent_type")

    metadata_dict = _parse_json_dict(lancedb_row.get("metadata"))
    total_steps = metadata_dict.get("total_steps")
    if isinstance(total_steps, (int, float)):
        total_steps = float(total_steps)
    else:
        total_steps = None

    retrieved_step_index = extract_step_index(retrieved_progress)
    if query_ctx.step_index is not None and retrieved_step_index is not None:
        progress_alignment_abs_diff = float(
            abs(query_ctx.step_index - retrieved_step_index)
        )
    else:
        progress_alignment_abs_diff = None

    consumer_model = task_query_meta.get("consumer_model")
    if consumer_model is None:
        consumer_model = query_ctx.query_meta.get("consumer_model")
    source_model = lancedb_row.get("agent_type")

    consumer_feature_map = resolve_model_feature_map(consumer_model)
    source_feature_map = resolve_model_feature_map(source_model)
    source_feature_map = maybe_apply_seq2seq_source_fallback(
        source_model, source_feature_map
    )

    features: Dict[str, Any] = {
        "embedding_similarity": embedding_similarity,
        "goal_only_similarity": goal_only_similarity,
        "state_only_similarity": state_only_similarity,
        "context_only_similarity": context_only_similarity,
        "text_overlap_jaccard": text_overlap_jaccard,
        "goal_only_jaccard": goal_only_jaccard,
        "state_only_jaccard": state_only_jaccard,
        "context_only_jaccard": context_only_jaccard,
        "query_overlap_ratio": query_overlap_ratio,
        "goal_only_query_overlap": goal_only_query_overlap,
        "state_only_query_overlap": state_only_query_overlap,
        "context_only_query_overlap": context_only_query_overlap,
        "text_overlap_tfidf": tfidf_features["text_overlap_tfidf"],
        "text_overlap_bigram": tfidf_features["text_overlap_bigram"],
        "goal_only_overlap_tfidf": tfidf_features["goal_only_overlap_tfidf"],
        "goal_only_overlap_bigram": tfidf_features["goal_only_overlap_bigram"],
        "state_only_overlap_tfidf": tfidf_features["state_only_overlap_tfidf"],
        "state_only_overlap_bigram": tfidf_features["state_only_overlap_bigram"],
        "context_only_overlap_tfidf": tfidf_features["context_only_overlap_tfidf"],
        "context_only_overlap_bigram": tfidf_features["context_only_overlap_bigram"],
        "task_match": task_match,
        "task_variation_match": task_variation_match,
        "trajectory_length": trajectory_length,
        "retrieved_text_length": retrieved_text_length,
        "success_flag": success_flag,
        "agent_type": agent_type,
        "consumer_source_same_model": 1.0
        if str(consumer_model) == str(source_model)
        else 0.0,
        "total_steps": total_steps,
        "source_benchmark_score": lancedb_row.get("source_benchmark_score"),
        "query_length": float(query_ctx.query_length),
        "step_index": float(query_ctx.step_index)
        if query_ctx.step_index is not None
        else None,
        "progress_alignment_abs_diff": progress_alignment_abs_diff,
        "retriever_score": float(distance)
        if isinstance(distance, (int, float))
        else None,
        "retrieved_rank": float(retrieved_rank),
        "consumer_model": consumer_model,
        "consumer_model_features_found": 1.0
        if consumer_feature_map is not None
        else 0.0,
        "source_model": source_model,
        "source_model_features_found": 1.0 if source_feature_map is not None else 0.0,
    }
    features.update(build_model_feature_bundle("consumer", consumer_feature_map))
    features.update(build_model_feature_bundle("source", source_feature_map))
    features.update(
        build_delta_feature_bundle(
            "consumer", consumer_feature_map, "source", source_feature_map
        )
    )

    vector, projection_diag = project_features_to_vector(features, feature_columns)
    diagnostics = {
        "feature_source": "runtime",
        "feature_missing_count": int(projection_diag["feature_missing_count"]),
        "feature_error_reason": None,
    }
    return features, vector, diagnostics
