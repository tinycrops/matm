#!/usr/bin/env python3
"""
Search configuration parser for LanceDB queries.

This module handles parsing of the search_config YAML structure and converts
it into LanceDB-compatible filter expressions with query logging.
"""

from typing import Dict, Any, Optional, Tuple, Set, List
import copy


# Valid diversity strategy values
VALID_DIVERSITY_STRATEGIES: Set[str] = {
    None,  # No diversity filter
    "different_task_or_variation",  # Exclude only exact match (task + variation)
    "different_task",  # Exclude same task (any variation)
    "same_task",  # Same task (all variations, including exact match)
    "same_task_different_variation",  # Same task but different variation
    "same_task_same_variation",  # Only exact match (task + variation) - most restrictive
}

VALID_RERANK_MODES: Set[str] = {
    "llm",
    "ltr",
    "cascade",
    "llm_then_ltr_with_llm_features",
}


def _parse_stage_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item).strip().lower() for item in raw if str(item).strip()]
    return [part.strip().lower() for part in str(raw).split(",") if part.strip()]


def _parse_stage_topk(raw: Any) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, int):
        return [int(raw)]
    if isinstance(raw, list):
        return [int(item) for item in raw]
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def validate_search_config(config: Dict[str, Any]) -> None:
    """
    Validate search_config structure and values.

    Args:
        config: Search configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValueError(
            f"search_config must be a dictionary, got {type(config).__name__}"
        )

    # Validate candidate_generation section
    if "candidate_generation" in config:
        cg = config["candidate_generation"]
        if not isinstance(cg, dict):
            raise ValueError(
                f"search_config.candidate_generation must be a dictionary, got {type(cg).__name__}"
            )

        # Validate top_k
        if "top_k" in cg:
            if not isinstance(cg["top_k"], int) or cg["top_k"] <= 0:
                raise ValueError(
                    f"search_config.candidate_generation.top_k must be a positive integer, got {cg['top_k']}"
                )

        # Validate rank_retrieve
        if "rank_retrieve" in cg:
            if not isinstance(cg["rank_retrieve"], int) or cg["rank_retrieve"] <= 0:
                raise ValueError(
                    f"search_config.candidate_generation.rank_retrieve must be a positive integer, got {cg['rank_retrieve']}"
                )

            # Validate rank_retrieve <= top_k (if both are present)
            if "top_k" in cg:
                if cg["rank_retrieve"] > cg["top_k"]:
                    raise ValueError(
                        f"search_config.candidate_generation.rank_retrieve ({cg['rank_retrieve']}) must be <= top_k ({cg['top_k']})\n"
                        f"rank_retrieve selects which trajectory to use from the top_k results (1-indexed).\n"
                        f"For example: rank_retrieve=1 selects the best match, rank_retrieve=2 selects the second-best, etc."
                    )

        # Validate prompt_top_n_enabled
        if "prompt_top_n_enabled" in cg and not isinstance(
            cg["prompt_top_n_enabled"], bool
        ):
            raise ValueError(
                f"search_config.candidate_generation.prompt_top_n_enabled must be a boolean, got {type(cg['prompt_top_n_enabled']).__name__}"
            )

        # Validate prompt_top_n
        if "prompt_top_n" in cg:
            if not isinstance(cg["prompt_top_n"], int) or cg["prompt_top_n"] <= 0:
                raise ValueError(
                    f"search_config.candidate_generation.prompt_top_n must be a positive integer, got {cg['prompt_top_n']}"
                )
            if (
                cg.get("prompt_top_n_enabled")
                and "top_k" in cg
                and cg["prompt_top_n"] > cg["top_k"]
            ):
                raise ValueError(
                    f"search_config.candidate_generation.prompt_top_n ({cg['prompt_top_n']}) must be <= top_k ({cg['top_k']}) when prompt_top_n_enabled=true"
                )

        # Validate query_embedding_field
        if "query_embedding_field" in cg:
            if (
                not isinstance(cg["query_embedding_field"], str)
                or not cg["query_embedding_field"]
            ):
                raise ValueError(
                    f"search_config.candidate_generation.query_embedding_field must be a non-empty string, got {cg.get('query_embedding_field')}"
                )

        # Validate filters section
        if "filters" in cg:
            filters = cg["filters"]
            if not isinstance(filters, dict):
                raise ValueError(
                    f"search_config.candidate_generation.filters must be a dictionary, got {type(filters).__name__}"
                )

            # Validate base filters
            if "base" in filters:
                base_filters = filters["base"]
                if not isinstance(base_filters, dict):
                    raise ValueError(
                        f"search_config.candidate_generation.filters.base must be a dictionary, got {type(base_filters).__name__}"
                    )

            # Validate diversity_strategy
            if "diversity_strategy" in filters:
                diversity_strategy = filters["diversity_strategy"]
                if diversity_strategy not in VALID_DIVERSITY_STRATEGIES:
                    raise ValueError(
                        f"Invalid diversity_strategy: '{diversity_strategy}'\n"
                        f"Valid values: {sorted([str(s) for s in VALID_DIVERSITY_STRATEGIES if s is not None]) + ['null/None']}\n"
                        f"  - null/None: No diversity filter - include anything\n"
                        f"  - 'different_task_or_variation': Exclude only exact match (task + variation)\n"
                        f"  - 'different_task': Exclude same task (any variation)\n"
                        f"  - 'same_task': Same task (all variations, including exact match)\n"
                        f"  - 'same_task_different_variation': Same task but different variation\n"
                        f"  - 'same_task_same_variation': Only exact match (most restrictive)"
                    )

    # Validate reranker section
    if "reranker" in config:
        reranker = config["reranker"]
        if not isinstance(reranker, dict):
            raise ValueError(
                f"search_config.reranker must be a dictionary, got {type(reranker).__name__}"
            )

        # Validate enabled
        if "enabled" in reranker:
            if not isinstance(reranker["enabled"], bool):
                raise ValueError(
                    f"search_config.reranker.enabled must be a boolean, got {type(reranker['enabled']).__name__}"
                )

        mode = str(reranker.get("mode", "llm") or "llm").strip().lower()
        if mode not in VALID_RERANK_MODES:
            raise ValueError(
                f"search_config.reranker.mode must be one of {sorted(VALID_RERANK_MODES)}, got '{mode}'"
            )

        # Validate top_k
        if "top_k" in reranker:
            if not isinstance(reranker["top_k"], int) or reranker["top_k"] <= 0:
                raise ValueError(
                    f"search_config.reranker.top_k must be a positive integer, got {reranker['top_k']}"
                )

        if "type" in reranker and not isinstance(reranker["type"], str):
            raise ValueError(
                f"search_config.reranker.type must be a string, got {type(reranker['type']).__name__}"
            )

        if (
            "mode" in reranker
            and reranker["mode"] is not None
            and not isinstance(reranker["mode"], str)
        ):
            raise ValueError(
                f"search_config.reranker.mode must be a string or null, got {type(reranker['mode']).__name__}"
            )

        if "stages" in reranker and reranker["stages"] is not None:
            stages = reranker["stages"]
            if not isinstance(stages, (str, list)):
                raise ValueError(
                    f"search_config.reranker.stages must be a string/list/null, got {type(stages).__name__}"
                )
            parsed_stages = _parse_stage_list(stages)
            if any(stage not in ("llm", "ltr") for stage in parsed_stages):
                raise ValueError(
                    f"search_config.reranker.stages only supports llm/ltr, got {parsed_stages}"
                )

        if "stage_topk" in reranker and reranker["stage_topk"] is not None:
            stage_topk = reranker["stage_topk"]
            if not isinstance(stage_topk, (int, str, list)):
                raise ValueError(
                    f"search_config.reranker.stage_topk must be int/string/list/null, got {type(stage_topk).__name__}"
                )
            values = _parse_stage_topk(stage_topk)
            if any(v < 0 for v in values):
                raise ValueError(
                    f"search_config.reranker.stage_topk values must be >= 0, got {values}"
                )

        if (
            "model" in reranker
            and reranker["model"] is not None
            and not isinstance(reranker["model"], str)
        ):
            raise ValueError(
                f"search_config.reranker.model must be a string or null, got {type(reranker['model']).__name__}"
            )

        if (
            "provider" in reranker
            and reranker["provider"] is not None
            and not isinstance(reranker["provider"], str)
        ):
            raise ValueError(
                f"search_config.reranker.provider must be a string or null, got {type(reranker['provider']).__name__}"
            )

        if (
            "method" in reranker
            and reranker["method"] is not None
            and not isinstance(reranker["method"], str)
        ):
            raise ValueError(
                f"search_config.reranker.method must be a string or null, got {type(reranker['method']).__name__}"
            )

        if "feature_source" in reranker and reranker["feature_source"] is not None:
            if not isinstance(reranker["feature_source"], str):
                raise ValueError(
                    f"search_config.reranker.feature_source must be a string or null, got {type(reranker['feature_source']).__name__}"
                )
            if str(reranker["feature_source"]).lower() not in {
                "runtime",
                "json",
                "hybrid",
            }:
                raise ValueError(
                    f"search_config.reranker.feature_source must be runtime/json/hybrid, got {reranker['feature_source']}"
                )

        if (
            "ltr_model_file" in reranker
            and reranker["ltr_model_file"] is not None
            and not isinstance(reranker["ltr_model_file"], str)
        ):
            raise ValueError(
                f"search_config.reranker.ltr_model_file must be a string or null, got {type(reranker['ltr_model_file']).__name__}"
            )

        if (
            "ltr_model_type" in reranker
            and reranker["ltr_model_type"] is not None
            and not isinstance(reranker["ltr_model_type"], str)
        ):
            raise ValueError(
                f"search_config.reranker.ltr_model_type must be a string or null, got {type(reranker['ltr_model_type']).__name__}"
            )

        if (
            "feature_tsv" in reranker
            and reranker["feature_tsv"] is not None
            and not isinstance(reranker["feature_tsv"], str)
        ):
            raise ValueError(
                f"search_config.reranker.feature_tsv must be a string or null, got {type(reranker['feature_tsv']).__name__}"
            )

        if (
            "missing_feature_policy" in reranker
            and reranker["missing_feature_policy"] is not None
        ):
            if not isinstance(reranker["missing_feature_policy"], str):
                raise ValueError(
                    f"search_config.reranker.missing_feature_policy must be a string or null, got {type(reranker['missing_feature_policy']).__name__}"
                )
            if str(reranker["missing_feature_policy"]).lower() not in {
                "min",
                "zero",
                "baseline",
            }:
                raise ValueError(
                    f"search_config.reranker.missing_feature_policy must be min/zero/baseline, got {reranker['missing_feature_policy']}"
                )

        if "normalize_features" in reranker and not isinstance(
            reranker["normalize_features"], bool
        ):
            raise ValueError(
                f"search_config.reranker.normalize_features must be a boolean, got {type(reranker['normalize_features']).__name__}"
            )

        if (
            "tfidf_base_dir" in reranker
            and reranker["tfidf_base_dir"] is not None
            and not isinstance(reranker["tfidf_base_dir"], str)
        ):
            raise ValueError(
                f"search_config.reranker.tfidf_base_dir must be a string or null, got {type(reranker['tfidf_base_dir']).__name__}"
            )

        if (
            "model_base_dir" in reranker
            and reranker["model_base_dir"] is not None
            and not isinstance(reranker["model_base_dir"], str)
        ):
            raise ValueError(
                f"search_config.reranker.model_base_dir must be a string or null, got {type(reranker['model_base_dir']).__name__}"
            )

        if (
            "model_base_dir_llm_features" in reranker
            and reranker["model_base_dir_llm_features"] is not None
            and not isinstance(reranker["model_base_dir_llm_features"], str)
        ):
            raise ValueError(
                f"search_config.reranker.model_base_dir_llm_features must be a string or null, got {type(reranker['model_base_dir_llm_features']).__name__}"
            )

        if "filters" in reranker:
            raise ValueError(
                "search_config.reranker.filters has been removed; "
                "state_consistency is no longer supported"
            )

        if (
            "api_key_env" in reranker
            and reranker["api_key_env"] is not None
            and not isinstance(reranker["api_key_env"], str)
        ):
            raise ValueError(
                f"search_config.reranker.api_key_env must be a string or null, got {type(reranker['api_key_env']).__name__}"
            )

        if (
            "api_key_file" in reranker
            and reranker["api_key_file"] is not None
            and not isinstance(reranker["api_key_file"], str)
        ):
            raise ValueError(
                f"search_config.reranker.api_key_file must be a string or null, got {type(reranker['api_key_file']).__name__}"
            )

        if "debug_enabled" in reranker and not isinstance(
            reranker["debug_enabled"], bool
        ):
            raise ValueError(
                f"search_config.reranker.debug_enabled must be a boolean, got {type(reranker['debug_enabled']).__name__}"
            )

        if (
            "debug_log_path" in reranker
            and reranker["debug_log_path"] is not None
            and not isinstance(reranker["debug_log_path"], str)
        ):
            raise ValueError(
                f"search_config.reranker.debug_log_path must be a string or null, got {type(reranker['debug_log_path']).__name__}"
            )

        if "debug_top_n" in reranker and reranker["debug_top_n"] is not None:
            if (
                not isinstance(reranker["debug_top_n"], int)
                or reranker["debug_top_n"] <= 0
            ):
                raise ValueError(
                    f"search_config.reranker.debug_top_n must be a positive integer or null, got {reranker['debug_top_n']}"
                )

        if (
            "prompt_template" in reranker
            and reranker["prompt_template"] is not None
            and not isinstance(reranker["prompt_template"], str)
        ):
            raise ValueError(
                f"search_config.reranker.prompt_template must be a string or null, got {type(reranker['prompt_template']).__name__}"
            )

        if "max_pairs" in reranker and reranker["max_pairs"] is not None:
            if not isinstance(reranker["max_pairs"], int) or reranker["max_pairs"] <= 0:
                raise ValueError(
                    f"search_config.reranker.max_pairs must be a positive integer or null, got {reranker['max_pairs']}"
                )

        if "batch_size" in reranker:
            if (
                not isinstance(reranker["batch_size"], int)
                or reranker["batch_size"] <= 0
            ):
                raise ValueError(
                    f"search_config.reranker.batch_size must be a positive integer, got {reranker['batch_size']}"
                )

        if "temperature" in reranker and reranker["temperature"] is not None:
            if not isinstance(reranker["temperature"], (int, float)):
                raise ValueError(
                    f"search_config.reranker.temperature must be a number or null, got {type(reranker['temperature']).__name__}"
                )

        if "timeout_s" in reranker and reranker["timeout_s"] is not None:
            if not isinstance(reranker["timeout_s"], int) or reranker["timeout_s"] <= 0:
                raise ValueError(
                    f"search_config.reranker.timeout_s must be a positive integer or null, got {reranker['timeout_s']}"
                )

        if "candidate_text_fields" in reranker:
            fields = reranker["candidate_text_fields"]
            if not isinstance(fields, list) or not all(
                isinstance(f, str) for f in fields
            ):
                raise ValueError(
                    "search_config.reranker.candidate_text_fields must be a list of strings"
                )

        if reranker.get("enabled"):
            require_llm = False
            if mode == "llm":
                require_llm = True
            elif mode == "cascade":
                stages = _parse_stage_list(reranker.get("stages", "llm,ltr"))
                if not stages:
                    stages = ["llm", "ltr"]
                stage_topk = _parse_stage_topk(reranker.get("stage_topk"))
                if not stage_topk:
                    stage_topk = [int(reranker.get("top_k", 20)) for _ in stages]
                elif len(stage_topk) == 1 and len(stages) > 1:
                    stage_topk = stage_topk * len(stages)
                require_llm = any(
                    stage == "llm" and int(topk) > 0
                    for stage, topk in zip(stages, stage_topk)
                )
            elif mode == "llm_then_ltr_with_llm_features":
                require_llm = True

            if require_llm:
                if not reranker.get("model"):
                    raise ValueError(
                        "search_config.reranker.model is required when llm stage is enabled"
                    )
                if not reranker.get("provider"):
                    raise ValueError(
                        "search_config.reranker.provider is required when llm stage is enabled"
                    )


def merge_search_config(user_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Merge user-provided search_config with default values and validate.

    Args:
        user_config: Optional user-provided search configuration

    Returns:
        Merged search configuration with defaults filled in

    Raises:
        ValueError: If user_config is invalid
    """
    # Default configuration structure
    default_config = {
        "candidate_generation": {
            "top_k": 100,
            "rank_retrieve": 1,
            "prompt_top_n_enabled": False,
            "prompt_top_n": 1,
            "query_embedding_field": "key_embed",
            "filters": {},
        },
        "reranker": {
            "enabled": False,
            "type": "llm_rankers_pairwise",
            "mode": "llm",
            "stages": "llm,ltr",
            "stage_topk": "20,20",
            "model": None,
            "provider": None,
            "method": "allpair",
            "api_key_env": None,
            "api_key_file": None,
            "prompt_template": None,
            "debug_enabled": False,
            "debug_log_path": None,
            "debug_top_n": None,
            "top_k": 20,
            "max_pairs": None,
            "batch_size": 8,
            "temperature": 0.0,
            "timeout_s": 60,
            "feature_source": "runtime",
            "ltr_model_file": None,
            "ltr_model_type": None,
            "feature_tsv": None,
            "missing_feature_policy": "min",
            "normalize_features": False,
            "tfidf_base_dir": "traj_retrieval/evaluation/ltr_data",
            "model_base_dir": "ltr/ltr_models",
            "model_base_dir_llm_features": "ltr/ltr_models_llm_features",
            "candidate_text_fields": [
                "key_raw_goal",
                "key_raw_state",
                "key_raw_context",
                "key_raw_progress",
            ],
        },
    }

    # If no user config provided, return defaults
    if user_config is None:
        return default_config

    # Validate user config before merging
    validate_search_config(user_config)

    # Deep merge: start with defaults, then update with user config
    # Use a deep copy to avoid modifying defaults
    merged = copy.deepcopy(default_config)

    # Merge candidate_generation
    if "candidate_generation" in user_config:
        user_cg = user_config["candidate_generation"]
        # Merge top-level keys
        for key in [
            "top_k",
            "rank_retrieve",
            "prompt_top_n_enabled",
            "prompt_top_n",
            "query_embedding_field",
        ]:
            if key in user_cg:
                merged["candidate_generation"][key] = user_cg[key]

        # Deep merge filters if present
        if "filters" in user_cg:
            user_filters = user_cg["filters"]
            # Merge filters dict
            merged["candidate_generation"]["filters"].update(user_filters)

            # Deep merge base filters if present
            if "base" in user_filters:
                if "base" not in merged["candidate_generation"]["filters"]:
                    merged["candidate_generation"]["filters"]["base"] = {}
                merged["candidate_generation"]["filters"]["base"].update(
                    user_filters["base"]
                )

    # Merge reranker
    if "reranker" in user_config:
        user_reranker = user_config["reranker"]
        # Merge top-level keys
        for key in [
            "enabled",
            "type",
            "mode",
            "stages",
            "stage_topk",
            "model",
            "provider",
            "method",
            "api_key_env",
            "api_key_file",
            "prompt_template",
            "debug_enabled",
            "debug_log_path",
            "debug_top_n",
            "top_k",
            "max_pairs",
            "batch_size",
            "temperature",
            "timeout_s",
            "feature_source",
            "ltr_model_file",
            "ltr_model_type",
            "feature_tsv",
            "missing_feature_policy",
            "normalize_features",
            "tfidf_base_dir",
            "model_base_dir",
            "model_base_dir_llm_features",
            "candidate_text_fields",
        ]:
            if key in user_reranker:
                merged["reranker"][key] = user_reranker[key]
    return merged


def build_lancedb_filter(
    filters_config: Dict[str, Any], context: Dict[str, Any]
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Build LanceDB filter string from config and runtime context.

    Config structure:
        filters:
          base:
            success: true
            version: 1
          diversity_strategy: "different_task_or_variation"

    Args:
        filters_config: Dictionary of filter specifications from YAML config
        context: Runtime context with dynamic values (task_name, variation_idx)

    Returns:
        Tuple of (filter_string, debug_info)
        - filter_string: SQL-like filter string for LanceDB
        - debug_info: Dictionary with parsing details for logging

    Raises:
        ValueError: If diversity_strategy is invalid or required context is missing
    """
    debug_info = {
        "config": filters_config,
        "context": {
            k: v for k, v in context.items() if k in ["task_name", "variation_idx"]
        },
        "base_conditions": [],
        "diversity_strategy": None,
        "diversity_condition": None,
        "final_query": None,
    }

    all_conditions = []

    # 1. Parse base filters
    base_filters = filters_config.get("base", {})
    if base_filters:
        for field, value in base_filters.items():
            if value is None:
                continue

            # Build condition based on value type
            if isinstance(value, bool):
                condition = f"{field} = {str(value).lower()}"
            elif isinstance(value, (int, float)):
                condition = f"{field} = {value}"
            elif isinstance(value, str):
                escaped_value = value.replace("'", "''")
                condition = f"{field} = '{escaped_value}'"
            else:
                condition = f"{field} = {repr(value)}"

            all_conditions.append(condition)
            debug_info["base_conditions"].append(
                {"field": field, "value": value, "condition": condition}
            )

    # 2. Apply diversity strategy
    diversity_strategy = filters_config.get("diversity_strategy")

    # Handle null/None/empty string as "no diversity filter"
    if (
        diversity_strategy is None
        or diversity_strategy == ""
        or diversity_strategy == "any"
    ):
        # No diversity filter - include anything (only base filters apply)
        debug_info["diversity_strategy"] = None
    elif diversity_strategy:
        # Validate diversity strategy
        if diversity_strategy not in VALID_DIVERSITY_STRATEGIES:
            raise ValueError(
                f"Invalid diversity_strategy: '{diversity_strategy}'\n"
                f"Valid values: {sorted([str(s) for s in VALID_DIVERSITY_STRATEGIES if s is not None]) + ['null/None']}\n"
                f"  - null/None: No diversity filter - include anything\n"
                f"  - 'different_task_or_variation': Exclude only exact match (task + variation)\n"
                f"  - 'different_task': Exclude same task (any variation)\n"
                f"  - 'same_task': Same task (all variations, including exact match)\n"
                f"  - 'same_task_different_variation': Same task but different variation\n"
                f"  - 'same_task_same_variation': Only exact match (most restrictive)"
            )

        debug_info["diversity_strategy"] = diversity_strategy

        task_name = context.get("task_name")
        variation_idx = context.get("variation_idx")

        if diversity_strategy == "different_task_or_variation":
            # Anything other than same task AND same variation
            # SQL: NOT (task_name = X AND variation_idx = Y)
            # Equivalent to: (task_name != X) OR (variation_idx != Y)
            if task_name is None or variation_idx is None:
                raise ValueError(
                    f"diversity_strategy '{diversity_strategy}' requires both 'task_name' and 'variation_idx' in context.\n"
                    f"Got: task_name={task_name}, variation_idx={variation_idx}"
                )

            escaped_task_name = task_name.replace("'", "''")
            # Convert variation_idx to string and escape for SQL (it's stored as string in DB)
            escaped_variation_idx = str(variation_idx).replace("'", "''")
            task_neq = f"task_name != '{escaped_task_name}'"
            var_neq = f"variation_idx != '{escaped_variation_idx}'"
            diversity_filter = f"({task_neq} OR {var_neq})"

            all_conditions.append(diversity_filter)
            debug_info["diversity_condition"] = diversity_filter

        elif diversity_strategy == "different_task":
            # Anything other than same task (any variation)
            # SQL: task_name != X
            if task_name is None:
                raise ValueError(
                    f"diversity_strategy '{diversity_strategy}' requires 'task_name' in context.\n"
                    f"Got: task_name={task_name}"
                )

            escaped_task_name = task_name.replace("'", "''")
            diversity_filter = f"task_name != '{escaped_task_name}'"

            all_conditions.append(diversity_filter)
            debug_info["diversity_condition"] = diversity_filter

        elif diversity_strategy == "same_task":
            # Same task (all variations, including exact match)
            # SQL: task_name = X
            if task_name is None:
                raise ValueError(
                    f"diversity_strategy '{diversity_strategy}' requires 'task_name' in context.\n"
                    f"Got: task_name={task_name}"
                )

            escaped_task_name = task_name.replace("'", "''")
            diversity_filter = f"task_name = '{escaped_task_name}'"

            all_conditions.append(diversity_filter)
            debug_info["diversity_condition"] = diversity_filter

        elif diversity_strategy == "same_task_different_variation":
            # Same task but different variation
            # SQL: task_name = X AND variation_idx != Y
            if task_name is None or variation_idx is None:
                raise ValueError(
                    f"diversity_strategy '{diversity_strategy}' requires both 'task_name' and 'variation_idx' in context.\n"
                    f"Got: task_name={task_name}, variation_idx={variation_idx}"
                )

            escaped_task_name = task_name.replace("'", "''")
            # Convert variation_idx to string and escape for SQL (it's stored as string in DB)
            escaped_variation_idx = str(variation_idx).replace("'", "''")
            task_eq = f"task_name = '{escaped_task_name}'"
            var_neq = f"variation_idx != '{escaped_variation_idx}'"
            diversity_filter = f"({task_eq} AND {var_neq})"

            all_conditions.append(diversity_filter)
            debug_info["diversity_condition"] = diversity_filter

        elif diversity_strategy == "same_task_same_variation":
            # Only exact match (task + variation) - most restrictive
            # SQL: task_name = X AND variation_idx = Y
            if task_name is None or variation_idx is None:
                raise ValueError(
                    f"diversity_strategy '{diversity_strategy}' requires both 'task_name' and 'variation_idx' in context.\n"
                    f"Got: task_name={task_name}, variation_idx={variation_idx}"
                )

            escaped_task_name = task_name.replace("'", "''")
            # Convert variation_idx to string and escape for SQL (it's stored as string in DB)
            escaped_variation_idx = str(variation_idx).replace("'", "''")
            task_eq = f"task_name = '{escaped_task_name}'"
            var_eq = f"variation_idx = '{escaped_variation_idx}'"
            diversity_filter = f"({task_eq} AND {var_eq})"

            all_conditions.append(diversity_filter)
            debug_info["diversity_condition"] = diversity_filter

    # 3. Join all conditions with AND
    if not all_conditions:
        debug_info["final_query"] = None
        return None, debug_info

    final_query = " AND ".join(all_conditions)
    debug_info["final_query"] = final_query

    return final_query, debug_info


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("Search Config Parser - Example Usage")
    print("=" * 80)

    context = {"task_name": "boil", "variation_idx": 5}

    base_filters = {"base": {"success": True, "version": 1}}

    # Example 1: No diversity filter (include anything)
    print("\nExample 1: No diversity strategy (null)")
    filters1 = {**base_filters, "diversity_strategy": None}
    result1, debug1 = build_lancedb_filter(filters1, context)
    print(f"Strategy: null (no diversity filter)")
    print(f"Context: {context}")
    print(f"SQL: {result1}")
    print(f"Meaning: Includes ANY trajectory (only base filters apply)")

    # Example 2: Different task OR different variation
    print("\nExample 2: different_task_or_variation")
    filters2 = {**base_filters, "diversity_strategy": "different_task_or_variation"}
    result2, debug2 = build_lancedb_filter(filters2, context)
    print(f"Strategy: {filters2['diversity_strategy']}")
    print(f"Context: {context}")
    print(f"SQL: {result2}")
    print(f"Meaning: Excludes only when task='boil' AND variation=5")
    print(f"         Includes: different tasks OR different variations")

    # Example 3: Different task
    print("\nExample 3: different_task")
    filters3 = {**base_filters, "diversity_strategy": "different_task"}
    result3, debug3 = build_lancedb_filter(filters3, context)
    print(f"Strategy: {filters3['diversity_strategy']}")
    print(f"Context: {context}")
    print(f"SQL: {result3}")
    print(f"Meaning: Excludes task='boil' (any variation)")
    print(f"         Includes: any other task")

    # Example 4: Same task (including exact match)
    print("\nExample 4: same_task")
    filters4 = {**base_filters, "diversity_strategy": "same_task"}
    result4, debug4 = build_lancedb_filter(filters4, context)
    print(f"Strategy: {filters4['diversity_strategy']}")
    print(f"Context: {context}")
    print(f"SQL: {result4}")
    print(f"Meaning: Includes only task='boil' (all variations, including 5)")
    print(f"         Includes: (boil, 5), (boil, 0-4,6+)")

    # Example 5: Same task but different variation
    print("\nExample 5: same_task_different_variation")
    filters5 = {**base_filters, "diversity_strategy": "same_task_different_variation"}
    result5, debug5 = build_lancedb_filter(filters5, context)
    print(f"Strategy: {filters5['diversity_strategy']}")
    print(f"Context: {context}")
    print(f"SQL: {result5}")
    print(f"Meaning: Includes only task='boil' with variation != 5")
    print(f"         Includes: (boil, 0-4,6+), (boil, any other variation)")
    print(f"         Excludes: (boil, 5) - exact match")
