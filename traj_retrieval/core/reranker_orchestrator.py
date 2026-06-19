from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .llm_allpair_feature_utils import (
    LLM_ALLPAIR_FEATURE_NAMES,
    zero_llm_allpair_features,
)
from .llm_reranker import LLMReranker
from .llm_retrieval_allpair_pipeline import (
    build_candidate_from_row as build_pairwise_candidate_from_row,
    compute_allpair_for_candidates,
)

KNOWN_STAGE_NAMES: Tuple[str, str] = ("llm", "ltr")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TFIDF_BASE_DIR = REPO_ROOT / "traj_retrieval" / "evaluation" / "ltr_data"
DEFAULT_MODEL_BASE_DIR = REPO_ROOT / "ltr" / "ltr_models"
DEFAULT_MODEL_BASE_DIR_LLM_FEATURES = (
    REPO_ROOT / "ltr" / "ltr_models_llm_features"
)
DEFAULT_FEATURE_TSV_BASE_DIR = REPO_ROOT / "ltr" / "data_out"
DEFAULT_FEATURE_TSV_BASE_DIR_LLM_FEATURES = (
    REPO_ROOT / "ltr" / "data_out_llm_features_new"
)
DEFAULT_LTR_MODEL_TYPE = "lambdamart"


def doc_id_from_text(dataset: str, raw_key: str) -> str:
    digest = hashlib.md5(raw_key.encode("utf-8")).hexdigest()
    return f"{dataset}_{digest}"


def get_row_doc_id(dataset: str, row: Dict[str, Any]) -> Optional[str]:
    doc_id = row.get("doc_id")
    if isinstance(doc_id, str) and doc_id:
        return doc_id

    raw_key = row.get("raw_key")
    if isinstance(raw_key, str) and raw_key:
        return doc_id_from_text(dataset, raw_key)

    goal = str(row.get("key_raw_goal", "") or "").strip()
    state = str(row.get("key_raw_state", "") or "").strip()
    context = str(row.get("key_raw_context", "") or "").strip()
    progress = str(row.get("key_raw_progress", "") or "").strip()
    if goal or state or context or progress:
        reconstructed = (
            f"goal: {goal} | state: {state} | context: {context} | progress: {progress}"
        )
        return doc_id_from_text(dataset, reconstructed)
    return None


def get_row_raw_key(row: Dict[str, Any]) -> Optional[str]:
    raw_key = row.get("raw_key")
    if isinstance(raw_key, str) and raw_key.strip():
        return raw_key.strip()
    goal = str(row.get("key_raw_goal", "") or "").strip()
    state = str(row.get("key_raw_state", "") or "").strip()
    context = str(row.get("key_raw_context", "") or "").strip()
    progress = str(row.get("key_raw_progress", "") or "").strip()
    if goal or state or context or progress:
        return (
            f"goal: {goal} | state: {state} | context: {context} | progress: {progress}"
        )
    return None


def ensure_1d_scores(values: Any) -> Any:
    import numpy as np

    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return np.asarray([float(array)], dtype=float)
    return array.reshape(-1)


def load_feature_columns(feature_tsv: Path) -> List[str]:
    if not feature_tsv.exists():
        raise FileNotFoundError(f"Feature TSV not found: {feature_tsv}")
    with feature_tsv.open("r", encoding="utf-8") as file_obj:
        reader = csv.reader(file_obj, delimiter="\t")
        header = next(reader, None)
    if not header:
        raise ValueError(f"Empty feature TSV: {feature_tsv}")
    ignore = {"qid", "label", "docid", "folder_nativeid"}
    return [column for column in header if column not in ignore]


def parse_stages(raw: Any) -> List[str]:
    if isinstance(raw, list):
        parts = [str(part).strip().lower() for part in raw if str(part).strip()]
    else:
        parts = [part.strip().lower() for part in str(raw).split(",") if part.strip()]

    if not parts:
        raise ValueError("reranker.stages must include at least one stage")
    if len(parts) > 2:
        raise ValueError("reranker.stages supports at most 2 stages")

    for part in parts:
        if part not in KNOWN_STAGE_NAMES:
            raise ValueError(
                f"Unsupported rerank stage {part}. Choose from {KNOWN_STAGE_NAMES}."
            )

    if len(set(parts)) != len(parts):
        raise ValueError("reranker.stages cannot contain duplicate stage names")
    return parts


def parse_stage_topk(raw: Any) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, int):
        if raw < 0:
            raise ValueError("reranker.stage_topk must be >= 0")
        return [int(raw)]
    if isinstance(raw, (list, tuple)):
        values: List[int] = []
        for item in raw:
            value = int(item)
            if value < 0:
                raise ValueError("reranker.stage_topk entries must be >= 0")
            values.append(value)
        return values

    items = [part.strip() for part in str(raw).split(",") if part.strip()]
    values = []
    for item in items:
        value = int(item)
        if value < 0:
            raise ValueError("reranker.stage_topk entries must be >= 0")
        values.append(value)
    return values


def resolve_stage_topks(
    raw_stage_topk: Any,
    stages: Sequence[str],
    default_topk: int,
    retrieval_top_k: int,
) -> List[int]:
    values = parse_stage_topk(raw_stage_topk)
    if not values:
        values = [default_topk for _ in stages]
    elif len(values) == 1 and len(stages) > 1:
        values = values * len(stages)
    elif len(values) != len(stages):
        raise ValueError(
            "reranker.stage_topk length must be 1 or equal to number of stages"
        )

    return [int(min(v, retrieval_top_k)) for v in values]


def doc_key(candidate: Dict[str, Any], fallback_idx: int) -> str:
    doc_id = candidate.get("docid")
    if isinstance(doc_id, str) and doc_id:
        return doc_id
    return f"__fallback_{fallback_idx}"


def candidate_docids(candidates: Sequence[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for idx, candidate in enumerate(candidates):
        out.append(doc_key(candidate, idx))
    return out


def dedupe_first_seen(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        key = doc_key(candidate, idx)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def merge_stage_order(
    old_candidates: Sequence[Dict[str, Any]],
    reranked_prefix: Sequence[Dict[str, Any]],
    topn: int,
) -> List[Dict[str, Any]]:
    if topn <= 0:
        return list(old_candidates)
    prefix = list(old_candidates[:topn])
    tail = list(old_candidates[topn:])
    merged = list(reranked_prefix) + prefix + tail
    return dedupe_first_seen(merged)


def build_reranked_prefix_from_output(
    reranker_output: Any,
    original_prefix: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not isinstance(reranker_output, list):
        raise ValueError("LLM reranker returned non-list output")

    prefix_by_docid: Dict[str, Dict[str, Any]] = {}
    for candidate in original_prefix:
        doc_id = candidate.get("docid")
        if isinstance(doc_id, str) and doc_id:
            prefix_by_docid[doc_id] = candidate

    ordered: List[Dict[str, Any]] = []
    for item in reranker_output:
        doc_id: Optional[str] = None
        if isinstance(item, dict):
            value = item.get("docid")
            if isinstance(value, str) and value:
                doc_id = value
        else:
            value = getattr(item, "docid", None)
            if isinstance(value, str) and value:
                doc_id = value
        if doc_id is None:
            continue
        candidate = prefix_by_docid.get(doc_id)
        if candidate is not None:
            ordered.append(candidate)
    return ordered


class UnifiedRerankerOrchestrator:
    """
    Unified end-to-end reranker orchestrator.

    Supported modes:
    - llm
    - ltr
    - cascade
    - llm_then_ltr_with_llm_features
    """

    def __init__(
        self,
        dataset: str,
        retrieval_top_k: int,
        reranker_config: Dict[str, Any],
        lancedb_uri: Optional[str] = None,
    ) -> None:
        self.dataset = str(dataset or "dataset")
        self.retrieval_top_k = int(max(1, retrieval_top_k))
        self.reranker_config = dict(reranker_config or {})
        self.lancedb_uri = lancedb_uri

        self._llm_reranker: Optional[LLMReranker] = None
        self._llm_stage_ranker: Any = None

        self._ltr_model: Any = None
        self._ltr_model_type: Optional[str] = None
        self._feature_columns: Optional[List[str]] = None
        self._tfidf_vectorizers: Any = None
        self._tfidf_unavailable = False
        self._tfidf_unavailable_reason: Optional[str] = None
        self._embedder: Any = None
        self._ltr_modules_loaded = False
        self._ltr_runtime = None
        self._ltr_data_loader = None
        self._ltr_model_map: Optional[Dict[str, Any]] = None

        self._webarena_source_score_cache: Dict[
            Tuple[str, str, str], Optional[float]
        ] = {}

    def rerank(
        self,
        query_text: str,
        candidates: Sequence[Dict[str, Any]],
        task_name: Optional[str],
        variation_idx: Any,
        current_step: int,
        current_reward: float,
    ) -> Dict[str, Any]:
        if not candidates:
            return {
                "candidates": [],
                "applied": False,
                "metadata": {
                    "mode": self.reranker_config.get("mode", "llm"),
                    "stages": [],
                    "stage_topk": [],
                    "rerank_errors_stage1": 0,
                    "rerank_errors_stage2": 0,
                },
            }

        mode, stages, stage_topks = self._resolve_mode_and_stages()
        self._validate_requirements(mode=mode, stages=stages, stage_topks=stage_topks)

        # Normalize raw LanceDB rows into one candidate schema so later stages can
        # attach ranks, feature vectors, and doc ids without branching per mode.
        initial_candidates = self._build_initial_candidates(
            rows=list(candidates),
            use_pairwise_builder=(mode == "llm_then_ltr_with_llm_features"),
        )

        current_candidates = list(initial_candidates)
        stage_changed = [0, 0]
        stage_error_counts = [0, 0]
        stage_errors: List[Dict[str, Any]] = []

        ltr_stats: Dict[str, int] = {
            "top_k": int(self.retrieval_top_k),
            "retrieved_runtime_attempts": 0,
            "retrieved_with_runtime_features": 0,
            "retrieved_feature_failures": 0,
            "retrieved_json_fallback_used": 0,
            "retrieved_task_meta_overrides": 0,
            "retrieved_task_meta_fallback_to_query": 0,
            "retrieved_with_features": 0,
            "retrieved_without_features": 0,
        }
        llm_feature_stats: Dict[str, Any] = {
            "injected_docs": 0,
            "missing_docs": 0,
            "values": {name: [] for name in LLM_ALLPAIR_FEATURE_NAMES},
        }
        llm_allpair_n_hist = Counter()
        llm_allpair_lt20_queries = 0
        query_llm_feature_map: Dict[str, Dict[str, float]] = {}

        has_ltr_stage = any(
            name == "ltr" and topk > 0 for name, topk in zip(stages, stage_topks)
        )
        if has_ltr_stage:
            feature_columns = self._get_feature_columns(mode=mode)
            if mode == "llm_then_ltr_with_llm_features":
                missing = [
                    name
                    for name in LLM_ALLPAIR_FEATURE_NAMES
                    if name not in feature_columns
                ]
                if missing:
                    raise ValueError(
                        "Feature TSV is missing required LLM allpair features: "
                        f"{missing}"
                    )
            llm_feature_indices = {
                name: feature_columns.index(name)
                for name in LLM_ALLPAIR_FEATURE_NAMES
                if name in feature_columns
            }
        else:
            feature_columns = []
            llm_feature_indices = {}

        feature_source = str(
            self.reranker_config.get("feature_source", "runtime")
        ).lower()

        query_ctx = None
        if has_ltr_stage and feature_source in {"runtime", "hybrid"}:
            # Runtime feature extraction reuses parsed query fields and embeddings
            # for every candidate, so build that query-side cache once here.
            query_ctx = self._build_query_context(
                query_text=query_text,
                task_name=task_name,
                variation_idx=variation_idx,
            )

        for stage_index, (stage_name, stage_topk) in enumerate(
            zip(stages, stage_topks), start=1
        ):
            # Each stage reranks only the configured prefix; merge_stage_order keeps
            # the untouched tail in upstream retrieval order.
            before_docids = candidate_docids(current_candidates)
            stage_error: Optional[str] = None

            if stage_name == "llm":
                if mode == "llm_then_ltr_with_llm_features":
                    (
                        updated_candidates,
                        query_llm_feature_map,
                        llm_stage_details,
                        stage_error,
                    ) = self._apply_llm_stage_with_features(
                        query_text=query_text,
                        candidates=current_candidates,
                        stage_topk=int(stage_topk),
                    )
                    allpair_n = int(llm_stage_details.get("n", 0))
                    llm_allpair_n_hist[allpair_n] += 1
                    if allpair_n < 20:
                        llm_allpair_lt20_queries += 1
                else:
                    updated_candidates, stage_error = self._apply_llm_stage(
                        query_text=query_text,
                        candidates=current_candidates,
                        stage_topk=int(stage_topk),
                    )
            elif stage_name == "ltr":
                updated_candidates, stage_error = self._apply_ltr_stage(
                    mode=mode,
                    query_ctx=query_ctx,
                    candidates=current_candidates,
                    stage_topk=int(stage_topk),
                    feature_columns=feature_columns,
                    feature_source=feature_source,
                    missing_feature_policy=str(
                        self.reranker_config.get("missing_feature_policy", "min")
                    ).lower(),
                    normalize_features=bool(
                        self.reranker_config.get("normalize_features", False)
                    ),
                    llm_feature_map=query_llm_feature_map,
                    llm_feature_indices=llm_feature_indices,
                    llm_feature_stats=llm_feature_stats,
                    ltr_stats=ltr_stats,
                    inject_llm_features=(mode == "llm_then_ltr_with_llm_features"),
                )
            else:
                raise ValueError(f"Unsupported stage name: {stage_name}")

            if stage_error is not None:
                if stage_index <= 2:
                    stage_error_counts[stage_index - 1] += 1
                stage_errors.append(
                    {
                        "stage": int(stage_index),
                        "name": stage_name,
                        "error": stage_error,
                    }
                )
                updated_candidates = list(current_candidates)

            after_docids = candidate_docids(updated_candidates)
            if before_docids != after_docids and stage_index <= 2:
                stage_changed[stage_index - 1] += 1

            current_candidates = updated_candidates
            rank_field = f"stage{stage_index}_rank"
            for rank_value, candidate in enumerate(current_candidates, start=1):
                candidate[rank_field] = int(rank_value)

        for final_rank, candidate in enumerate(current_candidates, start=1):
            candidate["final_rank"] = int(final_rank)

        llm_runtime: Dict[str, Any] = {}
        if any(name == "llm" and topk > 0 for name, topk in zip(stages, stage_topks)):
            llm_reranker = self._get_llm_reranker()
            try:
                llm_runtime = llm_reranker.get_runtime_stats()
            except Exception:
                llm_runtime = {}

        runtime_feature_coverage = {
            "retrieved_runtime_attempts": int(ltr_stats["retrieved_runtime_attempts"]),
            "retrieved_with_runtime_features": int(
                ltr_stats["retrieved_with_runtime_features"]
            ),
            "retrieved_feature_failures": int(ltr_stats["retrieved_feature_failures"]),
            "retrieved_json_fallback_used": int(
                ltr_stats["retrieved_json_fallback_used"]
            ),
            "retrieved_with_features": int(ltr_stats["retrieved_with_features"]),
            "retrieved_without_features": int(ltr_stats["retrieved_without_features"]),
            "coverage": (
                float(ltr_stats["retrieved_with_runtime_features"])
                / float(ltr_stats["retrieved_runtime_attempts"])
                if ltr_stats["retrieved_runtime_attempts"] > 0
                else None
            ),
        }

        ranking_rows: List[Dict[str, Any]] = []
        for candidate in current_candidates:
            ranking_rows.append(
                {
                    "docid": candidate.get("docid"),
                    "retrieved_rank": candidate.get("retrieved_rank"),
                    "stage1_rank": candidate.get("stage1_rank"),
                    "stage2_rank": candidate.get("stage2_rank"),
                    "final_rank": candidate.get("final_rank"),
                    "distance": candidate.get("distance"),
                }
            )

        metadata: Dict[str, Any] = {
            "mode": mode,
            "stages": list(stages),
            "stage_topk": [int(v) for v in stage_topks],
            "rerank_errors_stage1": int(stage_error_counts[0]),
            "rerank_errors_stage2": int(stage_error_counts[1]),
            "stage1_changed_queries": int(stage_changed[0]),
            "stage2_changed_queries": int(stage_changed[1]),
            "stage_errors": stage_errors,
            "llm_reranker_runtime": llm_runtime,
            "runtime_feature_coverage": runtime_feature_coverage,
            "llm_feature_injected_docs": int(llm_feature_stats["injected_docs"]),
            "llm_feature_missing_docs": int(llm_feature_stats["missing_docs"]),
            "llm_feature_names": list(LLM_ALLPAIR_FEATURE_NAMES),
            "llm_allpair_n_histogram": {
                str(k): int(v) for k, v in sorted(llm_allpair_n_hist.items())
            },
            "llm_allpair_lt20_queries": int(llm_allpair_lt20_queries),
            "ranking": ranking_rows,
        }

        applied = any(topk > 0 for topk in stage_topks)
        return {
            "candidates": current_candidates,
            "applied": bool(applied),
            "metadata": metadata,
        }

    def _resolve_mode_and_stages(self) -> Tuple[str, List[str], List[int]]:
        # Collapse the user-facing reranker config into one explicit execution plan
        # so the rest of the pipeline only reasons about ordered stages + top-k.
        mode = str(self.reranker_config.get("mode", "llm") or "llm").strip().lower()
        default_stage_topk = int(
            self.reranker_config.get("top_k", min(20, self.retrieval_top_k))
        )
        default_stage_topk = max(0, default_stage_topk)

        if mode == "llm":
            stages = ["llm"]
            stage_topks = [int(min(default_stage_topk, self.retrieval_top_k))]
            return mode, stages, stage_topks

        if mode == "ltr":
            stages = ["ltr"]
            stage_topks = [int(min(default_stage_topk, self.retrieval_top_k))]
            return mode, stages, stage_topks

        if mode == "cascade":
            stages = parse_stages(self.reranker_config.get("stages", "llm,ltr"))
            stage_topks = resolve_stage_topks(
                raw_stage_topk=self.reranker_config.get("stage_topk"),
                stages=stages,
                default_topk=default_stage_topk,
                retrieval_top_k=self.retrieval_top_k,
            )
            return mode, stages, stage_topks

        if mode == "llm_then_ltr_with_llm_features":
            # This specialized mode depends on a fixed all-pair LLM pass over the top
            # 20 docs and a second-stage LTR pass over the resulting top 10 docs. The
            # feature TSV/model are trained against exactly that contract, so keep the
            # config constrained here instead of letting downstream code drift.
            configured_stages = self.reranker_config.get("stages")
            if configured_stages is not None:
                parsed = parse_stages(configured_stages)
                if parsed != ["llm", "ltr"]:
                    raise ValueError(
                        "mode=llm_then_ltr_with_llm_features requires reranker.stages=llm,ltr"
                    )

            configured_topk = self.reranker_config.get("stage_topk")
            if configured_topk is not None:
                parsed_topk = resolve_stage_topks(
                    raw_stage_topk=configured_topk,
                    stages=["llm", "ltr"],
                    default_topk=20,
                    retrieval_top_k=max(self.retrieval_top_k, 20),
                )
                if parsed_topk != [20, 10]:
                    raise ValueError(
                        "mode=llm_then_ltr_with_llm_features requires reranker.stage_topk=20,10"
                    )

            method = str(
                self.reranker_config.get("method", "allpair") or "allpair"
            ).lower()
            if method != "allpair":
                raise ValueError(
                    "mode=llm_then_ltr_with_llm_features requires reranker.method=allpair"
                )

            return mode, ["llm", "ltr"], [20, 10]

        raise ValueError(
            "Unsupported reranker.mode: "
            f"{mode}. Supported: llm, ltr, cascade, llm_then_ltr_with_llm_features"
        )

    def _validate_requirements(
        self,
        mode: str,
        stages: Sequence[str],
        stage_topks: Sequence[int],
    ) -> None:
        require_llm = any(
            name == "llm" and topk > 0 for name, topk in zip(stages, stage_topks)
        )
        require_ltr = any(
            name == "ltr" and topk > 0 for name, topk in zip(stages, stage_topks)
        )

        # Fail fast before an episode starts stepping through the environment. A
        # missing model or feature schema here would otherwise surface much later.
        if require_llm:
            if not self.reranker_config.get("model"):
                raise ValueError("LLM rerank stage requires reranker.model")
            if not self.reranker_config.get("provider"):
                raise ValueError("LLM rerank stage requires reranker.provider")

        if require_ltr:
            self._ensure_ltr_runtime_ready()
            _ = self._get_ltr_model(mode=mode)
            feature_columns = self._get_feature_columns(mode=mode)
            if not feature_columns:
                raise ValueError("LTR stage requires non-empty feature columns")

    def _build_initial_candidates(
        self,
        rows: Sequence[Dict[str, Any]],
        use_pairwise_builder: bool,
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for retrieved_rank, row in enumerate(rows, start=1):
            base_row = dict(row)
            if use_pairwise_builder:
                base_candidate = build_pairwise_candidate_from_row(
                    dataset=self.dataset,
                    row=base_row,
                    retrieved_rank=int(retrieved_rank),
                    get_row_doc_id=get_row_doc_id,
                )
                if base_candidate is None:
                    base_candidate = dict(base_row)
                    base_candidate["retrieved_rank"] = int(retrieved_rank)
            else:
                base_candidate = dict(base_row)
                base_candidate["retrieved_rank"] = int(retrieved_rank)

            doc_id = get_row_doc_id(self.dataset, base_row)
            if not isinstance(doc_id, str) or not doc_id:
                doc_id = f"__fallback_{retrieved_rank}"

            distance = base_row.get("_distance")
            distance_value = (
                float(distance) if isinstance(distance, (int, float)) else None
            )

            candidate = dict(base_candidate)
            # docid/doc capture the reranker-facing identity and text surface, while
            # the *_rank fields are filled in later as the candidate moves through
            # stage1/stage2/final ordering.
            candidate["docid"] = doc_id
            candidate["doc"] = str(get_row_raw_key(base_row) or "")
            candidate["distance"] = distance_value
            candidate["_retrieved_rank"] = int(retrieved_rank)
            candidate["_distance_value"] = distance_value

            if "_task_meta_override" not in candidate:
                candidate["_task_meta_override"] = None
            if "_json_feature_vector" not in candidate:
                candidate["_json_feature_vector"] = None

            candidate["stage1_rank"] = None
            candidate["stage2_rank"] = None
            candidate["final_rank"] = None
            candidates.append(candidate)

        return candidates

    def _get_llm_reranker(self) -> LLMReranker:
        if self._llm_reranker is not None:
            return self._llm_reranker

        candidate_fields = self.reranker_config.get("candidate_text_fields")
        if not isinstance(candidate_fields, list):
            candidate_fields = [
                "key_raw_goal",
                "key_raw_state",
                "key_raw_context",
                "key_raw_progress",
            ]

        self._llm_reranker = LLMReranker(
            model=self.reranker_config.get("model"),
            provider=self.reranker_config.get("provider"),
            method=self.reranker_config.get("method", "allpair"),
            api_key_env=self.reranker_config.get("api_key_env"),
            api_key_file=self.reranker_config.get("api_key_file"),
            prompt_template=self.reranker_config.get("prompt_template"),
            tokenizer_name_or_path=self.reranker_config.get("tokenizer"),
            device=self.reranker_config.get(
                "llm_device", self.reranker_config.get("device")
            ),
            cache_dir=self.reranker_config.get("cache_dir"),
            debug_enabled=bool(self.reranker_config.get("debug_enabled", False)),
            debug_log_path=self.reranker_config.get("debug_log_path"),
            debug_top_n=self.reranker_config.get("debug_top_n"),
            batch_size=int(self.reranker_config.get("batch_size", 8)),
            max_pairs=self.reranker_config.get("max_pairs"),
            temperature=float(self.reranker_config.get("temperature", 0.0)),
            timeout_s=int(self.reranker_config.get("timeout_s", 60)),
            vllm_base_url=self.reranker_config.get("vllm_base_url"),
            vllm_api_key=self.reranker_config.get("vllm_api_key"),
            vllm_api_key_env=self.reranker_config.get("vllm_api_key_env"),
            vllm_api_key_file=self.reranker_config.get("vllm_api_key_file"),
            vllm_max_new_tokens=int(self.reranker_config.get("vllm_max_new_tokens", 1)),
            vllm_request_timeout_s=self.reranker_config.get("vllm_request_timeout_s"),
            vllm_bidirectional_compare=bool(
                self.reranker_config.get("vllm_bidirectional_compare", False)
            ),
            pair_batch_prompts=int(self.reranker_config.get("pair_batch_prompts", 32)),
            pair_max_inflight=int(self.reranker_config.get("pair_max_inflight", 4)),
            pair_retry=int(self.reranker_config.get("pair_retry", 2)),
            candidate_text_fields=[str(field) for field in candidate_fields],
        )
        return self._llm_reranker

    def _get_llm_stage_ranker(self) -> Any:
        if self._llm_stage_ranker is not None:
            return self._llm_stage_ranker
        # Reuse the raw stage ranker for all-pair feature extraction so the LLM
        # feature pipeline and normal LLM reranking see the same backend behavior.
        llm_reranker = self._get_llm_reranker()
        self._llm_stage_ranker = llm_reranker._get_ranker()
        return self._llm_stage_ranker

    def _ensure_ltr_runtime_ready(self) -> None:
        if self._ltr_modules_loaded:
            return

        from ltr.data import loader as ltr_data_loader
        from ltr.models import (
            FFNRanker,
            LambdaMARTRanker,
            ListNetRanker,
            SVMRanker,
            XGBoostRanker,
        )
        from ltr import runtime_features as ltr_runtime

        self._ltr_data_loader = ltr_data_loader
        self._ltr_runtime = ltr_runtime
        self._ltr_model_map = {
            "xgboost": XGBoostRanker,
            "lambdamart": LambdaMARTRanker,
            "ffn": FFNRanker,
            "listnet": ListNetRanker,
            "svmrank": SVMRanker,
        }
        self._ltr_modules_loaded = True

    def _build_query_context(
        self,
        query_text: str,
        task_name: Optional[str],
        variation_idx: Any,
    ) -> Any:
        self._ensure_ltr_runtime_ready()
        query_meta = {
            "task_name": task_name,
            "variation": variation_idx,
            "variation_idx": variation_idx,
            "consumer_model": self.reranker_config.get("consumer_model"),
        }
        return self._ltr_runtime.build_query_context(
            query_text=query_text,
            query_meta=query_meta,
            embed_model=self._get_embedder(),
        )

    def _get_embedder(self) -> Any:
        if self._embedder is not None:
            return self._embedder

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as error:
            raise ImportError(
                "LTR runtime features require sentence-transformers"
            ) from error

        model_kwargs: Dict[str, Any] = {}
        device = self.reranker_config.get("device")
        if device:
            model_kwargs["device"] = str(device)

        # Runtime LTR features own their embedder lifecycle here instead of sharing
        # the retrieval client so reranking can run independently from LanceDB setup.
        embedding_model = str(
            self.reranker_config.get("embedding_model", "intfloat/e5-base")
        )
        self._embedder = SentenceTransformer(embedding_model, **model_kwargs)
        return self._embedder

    def _get_tfidf_vectorizers(self) -> Any:
        if self._tfidf_vectorizers is not None:
            return self._tfidf_vectorizers
        if self._tfidf_unavailable:
            raise RuntimeError(
                self._tfidf_unavailable_reason or "TF-IDF vectorizers unavailable"
            )

        self._ensure_ltr_runtime_ready()
        base_dir = self.reranker_config.get("tfidf_base_dir")
        if base_dir is None:
            base_dir_path = DEFAULT_TFIDF_BASE_DIR
        else:
            base_dir_path = Path(str(base_dir))

        try:
            self._tfidf_vectorizers = self._ltr_runtime.load_tfidf_vectorizers(
                environment=self.dataset,
                base_dir=base_dir_path,
            )
        except Exception as error:
            # Cache the failure as well. Hybrid mode may choose to continue without
            # TF-IDF, but repeated retries here would just rethrow the same error.
            self._tfidf_unavailable = True
            self._tfidf_unavailable_reason = str(error)
            raise
        return self._tfidf_vectorizers

    def _get_feature_columns(self, mode: str) -> List[str]:
        if self._feature_columns is not None:
            return self._feature_columns

        feature_tsv = self.reranker_config.get("feature_tsv")
        if feature_tsv is None:
            if mode == "llm_then_ltr_with_llm_features":
                feature_tsv = (
                    DEFAULT_FEATURE_TSV_BASE_DIR_LLM_FEATURES
                    / self.dataset
                    / "ltr_train.tsv"
                )
            else:
                feature_tsv = (
                    DEFAULT_FEATURE_TSV_BASE_DIR / self.dataset / "ltr_train.tsv"
                )
        else:
            feature_tsv = Path(str(feature_tsv))

        self._feature_columns = load_feature_columns(feature_tsv)
        return self._feature_columns

    def _normalize_model_path(self, path: str) -> str:
        if path.endswith(".meta"):
            return path[:-5]
        if path.endswith(".xgb"):
            return path[:-4]
        return path

    def _infer_model_type_from_path(self, model_file: str) -> Optional[str]:
        name = Path(model_file).name.lower()
        if self._ltr_model_map is None:
            return None
        for model_type in self._ltr_model_map:
            if model_type in name:
                return model_type
        return None

    def _resolve_default_ltr_model_path(self, mode: str, model_type: str) -> Path:
        model_type = str(model_type).lower()

        if mode == "llm_then_ltr_with_llm_features":
            base_dir = Path(
                str(
                    self.reranker_config.get(
                        "model_base_dir_llm_features",
                        DEFAULT_MODEL_BASE_DIR_LLM_FEATURES,
                    )
                )
            )
        else:
            base_dir = Path(
                str(
                    self.reranker_config.get(
                        "model_base_dir",
                        DEFAULT_MODEL_BASE_DIR,
                    )
                )
            )

        extension_map = {
            "xgboost": ".pkl",
            "lambdamart": ".pkl",
            "svmrank": ".pkl",
            "ffn": ".pt",
            "listnet": ".pt",
        }
        suffix = extension_map.get(model_type, ".pkl")
        return base_dir / self.dataset / f"{model_type}_model{suffix}"

    def _load_ltr_model(
        self, model_file: str, model_type: Optional[str]
    ) -> Tuple[Any, str]:
        self._ensure_ltr_runtime_ready()
        assert self._ltr_model_map is not None

        model_path = self._normalize_model_path(model_file)
        tried: List[str] = []

        if model_type is not None:
            key = str(model_type).lower()
            model_class = self._ltr_model_map[key]
            model = model_class.load(model_path)
            return model, key

        # If the config does not pin a model type, try the filename hint first and
        # then fall back across all registered LTR model loaders.
        inferred = self._infer_model_type_from_path(model_file)
        model_try_order: List[str] = []
        if inferred is not None:
            model_try_order.append(inferred)
        for key in self._ltr_model_map:
            if key not in model_try_order:
                model_try_order.append(key)

        for key in model_try_order:
            model_class = self._ltr_model_map[key]
            tried.append(key)
            try:
                model = model_class.load(model_path)
                return model, key
            except Exception:
                continue

        raise ValueError(
            f"Could not load LTR model from {model_file}. Tried types: {tried}"
        )

    def _get_ltr_model(self, mode: str) -> Any:
        if self._ltr_model is not None:
            return self._ltr_model

        requested_model_type_raw = self.reranker_config.get("ltr_model_type")
        requested_model_type = (
            str(requested_model_type_raw).lower()
            if requested_model_type_raw is not None
            else None
        )
        model_file = self.reranker_config.get("ltr_model_file")
        if model_file is None:
            model_type_for_default_path = requested_model_type or DEFAULT_LTR_MODEL_TYPE
            model_path = self._resolve_default_ltr_model_path(
                mode=mode,
                model_type=model_type_for_default_path,
            )
            model_file = str(model_path)
        else:
            model_file = str(model_file)

        self._ltr_model, self._ltr_model_type = self._load_ltr_model(
            model_file=model_file,
            model_type=requested_model_type,
        )

        # Guard against silent feature drift between the trained LTR artifact and
        # the TSV schema used to build runtime/json feature vectors.
        feature_columns = self._get_feature_columns(mode=mode)
        model_n_features = getattr(self._ltr_model, "n_features", None)
        if model_n_features is not None and int(model_n_features) != len(
            feature_columns
        ):
            raise ValueError(
                f"Feature mismatch: model expects {model_n_features}, but feature TSV defines "
                f"{len(feature_columns)} columns."
            )

        return self._ltr_model

    def _infer_environment_base_path(self) -> Optional[Path]:
        if not self.lancedb_uri:
            return None
        uri_path = Path(self.lancedb_uri)
        if uri_path.name == "lancedb_indices":
            return uri_path.parent
        return None

    def _load_webarena_source_benchmark_score(
        self,
        thought_id: Any,
        task_name: Any,
        variation_idx: Any,
    ) -> Optional[float]:
        environment_base_path = self._infer_environment_base_path()
        if environment_base_path is None:
            return None

        thought_text = str(thought_id or "").strip()
        task_text = str(task_name or "").strip()
        variation_text = str(variation_idx or "").strip()
        if not thought_text or not task_text or not variation_text:
            return None

        cache_key = (thought_text, task_text, variation_text)
        if cache_key in self._webarena_source_score_cache:
            return self._webarena_source_score_cache[cache_key]

        source_path = (
            environment_base_path
            / "gold_repo"
            / task_text
            / variation_text
            / f"{thought_text}.json"
        )
        if not source_path.exists():
            self._webarena_source_score_cache[cache_key] = None
            return None

        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except Exception:
            self._webarena_source_score_cache[cache_key] = None
            return None

        source_file = payload.get("source_file")
        if not isinstance(source_file, str):
            self._webarena_source_score_cache[cache_key] = None
            return None

        webarena_benchmark_map = {
            "agentoccam-judge": 45.7,
            "2405_all_tasks_step_webarena_bugfix": 33.5,
            "webarena_clean_trajectory": 48.0,
        }

        parts = source_file.split("/")
        score: Optional[float] = None
        if "webarena" in parts:
            idx = parts.index("webarena")
            if idx + 2 < len(parts) and parts[idx + 1] == "data":
                directory = parts[idx + 2]
                mapped = webarena_benchmark_map.get(directory)
                if mapped is not None:
                    score = float(mapped)

        self._webarena_source_score_cache[cache_key] = score
        return score

    def _apply_llm_stage(
        self,
        query_text: str,
        candidates: List[Dict[str, Any]],
        stage_topk: int,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        if stage_topk <= 0 or not candidates:
            return list(candidates), None

        prefix = list(candidates[:stage_topk])
        reranker = self._get_llm_reranker()

        try:
            reranked_output = reranker.rerank(
                query_text,
                prefix,
                top_k=min(stage_topk, len(prefix)),
            )
            reranked_prefix = build_reranked_prefix_from_output(reranked_output, prefix)
        except Exception as error:
            return list(candidates), f"{type(error).__name__}: {error}"

        merged = merge_stage_order(candidates, reranked_prefix, topn=stage_topk)
        return merged, None

    def _apply_llm_stage_with_features(
        self,
        query_text: str,
        candidates: List[Dict[str, Any]],
        stage_topk: int,
    ) -> Tuple[
        List[Dict[str, Any]], Dict[str, Dict[str, float]], Dict[str, Any], Optional[str]
    ]:
        if stage_topk <= 0 or not candidates:
            return list(candidates), {}, {"n": 0, "ordered_docids": []}, None

        try:
            # This path does more than rerank: it also exports per-doc all-pair
            # statistics that the downstream LTR stage can splice into its vector.
            llm_reranker = self._get_llm_reranker()
            llm_ranker = self._get_llm_stage_ranker()
            computed = compute_allpair_for_candidates(
                query_text=query_text,
                candidates=candidates,
                llm_reranker=llm_reranker,
                ranker=llm_ranker,
                allpair_top_k=stage_topk,
            )
            reranked_prefix = list(computed.get("reranked_prefix", []))
        except Exception as error:
            return (
                list(candidates),
                {},
                {"n": int(min(stage_topk, len(candidates))), "ordered_docids": []},
                f"{type(error).__name__}: {error}",
            )

        merged = merge_stage_order(candidates, reranked_prefix, topn=stage_topk)
        return merged, dict(computed.get("features_by_docid", {})), computed, None

    def _apply_ltr_stage(
        self,
        mode: str,
        query_ctx: Any,
        candidates: List[Dict[str, Any]],
        stage_topk: int,
        feature_columns: Sequence[str],
        feature_source: str,
        missing_feature_policy: str,
        normalize_features: bool,
        llm_feature_map: Dict[str, Dict[str, float]],
        llm_feature_indices: Dict[str, int],
        llm_feature_stats: Dict[str, Any],
        ltr_stats: Dict[str, int],
        inject_llm_features: bool,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        if stage_topk <= 0 or not candidates:
            return list(candidates), None

        model = self._get_ltr_model(mode=mode)
        prefix = list(candidates[:stage_topk])
        import numpy as np

        tfidf_vecs = None
        if feature_source in {"runtime", "hybrid"}:
            try:
                tfidf_vecs = self._get_tfidf_vectorizers()
            except Exception:
                if feature_source == "runtime":
                    raise
                tfidf_vecs = None

        try:
            for candidate in prefix:
                # Normalize precomputed JSON features to either list-or-None so the
                # source-selection logic below can treat all candidates uniformly.
                json_feature_vector = candidate.get("_json_feature_vector")
                if isinstance(json_feature_vector, list):
                    candidate["_json_feature_vector"] = json_feature_vector
                else:
                    candidate["_json_feature_vector"] = None

                runtime_feature_vector: Optional[List[float]] = None
                runtime_missing_count = len(feature_columns)
                runtime_error_reason = ""

                if feature_source in {"runtime", "hybrid"}:
                    ltr_stats["retrieved_runtime_attempts"] += 1
                    # Some offline pipelines attach candidate-specific task metadata
                    # that should override the query-level task labels at feature time.
                    task_meta_override = candidate.get("_task_meta_override")
                    if not isinstance(task_meta_override, dict):
                        task_meta_override = None

                    if task_meta_override is not None:
                        ltr_stats["retrieved_task_meta_overrides"] += 1
                    else:
                        ltr_stats["retrieved_task_meta_fallback_to_query"] += 1

                    try:
                        if query_ctx is None:
                            raise ValueError("query_context_unavailable")

                        runtime_row = candidate
                        if self.dataset == "webarena":
                            # WebArena runtime features optionally use the source run's
                            # benchmark score, but that value lives outside the row.
                            source_benchmark_score = (
                                self._load_webarena_source_benchmark_score(
                                    thought_id=candidate.get("thought_id"),
                                    task_name=candidate.get("task_name"),
                                    variation_idx=candidate.get("variation_idx"),
                                )
                            )
                            if source_benchmark_score is not None:
                                runtime_row = dict(candidate)
                                runtime_row["source_benchmark_score"] = float(
                                    source_benchmark_score
                                )

                        (
                            _,
                            runtime_feature_vector,
                            runtime_diag,
                        ) = self._ltr_runtime.build_candidate_features(
                            query_ctx=query_ctx,
                            lancedb_row=runtime_row,
                            retrieved_rank=int(candidate.get("retrieved_rank", 0) or 0),
                            distance=candidate.get("distance"),
                            feature_columns=list(feature_columns),
                            tfidf_vecs=tfidf_vecs,
                            query_meta_override=task_meta_override,
                        )
                        runtime_missing_count = int(
                            runtime_diag.get("feature_missing_count", 0)
                        )
                        runtime_error_reason = str(
                            runtime_diag.get("feature_error_reason") or ""
                        )
                        ltr_stats["retrieved_with_runtime_features"] += 1
                    except Exception as error:
                        runtime_feature_vector = None
                        runtime_error_reason = f"{type(error).__name__}: {error}"
                        ltr_stats["retrieved_feature_failures"] += 1

                selected_feature_vector: Optional[List[float]] = None
                feature_source_selected = "error"
                feature_missing_count = len(feature_columns)
                feature_error_reason = ""

                # selected_feature_vector is the actual vector passed to the LTR
                # model after source selection and fallback handling.
                # runtime: always use on-the-fly features
                # json: only use precomputed vectors already attached to the row
                # hybrid: prefer runtime and fall back to json if runtime fails
                if feature_source == "json":
                    if isinstance(candidate.get("_json_feature_vector"), list):
                        selected_feature_vector = list(
                            candidate.get("_json_feature_vector")
                        )
                        feature_source_selected = "json"
                        feature_missing_count = 0
                    else:
                        feature_error_reason = "json_feature_missing"
                elif feature_source == "runtime":
                    if isinstance(runtime_feature_vector, list):
                        selected_feature_vector = runtime_feature_vector
                        feature_source_selected = "runtime"
                        feature_missing_count = runtime_missing_count
                        feature_error_reason = runtime_error_reason
                    else:
                        feature_error_reason = (
                            runtime_error_reason or "runtime_feature_missing"
                        )
                else:
                    if isinstance(runtime_feature_vector, list):
                        selected_feature_vector = runtime_feature_vector
                        feature_source_selected = "runtime"
                        feature_missing_count = runtime_missing_count
                        feature_error_reason = runtime_error_reason
                    elif isinstance(candidate.get("_json_feature_vector"), list):
                        selected_feature_vector = list(
                            candidate.get("_json_feature_vector")
                        )
                        feature_source_selected = "json_fallback"
                        feature_missing_count = 0
                        feature_error_reason = runtime_error_reason
                        ltr_stats["retrieved_json_fallback_used"] += 1
                    else:
                        feature_error_reason = (
                            runtime_error_reason or "runtime_and_json_missing"
                        )

                candidate["_feature_vector"] = selected_feature_vector
                candidate["_feature_source"] = feature_source_selected
                candidate["_feature_missing_count"] = int(feature_missing_count)
                candidate["_feature_error_reason"] = feature_error_reason
                candidate["_feature_available"] = (
                    1 if isinstance(selected_feature_vector, list) else 0
                )

                if candidate["_feature_available"]:
                    ltr_stats["retrieved_with_features"] += 1
                else:
                    ltr_stats["retrieved_without_features"] += 1

                if inject_llm_features and isinstance(
                    candidate.get("_feature_vector"), list
                ):
                    doc_id_for_llm = candidate.get("docid")
                    llm_values = None
                    if isinstance(doc_id_for_llm, str):
                        llm_values = llm_feature_map.get(doc_id_for_llm)

                    if llm_values is None:
                        llm_values = zero_llm_allpair_features()
                        llm_feature_stats["missing_docs"] += 1
                    else:
                        llm_feature_stats["injected_docs"] += 1

                    selected_vector = list(candidate["_feature_vector"])
                    for feature_name, feature_index in llm_feature_indices.items():
                        if feature_index < 0 or feature_index >= len(selected_vector):
                            continue
                        value = float(llm_values.get(feature_name, 0.0))
                        selected_vector[feature_index] = value
                        llm_feature_stats["values"][feature_name].append(value)

                    candidate["_feature_vector"] = selected_vector

            feature_vectors: List[List[float]] = []
            feature_indices: List[int] = []
            # Score only rows that ended up with a usable feature vector; the rest
            # stay in the candidate set and receive a backoff score below.
            for local_index, candidate in enumerate(prefix):
                feature_vector = candidate.get("_feature_vector")
                if isinstance(feature_vector, list):
                    feature_vectors.append(feature_vector)
                    feature_indices.append(local_index)

            model_scores: List[Optional[float]] = [None] * len(prefix)
            if feature_vectors:
                matrix = np.asarray(feature_vectors, dtype=float)
                qid_array = np.full(matrix.shape[0], 1, dtype=int)
                if normalize_features:
                    matrix = self._ltr_data_loader.normalize_features_query_level(
                        matrix, qid_array
                    )
                predicted = ensure_1d_scores(model.predict(matrix, qid_array))
                for idx_local, score in zip(feature_indices, predicted):
                    model_scores[idx_local] = float(score)

            scored_subset: List[Tuple[float, int, Dict[str, Any]]] = []
            for local_index, candidate in enumerate(prefix):
                score = model_scores[local_index]
                if score is None:
                    # Keep candidates without usable features in the ranking with a
                    # policy-controlled backoff score instead of dropping them.
                    if missing_feature_policy == "baseline":
                        score = float(
                            ltr_stats["top_k"]
                            - int(candidate.get("retrieved_rank", 0))
                            + 1
                        )
                    elif missing_feature_policy == "zero":
                        score = 0.0
                    else:
                        score = -1e9
                scored_subset.append(
                    (float(score), int(candidate.get("retrieved_rank", 0)), candidate)
                )

            # Break score ties by earlier retrieval rank so reranking remains stable.
            scored_subset.sort(key=lambda item: (item[0], -item[1]), reverse=True)
            reranked_prefix = [item[2] for item in scored_subset]
            merged = merge_stage_order(candidates, reranked_prefix, topn=stage_topk)
            return merged, None
        except Exception as error:
            return list(candidates), f"{type(error).__name__}: {error}"
