# traj_retrieval/core/lancedb_retrieval.py
# LanceDB-based retrieval implementation

import os
import json
import threading
from typing import Optional, Dict, Any, Tuple

from .retrieval_strategy import RetrievalStrategy, RetrievalResult
from .base_handler import EnvironmentHandler
from .lancedb_client import LanceDBClient


class LanceDBRetrievalStrategy(RetrievalStrategy):
    """
    LanceDB-based trajectory retrieval strategy.

    This implementation uses:
    - LanceDB for vector storage
    - Structured query building with goal, state, context, progress
    - Rich metadata from LanceDB entries
    """

    def __init__(
        self,
        indices_dir: str,
        model_name: str = "intfloat/e5-base",
        table_name: str = "alfworld",
        search_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize LanceDB retrieval strategy.

        Args:
            indices_dir: Directory containing LanceDB database
            model_name: Model name for encoding
            table_name: Name of the LanceDB table
            search_config: Optional search configuration for filters and query params
        """
        self.indices_dir = indices_dir
        self.model_name = model_name
        self.table_name = table_name
        self.search_config = search_config
        self._client = None  # Lazy loading of LanceDB client
        self._client_lock = (
            threading.Lock()
        )  # Prevents race conditions during concurrent initialization
        self._rerank_orchestrator = None
        self._rerank_signature = None
        self._rerank_lock = threading.Lock()

        print(f"[LanceDBRetrieval] Initialized LanceDB-based retrieval")
        print(f"[LanceDBRetrieval]   Indices dir: {indices_dir}")
        print(f"[LanceDBRetrieval]   Model: {model_name}")
        print(f"[LanceDBRetrieval]   Table: {table_name}")

    def _get_source_metadata(self, row: Dict[str, Any]) -> Tuple[Optional[str], str]:
        table = (self.table_name or "").strip().lower()

        if table == "alfworld":
            metadata = row.get("metadata")
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except Exception:
                    metadata = None
            if isinstance(metadata, dict):
                value = metadata.get("model_name")
                if value is not None and value != "":
                    return "metadata.model_name", str(value)
            value = row.get("model_name")
            if value is not None and value != "":
                return "model_name", str(value)
            return None, "unknown"

        if table == "webarena":
            value = row.get("agent_type")
            if value is not None and value != "":
                return "agent_type", str(value)
        return None, "unknown"

    def _get_rerank_orchestrator(
        self,
        reranker_config: Dict[str, Any],
        top_k: int,
    ):
        from .reranker_orchestrator import UnifiedRerankerOrchestrator

        # Reuse the orchestrator until the effective reranker config changes.
        # Model loading is expensive and retrieve() can be called at every step.
        signature_payload = {
            "table_name": self.table_name,
            "indices_dir": self.indices_dir,
            "top_k": int(top_k),
            "reranker": reranker_config,
        }
        signature = json.dumps(signature_payload, sort_keys=True, default=str)

        if (
            self._rerank_orchestrator is not None
            and self._rerank_signature == signature
        ):
            return self._rerank_orchestrator

        with self._rerank_lock:
            if (
                self._rerank_orchestrator is not None
                and self._rerank_signature == signature
            ):
                return self._rerank_orchestrator
            self._rerank_orchestrator = UnifiedRerankerOrchestrator(
                dataset=self.table_name,
                retrieval_top_k=int(top_k),
                reranker_config=reranker_config,
                lancedb_uri=self.indices_dir,
            )
            self._rerank_signature = signature
            return self._rerank_orchestrator

    @property
    def client(self):
        """
        Lazy load the LanceDB client with thread-safe initialization.

        Note: Uses threading.Lock (not asyncio.Lock) because:
        1. This property is accessed from synchronous retrieve() method
        2. Multiple async tasks can trigger concurrent initialization
        3. The lock is only held during one-time initialization (minimal contention)
        4. After initialization, all episodes share the client without locking

        This prevents race conditions when multiple episodes start concurrently
        and all try to initialize the client at Step 0.
        """
        # Double-checked locking pattern for thread safety
        if self._client is None:
            with self._client_lock:
                # Check again after acquiring lock (another thread might have initialized it)
                if self._client is None:
                    self._client = LanceDBClient(
                        db_uri=self.indices_dir,
                        model_name=self.model_name,
                        table_name=self.table_name,
                        search_config=self.search_config,
                    )
        return self._client

    def build_query(
        self,
        goal: str,
        observation: str,
        inventory: str = "",
        recent_history: str = "",
        current_step: int = 0,
        current_reward: float = 0.0,
        env_handler: EnvironmentHandler = None,
        **kwargs,
    ) -> str:
        """
        Build query string using environment-specific formatting.

        Delegates to environment handler to ensure the query format exactly matches
        the format used during database indexing. This is critical because different
        environments have different schemas (e.g., AlfWorld has no inventory).

        Args:
            goal: Task goal/description
            observation: Current observation
            inventory: Current inventory (may be ignored by some environments)
            recent_history: Recent action history
            current_step: Current step number
            current_reward: Current reward/score (may be ignored by some environments)
            env_handler: Environment handler with environment-specific query building logic
            **kwargs: Additional query building parameters

        Returns:
            Query string formatted to match database entries

        Raises:
            ValueError: If env_handler is None
        """
        if env_handler is None:
            raise ValueError(
                "env_handler is required for query building. "
                "The query format is environment-specific and must match the indexing format."
            )

        # Delegate to environment handler (knows the correct format for this environment)
        query_text = env_handler.build_retrieval_query(
            goal=goal,
            observation=observation,
            inventory=inventory,
            recent_history=recent_history,
            current_step=current_step,
            current_reward=current_reward,
        )

        return query_text

    def refresh(self) -> None:
        """
        Clear cached LanceDB and reranker state so the next retrieval reopens the DB.

        This is used by progressive-memory experiments where new rows may be appended
        during evaluation and subsequent episodes should observe the updated corpus.
        """
        with self._client_lock:
            self._client = None
        with self._rerank_lock:
            self._rerank_orchestrator = None
            self._rerank_signature = None

    def retrieve(
        self,
        query: str = None,
        env_handler: EnvironmentHandler = None,
        task_name: str = None,
        k: int = 1,
        max_steps_to_show: int = 20,
        # LanceDB-specific parameters
        goal: str = None,
        observation: str = None,
        inventory: str = None,
        recent_history: str = None,
        current_step: int = 0,
        current_reward: float = 0.0,
        **kwargs,
    ) -> Optional[RetrievalResult]:
        """
        Retrieve a trajectory using LanceDB search.

        This method handles the complete retrieval flow:
        1. Build query (if not provided)
        2. Search LanceDB
        3. Format results
        4. Extract metadata

        Args:
            query: Pre-built query string (if None, will build from context)
            env_handler: Environment handler for formatting
            task_name: Task name for filtering
            k: Number of results to retrieve
            max_steps_to_show: Maximum steps to include in formatted context
            goal: Task goal (for query building)
            observation: Current observation (for query building)
            inventory: Current inventory (for query building)
            recent_history: Recent action history (for query building)
            current_step: Current step number (for query building)
            current_reward: Current reward (for query building)
            **kwargs: Additional parameters

        Returns:
            RetrievalResult with formatted trajectory, or None if retrieval fails
        """
        try:
            # Build query if not provided
            if query is None:
                if goal is None or observation is None:
                    raise ValueError(
                        "Must provide either 'query' or both 'goal' and 'observation'"
                    )

                if env_handler is None:
                    raise ValueError(
                        "env_handler is required for query building. "
                        "Pass env_handler to retrieve() method."
                    )

                print(f"[LanceDBRetrieval] Building query from current state...")
                print(
                    f"[LanceDBRetrieval]   Goal: {goal[:100]}..."
                    if len(goal) > 100
                    else f"[LanceDBRetrieval]   Goal: {goal}"
                )
                print(
                    f"[LanceDBRetrieval]   Observation: {observation[:100]}..."
                    if len(observation) > 100
                    else f"[LanceDBRetrieval]   Observation: {observation}"
                )
                print(
                    f"[LanceDBRetrieval]   Environment: {env_handler.environment_name}"
                )

                query = self.build_query(
                    goal=goal,
                    observation=observation,
                    inventory=inventory or "",
                    recent_history=recent_history or "",
                    current_step=current_step,
                    current_reward=current_reward,
                    env_handler=env_handler,  # Pass env_handler for environment-specific formatting
                )

                print(
                    f"[LanceDBRetrieval] Built query (first 200 chars): {query[:200]}..."
                )

            print(
                f"[LanceDBRetrieval] Performing semantic search based on query embedding..."
            )
            print(f"[LanceDBRetrieval] Search with config-based filters")

            # Build context for dynamic filter values
            # This allows filters like $key_raw.goal or !$task_name to be evaluated
            # Only include fields that might be used in filters (current or future)
            search_context = {
                "key_raw": {"goal": goal},  # Used in filters like: goal: $key_raw.goal
                "task_name": task_name,  # Use task name directly (now matches LanceDB format)
                "variation_idx": kwargs.get(
                    "variation_idx"
                ),  # May be used in future filters
                "current_step": current_step,  # May be used in future filters
                "current_reward": current_reward,  # May be used in future filters
            }

            # Search with context - filters will be applied from search_config
            # The client will automatically use filters from search_config if provided,
            # otherwise falls back to defaults (success = true)

            # Get retrieval parameters from search_config
            # Handle case where search_config is None (default to empty dict)
            search_config = self.search_config if self.search_config is not None else {}
            candidate_config = search_config.get("candidate_generation", {})
            top_k = candidate_config.get("top_k", 100)
            rank_retrieve = candidate_config.get("rank_retrieve", 1)
            prompt_top_n_enabled = candidate_config.get("prompt_top_n_enabled", False)
            prompt_top_n = candidate_config.get("prompt_top_n", 1)

            # Validate rank_retrieve
            if rank_retrieve > top_k:
                raise ValueError(
                    f"rank_retrieve ({rank_retrieve}) must be <= top_k ({top_k})\n"
                    f"rank_retrieve selects which trajectory to use from the top_k results (1-indexed)."
                )
            if rank_retrieve < 1:
                raise ValueError(
                    f"rank_retrieve must be >= 1 (1-indexed), got {rank_retrieve}"
                )
            if not isinstance(prompt_top_n_enabled, bool):
                raise ValueError(
                    f"prompt_top_n_enabled must be a boolean, got {type(prompt_top_n_enabled).__name__}"
                )
            if not isinstance(prompt_top_n, int) or prompt_top_n < 1:
                raise ValueError(f"prompt_top_n must be >= 1, got {prompt_top_n}")
            if prompt_top_n_enabled and prompt_top_n > top_k:
                raise ValueError(
                    f"prompt_top_n ({prompt_top_n}) must be <= top_k ({top_k}) when prompt_top_n_enabled=true"
                )

            results_top_k = self.client.search(
                query_text=query,
                k=top_k,  # Get top_k for statistics and selection
                context=search_context,  # Provide context for dynamic filters
            )

            if not results_top_k:
                print(
                    f"[LanceDBRetrieval] No results found (this should never happen if DB has data)"
                )
                return None

            # Optional unified reranking (llm/ltr/cascade/llm_then_ltr_with_llm_features)
            reranker_applied = False
            reranker_error = None
            reranker_metadata = {}
            reranker_config = search_config.get("reranker", {})
            if isinstance(reranker_config, dict) and reranker_config.get("enabled"):
                try:
                    # The unified reranker only reorders the retrieved LanceDB rows.
                    # Prompt formatting still happens below on the same row objects.
                    reranker = self._get_rerank_orchestrator(
                        reranker_config=reranker_config,
                        top_k=top_k,
                    )
                    rerank_output = reranker.rerank(
                        query_text=query,
                        candidates=results_top_k,
                        task_name=task_name,
                        variation_idx=kwargs.get("variation_idx"),
                        current_step=current_step,
                        current_reward=current_reward,
                    )
                    reranked_candidates = rerank_output.get("candidates", [])
                    if isinstance(reranked_candidates, list) and reranked_candidates:
                        results_top_k = reranked_candidates
                    reranker_applied = bool(rerank_output.get("applied", False))
                    reranker_metadata = (
                        rerank_output.get("metadata", {})
                        if isinstance(rerank_output.get("metadata"), dict)
                        else {}
                    )
                    print(
                        "[LanceDBRetrieval] ✓ Unified reranker applied"
                        f" (mode={reranker_metadata.get('mode', reranker_config.get('mode', 'llm'))})",
                        flush=True,
                    )
                except Exception as error:
                    reranker_error = str(error)
                    print(
                        f"[LanceDBRetrieval] ⚠️  Reranker failed: {reranker_error}",
                        flush=True,
                    )

            # Ensure rank fields exist even when reranker is disabled or fails
            for final_rank, row in enumerate(results_top_k, start=1):
                if row.get("retrieved_rank") is None:
                    row["retrieved_rank"] = int(final_rank)
                if row.get("final_rank") is None:
                    row["final_rank"] = int(final_rank)

            # Check if we have enough results for the requested rank
            if len(results_top_k) < rank_retrieve:
                print(
                    f"[LanceDBRetrieval] ⚠️  Warning: Only {len(results_top_k)} results available, but rank_retrieve={rank_retrieve}"
                )
                print(f"[LanceDBRetrieval]   Using best available result (rank 1)")
                rank_retrieve = 1

            # Get the selected result (rank_retrieve is 1-indexed, so subtract 1 for array index)
            selected_result = results_top_k[rank_retrieve - 1]
            source_field_name, selected_source_value = self._get_source_metadata(
                selected_result
            )

            print(
                f"[LanceDBRetrieval] Retrieved trajectory at rank {rank_retrieve} from top {top_k} results"
            )

            # Extract score (distance from LanceDB, smaller is better)
            # Note: LanceDB returns L2 distance, smaller = more similar
            score = float(selected_result.get("_distance", 0.0))

            # Compute statistics on top_k results
            distances_top_k = [float(r.get("_distance", 0.0)) for r in results_top_k]
            mean_distance_top_k = (
                sum(distances_top_k) / len(distances_top_k) if distances_top_k else 0.0
            )

            # Calculate standard deviation
            if len(distances_top_k) > 1:
                variance = sum(
                    (d - mean_distance_top_k) ** 2 for d in distances_top_k
                ) / len(distances_top_k)
                std_distance_top_k = variance**0.5
            else:
                std_distance_top_k = 0.0

            # Log what was retrieved
            retrieved_task = selected_result.get("task_name", "unknown")
            retrieved_variation = selected_result.get("variation_idx", "unknown")
            retrieved_thought_id = selected_result.get("thought_id", "unknown")

            print(f"[LanceDBRetrieval] ✓ Retrieved trajectory at rank {rank_retrieve}:")
            print(f"[LanceDBRetrieval]   Task: {retrieved_task}")
            print(f"[LanceDBRetrieval]   Variation: {retrieved_variation}")
            print(f"[LanceDBRetrieval]   Trajectory ID: {retrieved_thought_id}")
            if source_field_name:
                print(
                    f"[LanceDBRetrieval]   Retrieved source ({source_field_name}): {selected_source_value}"
                )
            print(
                f"[LanceDBRetrieval]   Similarity distance: {score:.4f} (lower = more similar)"
            )
            print(
                f"[LanceDBRetrieval]   Top-{top_k} stats: mean={mean_distance_top_k:.4f}, std={std_distance_top_k:.4f}"
            )

            def _row_to_raw_trajectory(
                row: Dict[str, Any]
            ) -> Tuple[Dict[str, Any], list]:
                # Environment handlers format standardized trajectory payloads rather
                # than raw LanceDB rows, so adapt the row back into the legacy shape.
                guidance_str = row.get("guidance", "[]")
                if isinstance(guidance_str, str):
                    guidance_steps = json.loads(guidance_str)
                else:
                    guidance_steps = guidance_str if guidance_str else []
                trajectory_steps_local = [
                    {
                        "action": step.get("action", ""),
                        "observation": step.get("observation", ""),
                        "inventory": step.get("inventory", ""),
                        "score": step.get("score", 0.0),
                    }
                    for step in guidance_steps
                ]
                raw_trajectory_local = {
                    "trajectory_steps": trajectory_steps_local,
                    "task_description": row.get("key_raw_goal", ""),
                    "task_name": row.get("task_name", ""),
                    "variation_idx": row.get("variation_idx", ""),
                    "trajectory_id": row.get("thought_id", ""),
                    "is_successful": row.get("success", False),
                }
                return raw_trajectory_local, trajectory_steps_local

            def _build_prompt_ranking_entry(
                row: Dict[str, Any], rank_in_prompt: int
            ) -> Dict[str, Any]:
                return {
                    "rank_in_prompt": rank_in_prompt,
                    "thought_id": row.get("thought_id", ""),
                    "task_name": row.get("task_name", ""),
                    "variation_idx": row.get("variation_idx", ""),
                    "retrieved_rank": row.get("retrieved_rank"),
                    "final_rank": row.get("final_rank"),
                    "distance": float(row.get("_distance", 0.0)),
                    "is_successful": row.get("success", False),
                }

            raw_trajectory, trajectory_steps = _row_to_raw_trajectory(selected_result)
            prompt_context_ranking = []
            prompt_top_n_requested = prompt_top_n if prompt_top_n_enabled else 1

            if prompt_top_n_enabled:
                # selected_result still drives score/metadata, but the prompt can
                # expose multiple reranked trajectories for the action model.
                prompt_rows = results_top_k[: min(prompt_top_n, len(results_top_k))]
                prompt_top_n_used = len(prompt_rows)
                context_chunks = []
                for idx, row in enumerate(prompt_rows, start=1):
                    prompt_context_ranking.append(_build_prompt_ranking_entry(row, idx))
                    prompt_raw_trajectory, _ = _row_to_raw_trajectory(row)
                    chunk = env_handler.format_retrieval_result(
                        raw_data=prompt_raw_trajectory,
                        max_steps=max_steps_to_show,
                        retrieval_type="trajectory",
                    )
                    if chunk and chunk.strip():
                        context_chunks.append(f"[Retrieved #{idx}]\n{chunk.strip()}")
                formatted_context = "\n\n".join(context_chunks)
            else:
                prompt_top_n_used = 1
                prompt_context_ranking.append(
                    _build_prompt_ranking_entry(selected_result, 1)
                )
                # Keep legacy behavior exactly when prompt_top_n is disabled.
                formatted_context = env_handler.format_retrieval_result(
                    raw_data=raw_trajectory,
                    max_steps=max_steps_to_show,
                    retrieval_type="trajectory",
                )

            # Extract action-observation pairs for logging (first 10 steps)
            action_obs_pairs = [
                {"action": step["action"], "observation": step["observation"]}
                for step in trajectory_steps[:10]
            ]

            # Extract and structure metadata
            trajectory_length = len(trajectory_steps)
            metadata_str = selected_result.get("metadata", "{}")
            db_metadata = (
                json.loads(metadata_str)
                if isinstance(metadata_str, str)
                else metadata_str
            )

            # Get filter debug info from client (if available)
            filter_debug = getattr(self.client, "_last_filter_debug", None)

            # Standardized metadata structure
            metadata = {
                # Trajectory information
                "trajectory_length": trajectory_length,
                "steps_shown": min(trajectory_length, max_steps_to_show),
                # Task information
                "task_description": selected_result.get("key_raw_goal", ""),
                "task_name": selected_result.get("task_name", ""),
                "task_type": selected_result.get("task_type", ""),
                "variation_idx": selected_result.get("variation_idx", ""),
                # Retrieval information
                "retrieval_type": "trajectory",
                "vector_store": "lancedb",
                "thought_id": selected_result.get("thought_id", ""),
                "is_successful": selected_result.get("success", False),
                "retrieved_source_field": source_field_name,
                "retrieved_source_value": selected_source_value,
                # Similarity/distance information (selected rank)
                "similarity_score": score,  # L2 distance (smaller = more similar)
                "distance": score,  # Keep for backward compatibility
                "rank_retrieve": rank_retrieve,  # Which rank was selected (1-indexed)
                "prompt_top_n_enabled": prompt_top_n_enabled,
                "prompt_top_n_requested": prompt_top_n_requested,
                "prompt_top_n_used": prompt_top_n_used,
                "prompt_context_ranking": prompt_context_ranking,
                # Top-k statistics
                f"top_{top_k}_mean_distance": mean_distance_top_k,
                f"top_{top_k}_std_distance": std_distance_top_k,
                f"top_{top_k}_count": len(results_top_k),
                "top_k": top_k,  # How many results were retrieved for stats
                # Logging data (first 10 steps for debugging)
                "action_obs_pairs": action_obs_pairs,
                # Database-specific metadata
                "db_metadata": db_metadata,
                # Query debug info (NEW: for debugging filters and query text)
                "query_debug": filter_debug if filter_debug else {},
                # Query text (NEW: exact query string used for embedding and LanceDB search)
                "query": query,
                # reranker metadata records both the original retrieval order and the
                # post-rerank order. `rank_retrieve` says which final item became the
                # selected_result for metrics, while prompt_top_n may expose several
                # final-ranked rows to the action model at the same time.
                "reranker_enabled": bool(reranker_config.get("enabled"))
                if isinstance(reranker_config, dict)
                else False,
                "reranker_applied": reranker_applied,
                "reranker_mode": (
                    reranker_metadata.get("mode")
                    if isinstance(reranker_metadata, dict)
                    else None
                ),
                "reranker_stages": (
                    reranker_metadata.get("stages")
                    if isinstance(reranker_metadata, dict)
                    else None
                ),
                "reranker_stage_topk": (
                    reranker_metadata.get("stage_topk")
                    if isinstance(reranker_metadata, dict)
                    else None
                ),
                "reranker_model": reranker_config.get("model")
                if isinstance(reranker_config, dict)
                else None,
                "reranker_provider": reranker_config.get("provider")
                if isinstance(reranker_config, dict)
                else None,
                "reranker_top_k": reranker_config.get("top_k")
                if isinstance(reranker_config, dict)
                else None,
                # retrieved_rank: original rank straight from candidate generation
                # final_rank: rank after all reranker stages
                # stage1/stage2_rank: intermediate snapshots for cascade debugging
                "retrieved_rank": selected_result.get("retrieved_rank", rank_retrieve),
                "final_rank": selected_result.get("final_rank", rank_retrieve),
                "stage1_rank": selected_result.get("stage1_rank"),
                "stage2_rank": selected_result.get("stage2_rank"),
                "rerank_errors_stage1": (
                    reranker_metadata.get("rerank_errors_stage1", 0)
                    if isinstance(reranker_metadata, dict)
                    else 0
                ),
                "rerank_errors_stage2": (
                    reranker_metadata.get("rerank_errors_stage2", 0)
                    if isinstance(reranker_metadata, dict)
                    else 0
                ),
                "llm_reranker_runtime": (
                    reranker_metadata.get("llm_reranker_runtime", {})
                    if isinstance(reranker_metadata, dict)
                    else {}
                ),
                "runtime_feature_coverage": (
                    reranker_metadata.get("runtime_feature_coverage", {})
                    if isinstance(reranker_metadata, dict)
                    else {}
                ),
                "llm_feature_injected_docs": (
                    reranker_metadata.get("llm_feature_injected_docs", 0)
                    if isinstance(reranker_metadata, dict)
                    else 0
                ),
                "llm_feature_missing_docs": (
                    reranker_metadata.get("llm_feature_missing_docs", 0)
                    if isinstance(reranker_metadata, dict)
                    else 0
                ),
                "reranker_ranking": (
                    reranker_metadata.get("ranking", [])
                    if isinstance(reranker_metadata, dict)
                    else []
                ),
                "reranker_error": reranker_error,
            }

            return RetrievalResult(
                raw_data=raw_trajectory,
                score=score,
                formatted_context=formatted_context,
                metadata=metadata,
            )

        except FileNotFoundError as e:
            error_msg = f"[LanceDBRetrieval] CRITICAL: Database not found: {e}"
            print(error_msg)
            raise RuntimeError(f"CRITICAL RETRIEVAL ERROR: {error_msg}") from e
        except ValueError as e:
            error_msg = f"[LanceDBRetrieval] CRITICAL: Configuration error: {e}"
            print(error_msg)
            raise RuntimeError(f"CRITICAL RETRIEVAL ERROR: {error_msg}") from e
        except RuntimeError as e:
            # LanceDB errors (like the vector index error)
            error_msg = f"[LanceDBRetrieval] CRITICAL: LanceDB error: {e}"
            print(error_msg)
            import traceback

            traceback.print_exc()
            raise RuntimeError(f"CRITICAL RETRIEVAL ERROR: {error_msg}") from e
        except Exception as e:
            error_msg = (
                f"[LanceDBRetrieval] CRITICAL: Unexpected error during retrieval: {e}"
            )
            print(error_msg)
            import traceback

            traceback.print_exc()
            raise RuntimeError(f"CRITICAL RETRIEVAL ERROR: {error_msg}") from e

    def get_strategy_name(self) -> str:
        return "lancedb"

    def check_retrieval_possible(
        self, task_name: str, variation_idx: Any = None, **kwargs
    ) -> Tuple[bool, int, Optional[str]]:
        """
        Check if retrieval is possible for the given episode before starting execution.

        This preflight check verifies that at least one candidate trajectory exists
        with the configured filter constraints. If no candidates exist, the episode
        should be skipped to avoid running without retrieval.

        Args:
            task_name: Task name for filtering
            variation_idx: Variation index for filtering (if applicable)
            **kwargs: Additional context parameters

        Returns:
            Tuple of (is_possible, candidate_count, filter_sql):
            - is_possible: True if at least 1 candidate exists
            - candidate_count: Number of candidates found (limited to 1 for efficiency)
            - filter_sql: The SQL WHERE clause used (for logging)
        """
        # Build context for filter construction
        context = {"task_name": task_name, "variation_idx": variation_idx, **kwargs}

        # Use client to check candidates
        return self.client.check_candidates_exist(context)
