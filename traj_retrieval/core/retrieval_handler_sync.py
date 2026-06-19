# traj_retrieval/core/retrieval_handler_sync.py
# Synchronous version of retrieval_handler.py
# Centralized retrieval handling for trajectory-based agents (SYNC MODE)

import json
import random
from typing import Tuple, Dict, Optional
import requests

from .retrieval_strategy import RetrievalManager, RetrievalResult
from ..utils.logging_util import (
    create_planner_debug_info,
    update_step_debug_with_retrieval,
)


class SyncRetrievalPlanner:
    """
    Synchronous version of RetrievalPlanner.
    Uses the LLM to decide whether a new retrieval is needed at the current step.
    Emits one of: "[Retrieval]" or "[NoRetrieval]".
    """

    def __init__(
        self,
        sync_client,  # SimpleSyncHTTPClient
        model_slug: str,
    ):
        self._sync_client = sync_client
        self._model = model_slug

    def decide(
        self,
        goal_text: str,
        observation: str,
        recent_history_str: str,
        trajectory_context: str,
    ) -> Tuple[str, Dict]:
        """
        Decide whether retrieval is needed based on current context (synchronous).

        Args:
            goal_text: Task goal
            observation: Current observation
            recent_history_str: Recent action history
            trajectory_context: Current retrieved trajectory context

        Returns:
            Tuple of (decision_tag, debug_info)
            decision_tag is either "[Retrieval]" or "[NoRetrieval]"
        """
        system_msg = (
            "You are a retrieval planner for an agent. Decide if a new retrieval "
            "of a successful trajectory is needed now. Consider the goal, recent "
            "observations, and the current retrieved trajectory context."
            "Focus on identifying if the current situation is similar to the retrieved trajectory."
            "If the current situation and recent history is similar to the retrieved trajectory, return [NoRetrieval]."
            "If the current situation and recent history is not similar to the retrieved trajectory, return [Retrieval]."
        )
        user_payload = {
            "goal": goal_text,
            "current_observation": observation,
            "recent_history": recent_history_str,
            "current_retrieved_context": trajectory_context,
        }
        schema = (
            "Return only one of these exact tags: [Retrieval] or [NoRetrieval]. "
            "Do not include any other text."
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": json.dumps(user_payload)},
            {"role": "user", "content": schema},
        ]

        # Prepare simplified debug info for planner using utility
        debug_info = create_planner_debug_info(user_payload, self._model)

        # Retry with exponential backoff (synchronous)
        max_retries = 6
        base_delay = 1.5
        max_delay = 30.0
        jitter = (0.2, 0.6)

        attempt = 0
        while True:
            try:
                if attempt > 0:
                    print(
                        f"[RETRY] Retrieval planner attempt {attempt + 1}/{max_retries + 1}",
                        flush=True,
                    )

                # Synchronous API call
                resp = self._sync_client.chat.completions.create(
                    model=self._model, messages=messages, temperature=0.0
                )

                txt = (resp.choices[0].message.content or "").strip()
                tag = "[Retrieval]" if "[Retrieval]" in txt else "[NoRetrieval]"

                debug_info.update(
                    {
                        "response": {
                            "raw_text": txt,
                            "decision": tag,
                            "model_used": self._model,
                        },
                        "success": True,
                    }
                )
                return tag, debug_info

            except Exception as e:
                msg = str(e).lower()
                is_429 = ("rate limit" in msg) or ("429" in msg)
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
                            f"[RETRY] Retrieval planner failed after {attempt + 1} attempts: {e}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[RETRY] Retrieval planner failed with non-retryable error: {e}",
                            flush=True,
                        )

                    # Return fallback
                    print(f"[Planner] Error deciding retrieval: {e}")
                    debug_info["errors"].append(f"Planner call failed: {str(e)}")
                    debug_info.update(
                        {
                            "response": {"raw_text": "", "decision": "[NoRetrieval]"},
                            "success": False,
                        }
                    )
                    return "[NoRetrieval]", debug_info

                # Calculate delay
                delay = min(base_delay * (2**attempt), max_delay)
                delay *= 1.0 + random.uniform(*jitter)

                print(
                    f"[BACKOFF] Retrieval planner retrying in {delay:.2f}s due to: {e}",
                    flush=True,
                )
                import time

                time.sleep(delay)
                attempt += 1


class SyncRetrievalHandler:
    """
    Synchronous version of RetrievalHandler.
    Centralized handler for all retrieval operations based on frequency strategy.

    This class encapsulates all retrieval logic, making it easier to maintain and test.
    It handles initialization of planners and manages when/how to retrieve based on strategy.

    This is the main interface between episode execution (sync_with_traj_rag.py)
    and the retrieval system (retrieval_strategy.py, lancedb_retrieval.py).
    """

    def __init__(
        self,
        frequency_strategy: str,
        retrieval_manager: RetrievalManager,
        sync_client=None,  # SimpleSyncHTTPClient
        model_slug: str = None,
        simulation_loader=None,  # NEW: SimulationActionsLoader for simulate_till_tk/pseudo_simulate_till_closest_tk
        simulate_step_k: int = None,  # NEW: Per-episode simulate_step_k
        target_state_info: dict = None,  # NEW: Target state info (observation + URL) for pseudo_simulate_till_closest_tk
        agentic_until_step_k: int = None,
        retrieve_once_step_k: int = None,
    ):
        """
        Initialize synchronous retrieval handler.

        Args:
            frequency_strategy: When to retrieve ("none", "t0", "every_10", "agentic", "agentic_until_tk", "retrieve_once_at_tk", "simulate_till_tk", "pseudo_simulate_till_closest_tk")
            retrieval_manager: RetrievalManager instance
            sync_client: SimpleSyncHTTPClient (required for agentic strategy)
            model_slug: Model identifier (required for agentic strategy)
            simulation_loader: SimulationActionsLoader instance (required for simulate_till_tk/pseudo_simulate_till_closest_tk)
            simulate_step_k: Step at which to retrieve (required for simulate_till_tk/pseudo_simulate_till_closest_tk)
            target_state_info: Target state info dict with "observation" and "url" keys (required for pseudo_simulate_till_closest_tk)
            agentic_until_step_k: For agentic_until_tk, use agentic retrieval only while python_step < k
            retrieve_once_step_k: For retrieve_once_at_tk, force exactly one retrieval when python_step == k
        """
        self.frequency_strategy = frequency_strategy
        self.retrieval_manager = retrieval_manager

        # Initialize planner for agentic strategy
        self.planner = None
        if frequency_strategy in {"agentic", "agentic_until_tk"}:
            if not all([sync_client, model_slug]):
                raise ValueError("Agentic strategy requires sync_client and model_slug")
            self.planner = SyncRetrievalPlanner(sync_client, model_slug)

        self.agentic_until_step_k = agentic_until_step_k
        if frequency_strategy == "agentic_until_tk":
            if agentic_until_step_k is None:
                raise ValueError(
                    "agentic_until_tk strategy requires agentic_until_step_k parameter"
                )
            if not isinstance(agentic_until_step_k, int) or agentic_until_step_k < 0:
                raise ValueError("agentic_until_step_k must be a non-negative integer")

        self.retrieve_once_step_k = retrieve_once_step_k
        if frequency_strategy == "retrieve_once_at_tk":
            if retrieve_once_step_k is None:
                raise ValueError(
                    "retrieve_once_at_tk strategy requires retrieve_once_step_k parameter"
                )
            if not isinstance(retrieve_once_step_k, int) or retrieve_once_step_k < 0:
                raise ValueError("retrieve_once_step_k must be a non-negative integer")
            self._retrieve_once_done = False

        # Track if we've done initial retrieval for t0
        self._t0_retrieved = False

        # NEW: For simulate_till_tk strategy
        self.simulation_loader = simulation_loader
        self.simulate_step_k = simulate_step_k
        if frequency_strategy == "simulate_till_tk":
            if simulation_loader is None:
                raise ValueError(
                    "simulate_till_tk strategy requires simulation_loader parameter"
                )
            if simulate_step_k is None:
                raise ValueError(
                    "simulate_till_tk strategy requires simulate_step_k parameter"
                )
            self._simulate_tk_retrieved = False  # Track if we've retrieved at step k

        # NEW: For pseudo_simulate_till_closest_tk strategy
        self.target_state_info = target_state_info
        self._pseudo_retrieval_triggered = (
            False  # Track if we've triggered retrieval (owned by this instance)
        )
        if frequency_strategy == "pseudo_simulate_till_closest_tk":
            if simulation_loader is None:
                raise ValueError(
                    "pseudo_simulate_till_closest_tk strategy requires simulation_loader parameter"
                )
            if simulate_step_k is None:
                raise ValueError(
                    "pseudo_simulate_till_closest_tk strategy requires simulate_step_k parameter"
                )
            if target_state_info is None:
                raise ValueError(
                    "pseudo_simulate_till_closest_tk strategy requires target_state_info parameter"
                )

    def get_retrieval_triggered_status(self) -> bool:
        """
        Get whether retrieval has been triggered for pseudo_simulate_till_closest_tk strategy.

        Returns:
            True if retrieval was triggered, False otherwise
        """
        return self._pseudo_retrieval_triggered

    def retrieve_if_needed(
        self,
        python_step: int,
        current_node,
        metadata: Dict,
        recent_history_str: str,
        trajectory_context: str,
        task_display: str,
        step_debug_info: Dict,
        env_handler=None,
    ) -> Optional[RetrievalResult]:
        """
        Unified retrieval function that handles all frequency strategies (synchronous).

        This is the single point of retrieval in the episode loop.

        Args:
            python_step: Current step number
            current_node: Current environment node
            metadata: Episode metadata
            recent_history_str: Recent history string
            trajectory_context: Current trajectory context
            task_display: Task display string for logging
            step_debug_info: Debug info dict to update
            env_handler: Environment handler for formatting (optional)

        Returns:
            RetrievalResult if retrieval was performed and successful, None otherwise
        """
        # Strategy: none - no retrieval
        if self.frequency_strategy == "none":
            step_debug_info["retrieval"] = {"strategy": "none"}
            return None

        # Strategy: t0 - retrieve only at step 0
        if self.frequency_strategy == "t0":
            if python_step == 0 and not self._t0_retrieved:
                self._t0_retrieved = True
                return self._perform_retrieval(
                    current_node=current_node,
                    metadata=metadata,
                    recent_history_str=recent_history_str,
                    task_display=task_display,
                    step_debug_info=step_debug_info,
                    max_steps_to_show=30,
                    strategy_name="t0",
                    python_step=python_step,
                    env_handler=env_handler,
                )
            return None

        # Strategy: every_10 - retrieve every 10 steps
        if self.frequency_strategy == "every_10":
            if python_step % 10 == 0:
                return self._perform_retrieval(
                    current_node=current_node,
                    metadata=metadata,
                    recent_history_str=recent_history_str,
                    task_display=task_display,
                    step_debug_info=step_debug_info,
                    max_steps_to_show=40,
                    strategy_name="every_10",
                    python_step=python_step,
                    env_handler=env_handler,
                )
            return None

        # Strategy: agentic - use planner to decide
        if self.frequency_strategy == "agentic":
            return self._handle_agentic_retrieval(
                current_node=current_node,
                metadata=metadata,
                recent_history_str=recent_history_str,
                trajectory_context=trajectory_context,
                task_display=task_display,
                step_debug_info=step_debug_info,
                python_step=python_step,
                env_handler=env_handler,
            )

        # Strategy: agentic_until_tk - agentic planner before step k, then no retrieval
        if self.frequency_strategy == "agentic_until_tk":
            if python_step < self.agentic_until_step_k:
                return self._handle_agentic_retrieval(
                    current_node=current_node,
                    metadata=metadata,
                    recent_history_str=recent_history_str,
                    trajectory_context=trajectory_context,
                    task_display=task_display,
                    step_debug_info=step_debug_info,
                    python_step=python_step,
                    env_handler=env_handler,
                )
            print(
                f"[{task_display}][Step {python_step}] ⊘ Agentic retrieval window ended "
                f"(agentic_until_step_k={self.agentic_until_step_k}); no retrieval",
                flush=True,
            )
            step_debug_info["retrieval"] = {
                "strategy": "agentic_until_tk",
                "agentic_until_step_k": self.agentic_until_step_k,
                "current_step": python_step,
                "no_retrieval": True,
                "phase": "post_agentic_window",
            }
            return None

        # Strategy: retrieve_once_at_tk - force exactly one retrieval at step k
        if self.frequency_strategy == "retrieve_once_at_tk":
            if (
                python_step == self.retrieve_once_step_k
                and not self._retrieve_once_done
            ):
                self._retrieve_once_done = True
                print(
                    f"[RetrievalHandler] 🎯 retrieve_once_at_tk: Forcing one-time retrieval at step k={self.retrieve_once_step_k}",
                    flush=True,
                )
                return self._perform_retrieval(
                    current_node=current_node,
                    metadata=metadata,
                    recent_history_str=recent_history_str,
                    task_display=task_display,
                    step_debug_info=step_debug_info,
                    max_steps_to_show=30,
                    strategy_name="retrieve_once_at_tk",
                    python_step=python_step,
                    env_handler=env_handler,
                )
            step_debug_info["retrieval"] = {
                "strategy": "retrieve_once_at_tk",
                "retrieve_once_step_k": self.retrieve_once_step_k,
                "current_step": python_step,
                "status": "already_retrieved"
                if self._retrieve_once_done
                else "waiting_for_step_k",
            }
            return None

        # Strategy: simulate_till_tk - retrieve ONCE at step k only
        if self.frequency_strategy == "simulate_till_tk":
            if python_step == self.simulate_step_k and not self._simulate_tk_retrieved:
                self._simulate_tk_retrieved = True
                print(
                    f"[RetrievalHandler] 🎯 simulate_till_tk: Retrieving at step k={self.simulate_step_k}"
                )
                return self._perform_retrieval(
                    current_node=current_node,
                    metadata=metadata,
                    recent_history_str=recent_history_str,
                    task_display=task_display,
                    step_debug_info=step_debug_info,
                    max_steps_to_show=30,
                    strategy_name="simulate_till_tk",
                    python_step=python_step,
                    env_handler=env_handler,
                )
            else:
                # Not at step k, or already retrieved at step k
                step_debug_info["retrieval"] = {
                    "strategy": "simulate_till_tk",
                    "skipped": "not_at_step_k",
                }
                return None

        # Strategy: pseudo_simulate_till_closest_tk - retrieve ONCE when observation matches target
        if self.frequency_strategy == "pseudo_simulate_till_closest_tk":
            # Check if we've already triggered retrieval (use instance variable)
            if self._pseudo_retrieval_triggered:
                # Already retrieved, don't retrieve again
                step_debug_info["retrieval"] = {
                    "strategy": "pseudo_simulate_till_closest_tk",
                    "skipped": "already_retrieved",
                }
                return None

            # Check if current state matches target state
            if env_handler is None:
                print(
                    f"[RetrievalHandler] ⚠️  pseudo_simulate_till_closest_tk: env_handler is None, cannot check state similarity"
                )
                step_debug_info["retrieval"] = {
                    "strategy": "pseudo_simulate_till_closest_tk",
                    "skipped": "no_env_handler",
                }
                return None

            # Get current URL from env_handler
            current_url = env_handler.get_current_url()

            # Preprocess observations for comparison and storage (remove element IDs)
            preprocessed_current_obs = env_handler._preprocess_observation(
                current_node.observation
            )
            preprocessed_target_obs = env_handler._preprocess_observation(
                self.target_state_info["observation"]
            )

            # Calculate similarity score for logging
            obs_similarity_score = env_handler.compute_observation_similarity(
                preprocessed_current_obs, preprocessed_target_obs
            )

            # Build current state info (ONLY preprocessed observations)
            current_state_info = {
                "observation": preprocessed_current_obs,
                "url": current_url,
            }

            # Build target state info (ONLY preprocessed observations)
            target_state_info_debug = {
                "observation": preprocessed_target_obs,
                "url": self.target_state_info["url"],
            }

            # Use env_handler to check state similarity (observation + URL)
            should_trigger = env_handler.should_trigger_retrieval(
                current_observation=current_node.observation,
                target_observation=self.target_state_info["observation"],
                similarity_threshold=0.95,  # 95% similarity threshold
                current_url=current_url,
                target_url=self.target_state_info["url"],
            )

            # Check individual matches for detailed logging
            obs_matches = obs_similarity_score >= 0.95
            url_matches = (
                (current_url == self.target_state_info["url"])
                if self.target_state_info["url"]
                else False
            )

            # Log state comparison at each step for debugging
            obs_preview_current = (
                current_node.observation[:100] if current_node.observation else "None"
            )
            obs_preview_target = (
                self.target_state_info["observation"][:100]
                if self.target_state_info["observation"]
                else "None"
            )
            print(
                f"[pseudo_simulate_till_closest_tk][Step {python_step}] Checking state similarity..."
            )
            print(f"  Current observation preview: {obs_preview_current}...")
            print(f"  Target observation preview:  {obs_preview_target}...")
            print(f"  Current URL: {current_url if current_url else '(empty)'}")
            print(
                f"  Target URL:  {self.target_state_info['url'] if self.target_state_info['url'] else '(empty - not used for matching)'}"
            )
            print(f"  Match result: {'✓ MATCH' if should_trigger else '✗ NO MATCH'}")

            if should_trigger:
                # State matches - trigger retrieval ONCE
                self._pseudo_retrieval_triggered = True  # Update instance state
                print(f"\n{'='*100}")
                print(
                    f"🎯 pseudo_simulate_till_closest_tk: STATE MATCH DETECTED at step {python_step}"
                )
                print(f"{'='*100}")
                print(
                    f"[RetrievalHandler] Target step k from simulation data: {self.simulate_step_k}"
                )
                print(f"[RetrievalHandler] Actual trigger step: {python_step}")
                print(
                    f"[RetrievalHandler] State similarity: MATCH (using env_handler.should_trigger_retrieval)"
                )
                print(f"[RetrievalHandler] Matched state info:")
                print(f"[RetrievalHandler]   - Current state (preprocessed):")
                print(
                    f"[RetrievalHandler]     * Observation (first 200 chars): {preprocessed_current_obs[:200]}..."
                )
                print(f"[RetrievalHandler]     * URL: {current_url}")
                print(f"[RetrievalHandler]   - Target state (preprocessed):")
                print(
                    f"[RetrievalHandler]     * Observation (first 200 chars): {preprocessed_target_obs[:200]}..."
                )
                print(f"[RetrievalHandler]     * URL: {self.target_state_info['url']}")
                print(f"[RetrievalHandler]   - Similarity metrics:")
                print(
                    f"[RetrievalHandler]     * Edit distance score: {obs_similarity_score:.4f} ({obs_similarity_score*100:.2f}%)"
                )
                print(
                    f"[RetrievalHandler]     * Observation match: {'✓ YES' if obs_matches else '✗ NO'}"
                )
                print(
                    f"[RetrievalHandler]     * URL match: {'✓ YES' if url_matches else '✗ NO'}"
                )
                print(
                    f"[RetrievalHandler] 🔍 Performing ONE-TIME RETRIEVAL (just like simulate_till_tk)"
                )
                print(f"{'='*100}\n")

                # Store match info in a separate key that won't be overwritten
                # This will be merged back into retrieval key after _perform_retrieval
                step_debug_info["_pseudo_match_info"] = {
                    "matched_at_step": python_step,
                    "target_step_k": self.simulate_step_k,
                    "similarity_metrics": {
                        "edit_distance_score": round(obs_similarity_score, 4),
                        "edit_distance_percentage": round(
                            obs_similarity_score * 100, 2
                        ),
                        "observation_match": obs_matches,
                        "url_match": url_matches,
                        "threshold": 0.95,
                    },
                    "current_state_info": current_state_info,
                    "target_state_info": target_state_info_debug,
                }

                return self._perform_retrieval(
                    current_node=current_node,
                    metadata=metadata,
                    recent_history_str=recent_history_str,
                    task_display=task_display,
                    step_debug_info=step_debug_info,
                    max_steps_to_show=30,
                    strategy_name="pseudo_simulate_till_closest_tk",
                    python_step=python_step,
                    env_handler=env_handler,
                )
            else:
                # State doesn't match yet - continue without retrieval
                # Log monitoring status for debugging
                step_debug_info["retrieval"] = {
                    "strategy": "pseudo_simulate_till_closest_tk",
                    "skipped": "state_mismatch",
                    "monitoring": True,
                    "target_step_k": self.simulate_step_k,
                    "current_step": python_step,
                    "similarity_metrics": {
                        "edit_distance_score": round(obs_similarity_score, 4),
                        "edit_distance_percentage": round(
                            obs_similarity_score * 100, 2
                        ),
                        "observation_match": obs_matches,
                        "url_match": url_matches,
                        "threshold": 0.95,
                    },
                    "current_state_info": current_state_info,  # Preprocessed observations
                    "target_state_info": target_state_info_debug,  # Preprocessed observations
                }
                return None

        # Unknown strategy
        print(
            f"[RetrievalHandler] Warning: Unknown frequency_strategy '{self.frequency_strategy}'"
        )
        step_debug_info["retrieval"] = {
            "error": f"Unknown strategy: {self.frequency_strategy}"
        }
        return None

    def _perform_retrieval(
        self,
        current_node,
        metadata: Dict,
        recent_history_str: str,
        task_display: str,
        step_debug_info: Dict,
        max_steps_to_show: int,
        strategy_name: str,
        python_step: int,
        env_handler=None,
        planner_tag: str = "",
    ) -> Optional[RetrievalResult]:
        """
        Common retrieval logic used by all strategies (synchronous).

        Args:
            current_node: Current environment node
            metadata: Episode metadata
            recent_history_str: Recent history string
            task_display: Task display string for logging
            step_debug_info: Debug info dict to update
            max_steps_to_show: Max steps to show in context
            strategy_name: Name of strategy (for logging)
            python_step: Current step number
            env_handler: Environment handler for formatting
            planner_tag: Planner decision tag (for agentic)

        Returns:
            RetrievalResult or None
        """
        print(f"\n{'┄'*100}", flush=True)
        print(
            f"📚 RETRIEVAL ({strategy_name}): [{task_display}][Step {python_step}]",
            flush=True,
        )
        print(f"{'┄'*100}", flush=True)

        # Delegate query building and retrieval to the strategy
        try:
            retrieval_result = self.retrieval_manager.retrieve(
                env_handler=env_handler,
                task_name=metadata["task_name"],
                variation_idx=metadata["variation"],  # Pass variation_idx for filtering
                max_steps_to_show=max_steps_to_show,
                # Context for query building
                goal=current_node.goal,
                observation=current_node.observation,
                inventory=current_node.inventory,
                recent_history=recent_history_str,
            )
        except RuntimeError as e:
            # Critical retrieval error - re-raise to stop evaluation
            error_msg = str(e)
            if "CRITICAL RETRIEVAL ERROR" in error_msg:
                print(f"\n{'='*100}", flush=True)
                print(
                    f"🚨 CRITICAL RETRIEVAL FAILURE: [{task_display}][Step {python_step}]",
                    flush=True,
                )
                print(f"{'='*100}", flush=True)
                print(f"❌ {error_msg}", flush=True)
                print(
                    f"\n⛔ Stopping evaluation - retrieval is required but failed critically.",
                    flush=True,
                )
                print(
                    f"   This error indicates a fundamental issue with the retrieval system:",
                    flush=True,
                )
                print(f"   - Database/index not found or corrupted", flush=True)
                print(f"   - Vector index not created properly", flush=True)
                print(f"   - Configuration error", flush=True)
                print(
                    f"\n   Please fix the retrieval system before continuing.",
                    flush=True,
                )
                print(f"{'='*100}\n", flush=True)
                raise  # Re-raise to stop the entire evaluation
            else:
                # Non-critical runtime error - treat as retrieval failure
                print(
                    f"[{task_display}][Step {python_step}] ⚠ Retrieval failed: {e}",
                    flush=True,
                )
                retrieval_result = None

        if retrieval_result:
            traj_length = retrieval_result.metadata.get("trajectory_length", 0)
            action_obs_pairs = retrieval_result.metadata.get("action_obs_pairs", [])
            thought_id = retrieval_result.metadata.get("thought_id", "N/A")
            similarity_score = retrieval_result.metadata.get("similarity_score", 0.0)
            rank_retrieve = retrieval_result.metadata.get("rank_retrieve", 1)
            top_k = retrieval_result.metadata.get("top_k", 100)
            source_field = retrieval_result.metadata.get("retrieved_source_field")
            source_value = retrieval_result.metadata.get("retrieved_source_value")

            # Get top-k stats (flexible keys for backward compatibility)
            mean_top_k = retrieval_result.metadata.get(
                f"top_{top_k}_mean_distance",
                retrieval_result.metadata.get("top_100_mean_distance", 0.0),
            )
            std_top_k = retrieval_result.metadata.get(
                f"top_{top_k}_std_distance",
                retrieval_result.metadata.get("top_100_std_distance", 0.0),
            )

            # Update debug info with full retrieval metadata
            update_step_debug_with_retrieval(
                step_debug_info,
                strategy_name,
                planner_tag,
                action_obs_pairs,
                retrieval_result.formatted_context,
                triggered_by_planner=(strategy_name == "agentic"),
                retrieval_metadata=retrieval_result.metadata,  # Pass full metadata
            )

            # For pseudo_simulate_till_closest_tk, merge match info back into retrieval key
            if (
                strategy_name == "pseudo_simulate_till_closest_tk"
                and "_pseudo_match_info" in step_debug_info
            ):
                match_info = step_debug_info.pop("_pseudo_match_info")
                step_debug_info["retrieval"].update(match_info)

            print(
                f"[{task_display}][Step {python_step}] ✓ Retrieved trajectory ID: {thought_id}, length: {traj_length} steps, score: {similarity_score:.4f}, rank: {rank_retrieve}/{top_k}",
                flush=True,
            )
            if source_field:
                print(
                    f"[{task_display}][Step {python_step}] 🏷 Retrieved source ({source_field}): {source_value}",
                    flush=True,
                )
            print(
                f"[{task_display}][Step {python_step}] 📊 Top-{top_k} stats: mean={mean_top_k:.4f}, std={std_top_k:.4f}",
                flush=True,
            )
        else:
            print(
                f"[{task_display}][Step {python_step}] ⚠ No trajectory found",
                flush=True,
            )

        print(f"{'┄'*100}\n", flush=True)
        return retrieval_result

    def _handle_agentic_retrieval(
        self,
        current_node,
        metadata: Dict,
        recent_history_str: str,
        trajectory_context: str,
        task_display: str,
        step_debug_info: Dict,
        python_step: int,
        env_handler=None,
    ) -> Optional[RetrievalResult]:
        """
        Handle agentic retrieval strategy (use planner to decide) - synchronous.

        Args:
            current_node: Current environment node
            metadata: Episode metadata
            recent_history_str: Recent history string
            trajectory_context: Current trajectory context
            task_display: Task display string for logging
            step_debug_info: Debug info dict to update
            python_step: Current step number
            env_handler: Environment handler for formatting

        Returns:
            RetrievalResult or None
        """
        print(f"\n{'┄'*100}", flush=True)
        print(
            f"🤖 AGENTIC RETRIEVAL PLANNER: [{task_display}][Step {python_step}]",
            flush=True,
        )
        print(f"{'┄'*100}", flush=True)

        try:
            # Step 1: Planner decides (synchronous)
            tag, planner_debug = self.planner.decide(
                goal_text=current_node.goal,
                observation=current_node.observation,
                recent_history_str=recent_history_str,
                trajectory_context=trajectory_context,
            )
            step_debug_info["planner"] = planner_debug
            print(
                f"[{task_display}][Step {python_step}] 🎲 Planner Decision: {tag}",
                flush=True,
            )

            # Step 2: Retrieve if planner decided to
            if tag == "[Retrieval]":
                # Close planner header and open retrieval header
                print(f"{'┄'*100}\n", flush=True)
                return self._perform_retrieval(
                    current_node=current_node,
                    metadata=metadata,
                    recent_history_str=recent_history_str,
                    task_display=task_display,
                    step_debug_info=step_debug_info,
                    max_steps_to_show=10,
                    strategy_name="agentic",
                    python_step=python_step,
                    env_handler=env_handler,
                    planner_tag=tag,
                )
            else:
                print(
                    f"[{task_display}][Step {python_step}] ⊘ No retrieval needed",
                    flush=True,
                )
                step_debug_info["retrieval"] = {
                    "triggered_by_planner": True,
                    "planner_decision": tag,
                    "no_retrieval": True,
                }
                print(f"{'┄'*100}\n", flush=True)
                return None

        except Exception as e:
            step_debug_info["planner"] = {"errors": [str(e)]}
            step_debug_info["retrieval"] = {
                "error": "Planner failed, no retrieval attempted"
            }
            print(
                f"[{task_display}][Step {python_step}] ❌ Planner Error: {e}", flush=True
            )
            print(f"{'┄'*100}\n", flush=True)
            return None
