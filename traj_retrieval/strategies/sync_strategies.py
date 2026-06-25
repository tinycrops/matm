# traj_retrieval/strategies/sync_strategies.py
# Sync strategy implementations for LLM experiments

import json
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional

# Import base classes
from .base_strategy import BaseExperimentStrategy, HybridRerankingMixin

# Import utilities
from ..utils.logging_util import write_llm_debug_file
from ..utils.metrics_util import analyze_episode_errors


class SyncBaseExperimentStrategy(BaseExperimentStrategy):
    """
    Base class for sync strategies.
    Adds the sync execute() method to the base interface.
    """

    @abstractmethod
    def execute(
        self,
        sclient,
        limiter,
        requests_sem,
        model: str,
        goal_text: str,
        observation: str,
        inventory: str,
        admissible_actions: List[str],
        recent_history_str: str,
        trajectory_context: str,
        debug_info: Dict,
        env_handler,
        ultimate_fallback_action: str = "look around",
        episode_id: str = "unknown",
        **kwargs,
    ) -> Tuple[str, Dict]:
        """
        Execute the strategy and return (action, debug_info).
        This is the main entry point for each sync strategy.

        Args:
            env_handler: Environment handler for environment-specific operations
            episode_id: Episode identifier for logging
        """
        pass


class ActionStringSyncStrategy(SyncBaseExperimentStrategy, HybridRerankingMixin):
    """
    Strategy that uses action strings directly without integer mapping.
    Flow:
    1. Try with all actions (retry up to 3 times for any error including inadmissible actions)
    2. If still fails -> hybrid reranking (BM25 top 100 + Qwen reranker top 400) -> structured JSON with filtered actions (retry up to 3 times)
    3. If JSON parse error -> ultimate fallback (look around)
    4. If API error -> ultimate fallback (look around)
    """

    def __init__(self):
        """Initialize the strategy with hybrid reranking capability."""
        BaseExperimentStrategy.__init__(self)
        HybridRerankingMixin.__init__(self)

    def get_strategy_name(self) -> str:
        return "action_string_direct"

    def _log_payload_debug(
        self, payload: Dict[str, Any], stage: str = "initial"
    ) -> None:
        """Log payload structure with intelligent truncation for readability."""
        print(f"\n{'='*80}")
        print(f"[PAYLOAD-DEBUG] {stage.upper()} USER PAYLOAD")
        print(f"{'='*80}")

        for key, value in payload.items():
            if isinstance(value, str):
                char_count = len(value)
                token_estimate = char_count // 4  # Rough estimate: 1 token ≈ 4 chars
                preview = value[:100].replace("\n", "\\n")
                if char_count > 100:
                    preview += "..."
                print(f"  {key}:")
                print(f"    Preview: {preview}")
                print(f"    Stats: {char_count} chars, ~{token_estimate} tokens")
            elif isinstance(value, list):
                print(f"  {key}: {len(value)} items")
                if len(value) <= 5:
                    print(f"    All items: {value}")
                else:
                    print(f"    First 3: {value[:3]}")
                    print(f"    Last 2: {value[-2:]}")
            elif isinstance(value, dict):
                print(f"  {key}: (nested dict)")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, str):
                        char_count = len(sub_value)
                        token_estimate = char_count // 4
                        preview = sub_value[:100].replace("\n", "\\n")
                        if char_count > 100:
                            preview += "..."
                        print(f"    {sub_key}:")
                        print(f"      Preview: {preview}")
                        print(
                            f"      Stats: {char_count} chars, ~{token_estimate} tokens"
                        )
                    else:
                        print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")

        print(f"{'='*80}\n")

    def build_user_payload(
        self,
        goal_text: str,
        recent_history_str: str,
        observation: str,
        inventory: str,
        admissible_actions: List[str],
        trajectory_context: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Build user payload with clear context descriptions.

        Args:
            goal_text: The goal/task to accomplish
            recent_history_str: History of previous steps (observation, action, reward)
            observation: Current observation
            inventory: Current inventory
            admissible_actions: List of valid actions
            trajectory_context: Optional retrieved trajectory for guidance
            **kwargs: Additional context like current_step, max_steps
        """
        # Build the base payload
        payload = {
            "goal": goal_text,
            "current_observation": observation,
            "admissible_actions": admissible_actions,
        }
        if inventory is not None and inventory != "":
            payload["inventory"] = inventory

        # Add step information if available
        if "current_step" in kwargs and "max_steps" in kwargs:
            payload["current_step"] = kwargs["current_step"]
            payload["max_steps"] = kwargs["max_steps"]

        # Add recent history with clear description
        if recent_history_str and recent_history_str.strip():
            payload["recent_history"] = {
                "description": "These are the observation, action, and reward information from your previous steps in this episode. This is very critical for you so that you can decide your next steps given your previous actions and observations.",
                "history": recent_history_str,
            }

        # Only add trajectory context if it exists
        if trajectory_context and trajectory_context.strip():
            payload["retrieved_trajectory_guidance"] = {
                "description": "This is a retrieved trajectory from a successful past episode that may help guide your decision. Use it as reference for planning your next action.",
                "trajectory": trajectory_context,
            }

        # Log the payload structure for debugging
        self._log_payload_debug(payload, stage="initial")

        return payload

    def build_schema_message(self) -> str:
        return (
            "IMPORTANT: Return ONLY valid JSON in this exact format:\n"
            '{"reasoning": "detailed step-by-step reasoning about the situation and why this action is best", "action": "exact action from admissible_actions"}\n'
            "Where:\n"
            "- reasoning provides detailed analysis of the current situation and goal\n"
            "- action is EXACTLY one of the strings from the admissible_actions list above\n"
            "Do not include any other text before or after the JSON."
        )

    class ActionParsingError(Exception):
        """Custom exception to distinguish between JSON parsing errors and invalid action errors."""

        def __init__(self, message, error_type, reasoning=None):
            super().__init__(message)
            self.error_type = error_type  # 'json_parse' or 'invalid_action'
            self.reasoning = reasoning

    def parse_response(
        self,
        response_text: str,
        admissible_actions: List[str],
        user_payload: Optional[Dict] = None,
        env_handler=None,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Parse action string response format with reasoning.

        IMPORTANT: Our parsing strategy (may differ from vanilla WebArena):
        - REASONING: Free-form text BEFORE "In summary, the next action I will perform is"
        - ACTION: Content between ``` delimiters after "In summary"

        Expected format:
        "Let's think step-by-step. [free-form reasoning here].
         In summary, the next action I will perform is ```action```"

        This gives us clean separation between thinking/reasoning and the actual action.
        """

        # FIRST: Try WebArena format with ``` delimiters
        # Extract action from ``` block (after "In summary" phrase)
        import re

        pattern = r"```(.*?)```"
        matches = re.findall(pattern, response_text, re.DOTALL)

        if matches:
            # Extract action from the first ``` block
            action = matches[0].strip()
            print(
                f"[PARSE-DEBUG] Extracted action from ``` delimiters: '{action}'",
                flush=True,
            )

            # Extract reasoning: FREE-FORM text BEFORE "In summary"
            # This is our intentional design - full reasoning without the action
            reasoning = response_text
            summary_idx = response_text.lower().find("in summary")
            if summary_idx > 0:
                reasoning = response_text[:summary_idx].strip()
            else:
                # If no "In summary" found, use everything before the action
                reasoning = response_text.split("```")[0].strip()

            # Check if reasoning is too short (likely missing step-by-step thinking)
            if len(reasoning) < 100 or not reasoning.lower().startswith("let"):
                print(
                    f"[PARSE-WARNING] ⚠️  Reasoning seems incomplete (only {len(reasoning)} chars, should have 'Let's think step-by-step' analysis)",
                    flush=True,
                )
                print(
                    f"[PARSE-WARNING] ⚠️  LLM may not be following instructions to think step-by-step",
                    flush=True,
                )
                print(
                    f"[PARSE-WARNING] ⚠️  Reasoning content: '{reasoning}'", flush=True
                )
            else:
                print(
                    f"[PARSE-DEBUG] ✓ Extracted reasoning (free-form, {len(reasoning)} chars): {reasoning[:200]}...",
                    flush=True,
                )

            # Validate the action using environment-specific validation
            is_valid = False
            if env_handler is not None:
                is_valid = env_handler.validate_action(action, admissible_actions)
                print(
                    f"[PARSE-DEBUG] Action validation (env-specific): {is_valid}",
                    flush=True,
                )
            else:
                # Fallback to exact match if no env_handler
                is_valid = action in admissible_actions
                print(
                    f"[PARSE-DEBUG] Action validation (exact match): {is_valid}",
                    flush=True,
                )

            if not is_valid:
                # This is a successful parse but invalid action - pass reasoning for fallback
                raise self.ActionParsingError(
                    f"Model returned non-admissible action: '{action}'. Available actions: {len(admissible_actions)} total",
                    "invalid_action",
                    reasoning=reasoning,
                )

            additional_info = {
                "total_admissible_actions": len(admissible_actions),
                "reasoning": reasoning,
            }

            return action, additional_info

        # SECOND: Try to extract JSON from the response (fallback)
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}")

        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            raise self.ActionParsingError(
                "No valid action found in response (neither ``` format nor JSON)",
                "json_parse",
            )

        json_str = response_text[start_idx : end_idx + 1]

        # Try to parse the JSON
        try:
            data = json.loads(json_str)
            reasoning = data.get("reasoning", "")
            action = data.get("action", "")

            # Log the extracted reasoning and action
            print(f"[PARSE-DEBUG] Extracted reasoning: {reasoning}", flush=True)
            print(f"[PARSE-DEBUG] Extracted action: '{action}'", flush=True)

            # Validate the action using environment-specific validation
            is_valid = False
            if env_handler is not None:
                is_valid = env_handler.validate_action(action, admissible_actions)
                print(
                    f"[PARSE-DEBUG] Action validation (env-specific): {is_valid}",
                    flush=True,
                )
            else:
                # Fallback to exact match if no env_handler
                is_valid = action in admissible_actions
                print(
                    f"[PARSE-DEBUG] Action validation (exact match): {is_valid}",
                    flush=True,
                )

            if not is_valid:
                # This is a successful parse but invalid action - pass reasoning for fallback
                raise self.ActionParsingError(
                    f"Model returned non-admissible action: '{action}'. Available actions: {len(admissible_actions)} total",
                    "invalid_action",
                    reasoning=reasoning,
                )

            additional_info = {
                "total_admissible_actions": len(admissible_actions),
                "reasoning": reasoning,
            }

            return action, additional_info

        except json.JSONDecodeError as json_err:
            # Try to find a complete JSON object
            brace_count = 0
            for i, char in enumerate(json_str):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        # Found complete JSON object
                        complete_json = json_str[: i + 1]
                        try:
                            data = json.loads(complete_json)
                            reasoning = data.get("reasoning", "")
                            action = data.get("action", "")

                            # Validate the action using environment-specific validation
                            is_valid = False
                            if env_handler is not None:
                                is_valid = env_handler.validate_action(
                                    action, admissible_actions
                                )
                            else:
                                is_valid = action in admissible_actions

                            if not is_valid:
                                # This is a successful parse but invalid action - pass reasoning for fallback
                                raise self.ActionParsingError(
                                    f"Model returned non-admissible action: '{action}'. Available actions: {len(admissible_actions)} total",
                                    "invalid_action",
                                    reasoning=reasoning,
                                )

                            additional_info = {
                                "total_admissible_actions": len(admissible_actions),
                                "reasoning": reasoning,
                                "fallback_parsing": True,
                            }

                            return action, additional_info

                        except json.JSONDecodeError:
                            continue

            # If we get here, no valid JSON was found
            raise self.ActionParsingError(
                "JSON parsing failed completely", "json_parse"
            )

    def get_structured_response_format(
        self, filtered_actions: List[str]
    ) -> Dict[str, Any]:
        """Response format for the structured fallback call (with enum constraint on filtered actions)."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "agent_action",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "detailed step-by-step reasoning about the situation and why this action is the best choice",
                        },
                        "action": {"type": "string", "enum": filtered_actions},
                    },
                    "required": ["reasoning", "action"],
                    "additionalProperties": False,
                },
            },
        }

    def parse_structured_fallback_response(
        self,
        response_text: str,
        filtered_actions: List[str],
        debug_info: Optional[Dict] = None,
        model: str = "unknown",
        ultimate_fallback_action: str = "look around",
        env_handler=None,
    ) -> Tuple[str, str]:
        """Parse the structured fallback response and return reasoning and action."""
        try:
            data = json.loads(response_text)
            reasoning = data.get("reasoning", "")
            action = data.get("action", "")

            print(
                f"[ACTION-STRING] Structured fallback parse debug: Extracted action: '{action}'",
                flush=True,
            )
            print(
                f"[ACTION-STRING] Structured fallback parse debug: Filtered actions count: {len(filtered_actions)}",
                flush=True,
            )

            # Validate the action using environment-specific validation
            is_valid = False
            if env_handler is not None:
                is_valid = env_handler.validate_action(action, filtered_actions)
                print(
                    f"[ACTION-STRING] Structured action validation (env-specific): {is_valid}",
                    flush=True,
                )
            else:
                is_valid = action in filtered_actions
                print(
                    f"[ACTION-STRING] Structured action validation (exact match): {is_valid}",
                    flush=True,
                )

            if not is_valid:
                print(
                    f"[ACTION-STRING] Structured fallback parse debug: Action '{action}' NOT found in filtered actions",
                    flush=True,
                )
                if len(filtered_actions) <= 20:
                    print(
                        f"[ACTION-STRING] Structured fallback parse debug: All filtered actions: {filtered_actions}",
                        flush=True,
                    )
                else:
                    print(
                        f"[ACTION-STRING] Structured fallback parse debug: First 10 filtered actions: {filtered_actions[:10]}",
                        flush=True,
                    )
                    print(
                        f"[ACTION-STRING] Structured fallback parse debug: Last 10 filtered actions: {filtered_actions[-10:]}",
                        flush=True,
                    )
                raise ValueError(
                    f"Selected action not in filtered list: '{action}'. Available actions: {len(filtered_actions)} total"
                )

            return reasoning, action

        except (json.JSONDecodeError, ValueError) as e:
            print(f"[WARNING] Failed to parse structured fallback response: {e}")

            # Add warning to debug_info if provided
            if debug_info is not None:
                debug_info["warnings"] = debug_info.get("warnings", [])
                debug_info["warnings"].append(
                    {
                        "stage": "structured_fallback_response_parsing",
                        "message": f"Failed to parse structured fallback response: {str(e)}",
                        "response_text_length": len(response_text),
                        "filtered_actions_count": len(filtered_actions),
                        "fallback_used": True,
                    }
                )

            # Always use configured ultimate fallback action
            action = ultimate_fallback_action

            # Log warning if fallback is not in filtered actions (for debugging)
            if filtered_actions and ultimate_fallback_action not in filtered_actions:
                print(
                    f"[ACTION-STRING] ⚠️  WARNING: Configured fallback '{ultimate_fallback_action}' not in filtered actions (count: {len(filtered_actions)})",
                    flush=True,
                )
                print(
                    f"[ACTION-STRING] Using it anyway - validation will happen in environment handler if needed",
                    flush=True,
                )
            else:
                print(
                    f"[ACTION-STRING] Using configured fallback: '{action}'", flush=True
                )

            return (
                f"Fallback: structured parsing failed, using {ultimate_fallback_action} action",
                action,
            )

    def get_ultimate_fallback_action(
        self,
        admissible_actions: List[str],
        ultimate_fallback_action: str = "look around",
        **kwargs,
    ) -> Tuple[str, str]:
        """Get ultimate fallback action when all else fails.

        Always returns the configured ultimate_fallback_action regardless of whether
        it's in the admissible_actions list. Validation (if needed) happens elsewhere.

        Returns:
            Tuple of (reasoning, action)
        """

        print(
            f"[ACTION-STRING] Using ultimate fallback for action selection", flush=True
        )

        action = ultimate_fallback_action
        reasoning = f"Agent could not decide the action so falling back to {ultimate_fallback_action} action"

        # Log warning if fallback is not in admissible actions (for debugging)
        if (
            len(admissible_actions) > 0
            and ultimate_fallback_action not in admissible_actions
        ):
            print(
                f"[ACTION-STRING] ⚠️  WARNING: Configured fallback '{ultimate_fallback_action}' not in admissible actions list (count: {len(admissible_actions)})",
                flush=True,
            )
            print(
                f"[ACTION-STRING] Using it anyway - validation will happen in environment handler if needed",
                flush=True,
            )
        else:
            print(f"[ACTION-STRING] Using configured fallback: '{action}'", flush=True)

        return reasoning, action

    def get_fallback_action(
        self,
        admissible_actions: List[str],
        ultimate_fallback_action: str = "look around",
        **kwargs,
    ) -> Tuple[str, str]:
        """Get fallback action when parsing fails."""
        return self.get_ultimate_fallback_action(
            admissible_actions,
            ultimate_fallback_action=ultimate_fallback_action,
            **kwargs,
        )

    def _retry_llm_call_with_parsing(
        self,
        call_func,
        parse_func,
        context: str,
        max_retries: int = 2,
        base_delay: float = 1.0,
        episode_id: str = "unknown",
    ):
        """
        Retry wrapper that handles both API errors and JSON parsing errors.
        Makes total of (max_retries + 1) attempts with exponential backoff for ALL error types.
        Default max_retries=2 means 3 total attempts (1 initial + 2 retries).

        SYNC VERSION: No asyncio imports or async constructs.
        """
        import random

        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                if attempt > 0:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5)
                    print(
                        f"[ACTION-STRING:{episode_id}] 🔄 Retry {attempt}/{max_retries}: {context} (waiting {delay:.2f}s)",
                        flush=True,
                    )
                    time.sleep(delay)
                else:
                    print(
                        f"[ACTION-STRING:{episode_id}] Attempt {attempt + 1}/{max_retries + 1}: {context}",
                        flush=True,
                    )

                # Try the API call
                resp = call_func()

                # Try to parse the response
                result = parse_func(resp)
                if attempt > 0:
                    print(
                        f"[ACTION-STRING:{episode_id}] ✓ Retry successful on attempt {attempt + 1}",
                        flush=True,
                    )
                return result

            except Exception as e:
                is_last_attempt = attempt == max_retries
                error_type = type(e).__name__
                error_msg = str(e)  # NO TRUNCATION - show full error

                if is_last_attempt:
                    print(
                        f"[ACTION-STRING:{episode_id}] ❌ All retries exhausted after {attempt + 1} attempts",
                        flush=True,
                    )
                    print(
                        f"[ACTION-STRING:{episode_id}] Final error: {error_type} - {error_msg}",
                        flush=True,
                    )
                    raise e
                else:
                    # Log the error and retry (for any error type)
                    print(
                        f"[ACTION-STRING:{episode_id}] ⚠️  Attempt {attempt + 1} failed: {error_type} - {error_msg}",
                        flush=True,
                    )
                    continue

    def get_debug_info_keys(self) -> List[str]:
        return [
            "total_admissible_actions",
            "reasoning",
            "hybrid_reranking_fallback_used",
            "structured_fallback_used",
        ]

    def _get_execution_path(self, policy: str, num_actions: int) -> str:
        """
        Determine the execution path based on policy and number of actions.

        Args:
            policy: Agent call policy ("only_normal", "only_structured", "normal_then_structured", "adaptive")
            num_actions: Number of admissible actions

        Returns:
            Execution path: "only_normal", "only_structured", or "normal_then_structured"
        """
        if policy == "only_normal":
            return "only_normal"
        elif policy == "only_structured":
            return "only_structured"
        elif policy == "normal_then_structured":
            return "normal_then_structured"
        elif policy == "adaptive":
            # Adaptive: Use structured if < 500 actions, else normal_then_structured
            if num_actions < 500:
                return "only_structured"
            else:
                return "normal_then_structured"
        else:
            # Unknown policy, default to normal_then_structured
            print(
                f"[WARNING] Unknown agent_call_policy '{policy}', defaulting to 'normal_then_structured'",
                flush=True,
            )
            return "normal_then_structured"

    def _call_llm_normal(
        self,
        sclient,
        limiter,
        requests_sem,
        model: str,
        user_message: str,
        admissible_actions: List[str],
        user_payload: Dict,
        system_message: str = None,
        env_handler=None,
        episode_id: str = "unknown",
    ) -> Tuple[str, Dict]:
        """
        Helper method for normal (unstructured) LLM call.
        Returns (action, additional_info).

        SYNC VERSION: No async constructs. Uses synchronous API client.
        """

        def normal_call():
            # Build messages array with system message if provided
            messages = []
            if system_message and system_message.strip():
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": user_message})

            # Synchronous API call
            return sclient.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_completion_tokens=2000,
            )

        def normal_parse(response):
            response_text = response.choices[0].message.content
            # Handle None response
            if response_text is None:
                response_text = ""
            print(
                f"[ACTION-STRING:{episode_id}] Normal LLM response length: {len(response_text)} chars",
                flush=True,
            )
            print(
                f"[ACTION-STRING:{episode_id}] Normal LLM FULL response:\n{response_text}",
                flush=True,
            )
            return self.parse_response(
                response_text, admissible_actions, user_payload, env_handler
            )

        action, additional_info = self._retry_llm_call_with_parsing(
            normal_call,
            normal_parse,
            "Normal action selection",
            max_retries=2,  # 3 total attempts (1 initial + 2 retries)
            episode_id=episode_id,
        )

        return action, additional_info

    def _call_llm_structured(
        self,
        sclient,
        limiter,
        requests_sem,
        model: str,
        user_message: str,
        filtered_actions: List[str],
        debug_info: Dict,
        ultimate_fallback_action: str,
        system_message: str = None,
        env_handler=None,
        episode_id: str = "unknown",
    ) -> Tuple[str, str]:
        """
        Helper method for structured LLM call with response_format.
        Returns (reasoning, action).

        SYNC VERSION: No async constructs. Uses synchronous API client.
        """

        def structured_call():
            # Build messages array with system message if provided
            messages = []
            if system_message and system_message.strip():
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": user_message})

            # Synchronous API call
            return sclient.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_completion_tokens=2000,
                response_format=self.get_structured_response_format(filtered_actions),
            )

        def structured_parse(response):
            response_text = response.choices[0].message.content
            # Handle None response
            if response_text is None:
                response_text = ""
            print(
                f"[ACTION-STRING:{episode_id}] Structured LLM response length: {len(response_text)} chars",
                flush=True,
            )
            print(
                f"[ACTION-STRING:{episode_id}] Structured LLM FULL response:\n{response_text}",
                flush=True,
            )
            return self.parse_structured_fallback_response(
                response_text,
                filtered_actions,
                debug_info,
                model,
                ultimate_fallback_action,
                env_handler,
            )

        reasoning, action = self._retry_llm_call_with_parsing(
            structured_call,
            structured_parse,
            "Structured action selection",
            max_retries=2,  # 3 total attempts (1 initial + 2 retries)
            episode_id=episode_id,
        )

        return reasoning, action

    def execute(
        self,
        sclient,
        limiter,
        requests_sem,
        model: str,
        goal_text: str,
        observation: str,
        inventory: str,
        admissible_actions: List[str],
        recent_history_str: str,
        trajectory_context: str,
        debug_info: Dict,
        env_handler,
        ultimate_fallback_action: str = "look around",
        episode_id: str = "unknown",
        **kwargs,
    ) -> Tuple[str, Dict]:
        """
        Execute the action selection strategy based on environment-specific policy.

        EXECUTION PATHS (based on agent_call_policy):
        1. only_normal: Normal call → Ultimate fallback
        2. only_structured: Structured call → Ultimate fallback (requires admissible_actions)
        3. normal_then_structured: Normal → Structured with reranking → Ultimate fallback

        SPECIAL CASE - WebArena (no admissible actions):
        - WebArena provides empty admissible_actions list
        - LLM parses element IDs directly from observation
        - Falls back to only_normal path regardless of policy
        - Ultimate fallback is "none" action

        CRITICAL:
        - For environments with admissible_actions: fallback MUST be in list
        - For WebArena (empty list): fallback validated by env_handler
        """

        print(f"\n{'═'*100}", flush=True)
        print(f"🎯 ACTION-STRING STRATEGY: Starting execution", flush=True)
        print(f"{'═'*100}", flush=True)
        print(
            f"[ACTION-STRING:{episode_id}] Total Admissible Actions: {len(admissible_actions)}",
            flush=True,
        )

        # Get system message from environment handler
        system_message = None
        if env_handler is not None:
            try:
                # Pass admissible_actions to build_system_message (as per handler interface)
                system_message = env_handler.build_system_message(admissible_actions)
                print(
                    f"[ACTION-STRING:{episode_id}] ✓ Retrieved environment-specific system message (length: {len(system_message)} chars)",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[ACTION-STRING:{episode_id}] ⚠️ Failed to get system message from env_handler: {e}",
                    flush=True,
                )
                system_message = None

        # Get agent call policy and determine execution path
        agent_call_policy = "normal_then_structured"  # default
        if env_handler is not None:
            agent_call_policy = env_handler.get_agent_call_policy()

        execution_path = self._get_execution_path(
            agent_call_policy, len(admissible_actions)
        )

        # SPECIAL CASE: WebArena with no admissible actions
        # Force only_normal path since we can't do structured fallback without action list
        if len(admissible_actions) == 0:
            print(
                f"[ACTION-STRING:{episode_id}] Empty admissible actions (WebArena mode) - forcing only_normal path",
                flush=True,
            )
            execution_path = "only_normal"

        print(
            f"[ACTION-STRING:{episode_id}] Agent Call Policy: '{agent_call_policy}' → Execution Path: '{execution_path}'",
            flush=True,
        )

        # Build user payload with clear context
        user_payload = self.build_user_payload(
            goal_text=goal_text,
            recent_history_str=recent_history_str,
            observation=observation,
            inventory=inventory,
            admissible_actions=admissible_actions,
            trajectory_context=trajectory_context,
            **kwargs,  # Pass through step info and other kwargs
        )

        # Convert payload to readable format for LLM (delegate to environment handler)
        # REQUIRED: All environment handlers MUST implement build_user_message
        if env_handler is None:
            raise RuntimeError(
                "Environment handler is required but not provided. "
                "Cannot build user message without environment handler."
            )

        if not hasattr(env_handler, "build_user_message"):
            raise NotImplementedError(
                f"Environment handler {type(env_handler).__name__} must implement build_user_message() method. "
                f"This is required for proper prompt formatting."
            )

        user_message = env_handler.build_user_message(
            goal_text=goal_text,
            observation=observation,
            inventory=inventory,
            admissible_actions=admissible_actions,
            recent_history_str=recent_history_str,
            trajectory_context=trajectory_context,
            current_step=kwargs.get("current_step", 0),
            max_steps=kwargs.get("max_steps", 0),
        )

        # CRITICAL: Structure LLM debug info properly (nested under llm_call)
        # This ensures complete prompt reconstruction in debug files
        # Get current URL from handler if available (for web environments like WebArena)
        current_url = env_handler.get_current_url() if env_handler else ""

        debug_info["llm_call"] = {
            "prompt_components": {
                "goal": goal_text,
                "observation": observation,
                "inventory": inventory,
                "admissible_actions": admissible_actions[:10]
                if len(admissible_actions) > 10
                else admissible_actions,
                "total_admissible_actions": len(admissible_actions),
                "recent_history": recent_history_str,
                "trajectory_context": trajectory_context if trajectory_context else "",
                "trajectory_context_present": bool(trajectory_context.strip()),
                "url": current_url  # URL for web environments (WebArena), empty string otherwise
                # Note: reasoning is NOT here - it's the LLM response, stored in result.reasoning
            },
            "model": model,
            "system_message": system_message if system_message else "",
            "user_message": user_message,
            "errors": [],
            "warnings": [],
        }

        # Route to appropriate execution path
        if execution_path == "only_normal":
            return self._execute_only_normal(
                sclient,
                limiter,
                requests_sem,
                model,
                user_message,
                admissible_actions,
                user_payload,
                debug_info,
                ultimate_fallback_action,
                system_message,
                env_handler,
                episode_id,
            )
        elif execution_path == "only_structured":
            return self._execute_only_structured(
                sclient,
                limiter,
                requests_sem,
                model,
                user_message,
                admissible_actions,
                debug_info,
                ultimate_fallback_action,
                system_message,
                env_handler,
                episode_id,
            )
        else:  # normal_then_structured
            return self._execute_normal_then_structured(
                sclient,
                limiter,
                requests_sem,
                model,
                user_message,
                admissible_actions,
                user_payload,
                debug_info,
                goal_text,
                recent_history_str,
                observation,
                inventory,
                trajectory_context,
                ultimate_fallback_action,
                system_message,
                env_handler,
                episode_id,
            )

    def _execute_only_normal(
        self,
        sclient,
        limiter,
        requests_sem,
        model: str,
        user_message: str,
        admissible_actions: List[str],
        user_payload: Dict,
        debug_info: Dict,
        ultimate_fallback_action: str,
        system_message: str,
        env_handler,
        episode_id: str,
    ) -> Tuple[str, Dict]:
        """Execute only_normal path: Normal call → Ultimate fallback."""
        print(f"\n{'─'*100}", flush=True)
        print(f"📝 EXECUTION PATH: only_normal", flush=True)
        print(f"{'─'*100}", flush=True)

        try:
            print(
                f"[ACTION-STRING:{episode_id}] Calling normal LLM with {len(admissible_actions)} actions...",
                flush=True,
            )

            action, additional_info = self._call_llm_normal(
                sclient=sclient,
                limiter=limiter,
                requests_sem=requests_sem,
                model=model,
                user_message=user_message,
                admissible_actions=admissible_actions,
                user_payload=user_payload,
                system_message=system_message,
                env_handler=env_handler,
                episode_id=episode_id,
            )

            # Extract reasoning and add to llm_call structure
            reasoning = additional_info.get("reasoning", "")
            debug_info["llm_call"]["reasoning"] = reasoning
            debug_info["llm_call"]["response_metadata"] = {
                "total_admissible_actions": additional_info.get(
                    "total_admissible_actions", len(admissible_actions)
                ),
                "parsing_method": additional_info.get("parsing_method", "normal"),
            }
            debug_info[
                "reasoning"
            ] = reasoning  # Also keep at top level for backwards compatibility
            print(f"[ACTION-STRING:{episode_id}] ✅ Normal Call SUCCESS", flush=True)
            print(
                f"[ACTION-STRING:{episode_id}] Selected Action: '{action}'", flush=True
            )
            print(f"{'─'*100}\n", flush=True)
            return action, debug_info

        except Exception as error:
            print(
                f"[ACTION-STRING:{episode_id}] ❌ Normal Call FAILED: {type(error).__name__}",
                flush=True,
            )
            print(f"[ACTION-STRING:{episode_id}] Error: {str(error)}", flush=True)
            print(f"\n{'╳'*100}", flush=True)
            print(f"🔄 Ultimate Fallback", flush=True)
            print(f"{'╳'*100}", flush=True)
            reasoning, action = self.get_ultimate_fallback_action(
                admissible_actions, ultimate_fallback_action=ultimate_fallback_action
            )
            # Update llm_call structure with error and fallback info
            debug_info["llm_call"]["errors"].append(
                {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "stage": "normal_call",
                }
            )
            debug_info["llm_call"]["reasoning"] = reasoning
            debug_info["llm_call"]["ultimate_fallback_used"] = True
            debug_info["llm_call"][
                "ultimate_fallback_reason"
            ] = f"Normal call failed: {str(error)}"
            debug_info[
                "reasoning"
            ] = reasoning  # Also keep at top level for backwards compatibility
            print(
                f"[ACTION-STRING:{episode_id}] Ultimate Fallback Action: '{action}'",
                flush=True,
            )
            print(f"{'╳'*100}\n", flush=True)
            return action, debug_info

    def _execute_only_structured(
        self,
        sclient,
        limiter,
        requests_sem,
        model: str,
        user_message: str,
        admissible_actions: List[str],
        debug_info: Dict,
        ultimate_fallback_action: str,
        system_message: str,
        env_handler,
        episode_id: str,
    ) -> Tuple[str, Dict]:
        """Execute only_structured path: Structured call → Ultimate fallback."""
        print(f"\n{'─'*100}", flush=True)
        print(f"⚡ EXECUTION PATH: only_structured", flush=True)
        print(f"{'─'*100}", flush=True)

        try:
            print(
                f"[ACTION-STRING:{episode_id}] Calling structured LLM with {len(admissible_actions)} actions...",
                flush=True,
            )

            reasoning, action = self._call_llm_structured(
                sclient=sclient,
                limiter=limiter,
                requests_sem=requests_sem,
                model=model,
                user_message=user_message,
                filtered_actions=admissible_actions,
                debug_info=debug_info,
                ultimate_fallback_action=ultimate_fallback_action,
                system_message=system_message,
                env_handler=env_handler,
                episode_id=episode_id,
            )

            # Update llm_call structure with structured call info
            debug_info["llm_call"]["reasoning"] = reasoning
            debug_info["llm_call"]["response_metadata"] = {
                "total_admissible_actions": len(admissible_actions),
                "parsing_method": "structured",
                "direct_structured_used": True,
            }
            debug_info[
                "reasoning"
            ] = reasoning  # Also keep at top level for backwards compatibility
            print(f"[ACTION-STRING:{episode_id}] ✅ Structured Call SUCCESS", flush=True)
            print(
                f"[ACTION-STRING:{episode_id}] Selected Action: '{action}'", flush=True
            )
            print(f"{'─'*100}\n", flush=True)
            return action, debug_info

        except Exception as error:
            print(
                f"[ACTION-STRING:{episode_id}] ❌ Structured Call FAILED: {type(error).__name__}",
                flush=True,
            )
            print(f"[ACTION-STRING:{episode_id}] Error: {str(error)}", flush=True)
            print(f"\n{'╳'*100}", flush=True)
            print(f"🔄 Ultimate Fallback", flush=True)
            print(f"{'╳'*100}", flush=True)
            reasoning, action = self.get_ultimate_fallback_action(
                admissible_actions, ultimate_fallback_action=ultimate_fallback_action
            )
            # Update llm_call structure with error and fallback info
            debug_info["llm_call"]["errors"].append(
                {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "stage": "structured_call",
                }
            )
            debug_info["llm_call"]["reasoning"] = reasoning
            debug_info["llm_call"]["ultimate_fallback_used"] = True
            debug_info["llm_call"][
                "ultimate_fallback_reason"
            ] = f"Structured call failed: {str(error)}"
            debug_info[
                "reasoning"
            ] = reasoning  # Also keep at top level for backwards compatibility
            print(
                f"[ACTION-STRING:{episode_id}] Ultimate Fallback Action: '{action}'",
                flush=True,
            )
            print(f"{'╳'*100}\n", flush=True)
            return action, debug_info

    def _execute_normal_then_structured(
        self,
        sclient,
        limiter,
        requests_sem,
        model: str,
        user_message: str,
        admissible_actions: List[str],
        user_payload: Dict,
        debug_info: Dict,
        goal_text: str,
        recent_history_str: str,
        observation: str,
        inventory: str,
        trajectory_context: str,
        ultimate_fallback_action: str,
        system_message: str,
        env_handler,
        episode_id: str,
    ) -> Tuple[str, Dict]:
        """Execute normal_then_structured path: Normal → Structured with reranking → Ultimate fallback."""
        # Phase 1: Initial LLM call with all actions (retry up to 3 times for ANY error)
        print(f"\n{'━'*100}", flush=True)
        print(f"📝 PHASE 1: Normal LLM Call", flush=True)
        print(f"{'━'*100}", flush=True)
        try:
            print(
                f"[ACTION-STRING:{episode_id}] Calling LLM with all {len(admissible_actions)} actions (JSON format)...",
                flush=True,
            )

            action, additional_info = self._call_llm_normal(
                sclient=sclient,
                limiter=limiter,
                requests_sem=requests_sem,
                model=model,
                user_message=user_message,
                admissible_actions=admissible_actions,
                user_payload=user_payload,
                system_message=system_message,
                env_handler=env_handler,
                episode_id=episode_id,
            )

            # Extract reasoning and add to llm_call structure
            reasoning = additional_info.get("reasoning", "")
            debug_info["llm_call"]["reasoning"] = reasoning
            debug_info["llm_call"]["response_metadata"] = {
                "total_admissible_actions": additional_info.get(
                    "total_admissible_actions", len(admissible_actions)
                ),
                "parsing_method": additional_info.get("parsing_method", "normal"),
                "phase": "phase_1_normal",
            }
            debug_info[
                "reasoning"
            ] = reasoning  # Also keep at top level for backwards compatibility
            print(f"[ACTION-STRING:{episode_id}] ✅ PHASE 1 SUCCESS", flush=True)
            print(
                f"[ACTION-STRING:{episode_id}] Selected Action: '{action}'", flush=True
            )
            print(f"{'━'*100}\n", flush=True)
            return action, debug_info

        except self.ActionParsingError as e:
            # Phase 2: Handle ALL parsing/validation errors after retries exhausted (both invalid_action and json_parse)
            print(
                f"[ACTION-STRING:{episode_id}] ❌ PHASE 1 FAILED: {type(e).__name__} (error_type={e.error_type})",
                flush=True,
            )
            print(f"[ACTION-STRING:{episode_id}] Error details: {str(e)}", flush=True)

            # Check if we have reasoning to use for reranking
            original_reasoning = e.reasoning or ""
            has_reasoning = bool(original_reasoning.strip())

            if not has_reasoning:
                # No reasoning available - skip hybrid reranking and use structured call with ALL actions
                print(f"\n{'━'*100}", flush=True)
                print(
                    f"🔍 PHASE 2: Structured Call with ALL Actions (No Reasoning for Reranking)",
                    flush=True,
                )
                print(f"{'━'*100}", flush=True)
                print(
                    f"[ACTION-STRING:{episode_id}] ⚠️  No reasoning available from Phase 1 - skipping hybrid reranking",
                    flush=True,
                )
                print(
                    f"[ACTION-STRING:{episode_id}] Using ALL {len(admissible_actions)} actions for structured call",
                    flush=True,
                )

                # Use all actions (no filtering)
                filtered_actions = admissible_actions
            else:
                # We have reasoning - use hybrid reranking to filter actions
                print(f"\n{'━'*100}", flush=True)
                print(f"🔍 PHASE 2: Hybrid Reranking + Structured Call", flush=True)
                print(f"{'━'*100}", flush=True)

                print(
                    f"[ACTION-STRING:{episode_id}] Extracting key concepts from LLM reasoning...",
                    flush=True,
                )

                # Extract key concepts from reasoning for BM25
                key_concepts = []
                words = original_reasoning.lower().split()
                common_words = {
                    "the",
                    "a",
                    "an",
                    "and",
                    "or",
                    "but",
                    "is",
                    "are",
                    "was",
                    "were",
                    "to",
                    "from",
                    "in",
                    "on",
                    "at",
                    "for",
                    "with",
                    "by",
                }
                key_concepts = [
                    w for w in words if len(w) > 3 and w not in common_words
                ][:10]
                print(
                    f"[ACTION-STRING:{episode_id}] Key Concepts: {key_concepts[:5]}{'...' if len(key_concepts) > 5 else ''}",
                    flush=True,
                )

                print(
                    f"[ACTION-STRING:{episode_id}] Running hybrid reranking (BM25 + Qwen3-Reranker)...",
                    flush=True,
                )
                filtered_actions = self.rerank_actions_hybrid(
                    admissible_actions=admissible_actions,
                    reasoning=original_reasoning,
                    key_concepts=key_concepts,
                    goal_text=goal_text,
                    recent_history_str=recent_history_str,
                    observation=observation,
                    inventory=inventory,
                )

            # Try to ensure ultimate_fallback_action is in filtered actions (best effort)
            if ultimate_fallback_action not in filtered_actions:
                if ultimate_fallback_action in admissible_actions:
                    # Add it to the end of filtered actions if it's in the original list
                    filtered_actions.append(ultimate_fallback_action)
                    print(
                        f"[ACTION-STRING:{episode_id}] ⚠️  Added ultimate_fallback_action '{ultimate_fallback_action}' to filtered actions",
                        flush=True,
                    )
                else:
                    # Ultimate fallback not in admissible actions - log warning but continue
                    print(
                        f"[ACTION-STRING:{episode_id}] ⚠️  WARNING: Ultimate fallback '{ultimate_fallback_action}' not in admissible actions",
                        flush=True,
                    )
                    print(
                        f"[ACTION-STRING:{episode_id}] Will use configured fallback anyway if structured call fails",
                        flush=True,
                    )

            print(
                f"[ACTION-STRING:{episode_id}] ✓ Reranking Complete: {len(admissible_actions)} → {len(filtered_actions)} actions",
                flush=True,
            )

            # Build structured fallback payload
            structured_payload_parts = []
            structured_payload_parts.append(f"GOAL: {goal_text}")

            if recent_history_str and recent_history_str.strip():
                structured_payload_parts.append(
                    f"\nRECENT HISTORY:\nThese are the observation, action, and reward information from your previous steps in this episode\n{recent_history_str}"
                )

            if trajectory_context and trajectory_context.strip():
                structured_payload_parts.append(
                    f"\nRETRIEVED TRAJECTORY GUIDANCE:\nThis is a retrieved trajectory from a successful past episode that may help guide your decision. Use it as reference.\n{trajectory_context}"
                )

            structured_payload_parts.append(f"\nCURRENT OBSERVATION: {observation}")

            # Only add inventory if it exists and is non-empty
            if inventory and inventory.strip():
                structured_payload_parts.append(f"\nINVENTORY: {inventory}")

            structured_payload_parts.append(
                f"\nFILTERED ADMISSIBLE ACTIONS ({len(filtered_actions)} total):"
            )
            for i, action in enumerate(filtered_actions, 1):
                structured_payload_parts.append(f"  {i}. {action}")

            structured_message = "\n".join(structured_payload_parts)

            # Log structured fallback payload
            print(f"\n{'='*80}")
            print(f"[LLM-REQUEST] STRUCTURED FALLBACK MESSAGE TO {model}")
            print(f"{'='*80}")
            char_count = len(structured_message)
            token_estimate = char_count // 4
            preview = structured_message[:300].replace("\n", "\\n")
            if char_count > 300:
                preview += "..."
            print(f"Message preview: {preview}")
            print(f"Message stats: {char_count} chars, ~{token_estimate} tokens")
            print(f"Filtered actions: {len(filtered_actions)}")
            print(f"{'='*80}\n")

            print(
                f"[ACTION-STRING:{episode_id}] Calling structured API with {len(filtered_actions)} filtered actions...",
                flush=True,
            )
            try:
                reasoning, action = self._call_llm_structured(
                    sclient=sclient,
                    limiter=limiter,
                    requests_sem=requests_sem,
                    model=model,
                    user_message=structured_message,
                    filtered_actions=filtered_actions,
                    debug_info=debug_info,
                    ultimate_fallback_action=ultimate_fallback_action,
                    system_message=system_message,
                    env_handler=env_handler,
                    episode_id=episode_id,
                )

                # Update llm_call structure with phase 2 info
                debug_info["llm_call"]["reasoning"] = reasoning
                debug_info["llm_call"]["response_metadata"] = {
                    "total_admissible_actions": len(admissible_actions),
                    "filtered_actions_count": len(filtered_actions),
                    "parsing_method": "structured",
                    "phase": "phase_2_structured",
                    "hybrid_reranking_fallback_used": True,
                    "structured_fallback_used": True,
                    "had_reasoning_from_phase1": has_reasoning,
                }
                # Also store phase 1 error info
                debug_info["llm_call"]["phase_1_error"] = {
                    "error_type": type(e).__name__,
                    "error_subtype": e.error_type,
                    "error_message": str(e),
                }
                debug_info[
                    "reasoning"
                ] = reasoning  # Also keep at top level for backwards compatibility

                print(f"[ACTION-STRING:{episode_id}] ✅ PHASE 2 SUCCESS", flush=True)
                print(
                    f"[ACTION-STRING:{episode_id}] Selected Action: '{action}'",
                    flush=True,
                )
                print(f"{'━'*100}\n", flush=True)
                return action, debug_info

            except Exception as structured_error:
                print(
                    f"[ACTION-STRING:{episode_id}] ❌ PHASE 2 FAILED: {type(structured_error).__name__}",
                    flush=True,
                )
                print(
                    f"[ACTION-STRING:{episode_id}] Error details: {str(structured_error)}",
                    flush=True,
                )
                print(f"\n{'╳'*100}", flush=True)
                print(f"🔄 PHASE 3: Ultimate Fallback", flush=True)
                print(f"{'╳'*100}", flush=True)
                reasoning, action = self.get_ultimate_fallback_action(
                    admissible_actions,
                    ultimate_fallback_action=ultimate_fallback_action,
                )
                # Update llm_call structure with all phase errors
                debug_info["llm_call"]["errors"].append(
                    {
                        "error_type": type(e).__name__,
                        "error_subtype": e.error_type,
                        "error_message": str(e),
                        "stage": "phase_1_normal",
                    }
                )
                debug_info["llm_call"]["errors"].append(
                    {
                        "error_type": type(structured_error).__name__,
                        "error_message": str(structured_error),
                        "stage": "phase_2_structured",
                    }
                )
                debug_info["llm_call"]["reasoning"] = reasoning
                debug_info["llm_call"]["ultimate_fallback_used"] = True
                debug_info["llm_call"][
                    "ultimate_fallback_reason"
                ] = f"Structured fallback failed: {str(structured_error)}"
                debug_info["llm_call"]["response_metadata"] = {
                    "phase": "phase_3_ultimate_fallback",
                    "filtered_actions_count": len(filtered_actions)
                    if "filtered_actions" in locals()
                    else 0,
                }
                debug_info[
                    "reasoning"
                ] = reasoning  # Also keep at top level for backwards compatibility
                print(
                    f"[ACTION-STRING:{episode_id}] Ultimate Fallback Action: '{action}'",
                    flush=True,
                )
                print(f"{'╳'*100}\n", flush=True)
                return action, debug_info

        except Exception as api_error:
            # Phase 3: Handle API errors after retries exhausted
            print(
                f"[ACTION-STRING:{episode_id}] ❌ PHASE 1 FAILED: API error - {type(api_error).__name__}",
                flush=True,
            )
            print(
                f"[ACTION-STRING:{episode_id}] Error details: {str(api_error)}",
                flush=True,
            )
            print(f"\n{'╳'*100}", flush=True)
            print(f"🔄 PHASE 3: Ultimate Fallback", flush=True)
            print(f"{'╳'*100}", flush=True)
            reasoning, action = self.get_ultimate_fallback_action(
                admissible_actions, ultimate_fallback_action=ultimate_fallback_action
            )
            # Update llm_call structure with API error
            debug_info["llm_call"]["errors"].append(
                {
                    "error_type": type(api_error).__name__,
                    "error_message": str(api_error),
                    "stage": "phase_1_api_error",
                }
            )
            debug_info["llm_call"]["reasoning"] = reasoning
            debug_info["llm_call"]["ultimate_fallback_used"] = True
            debug_info["llm_call"][
                "ultimate_fallback_reason"
            ] = f"API error: {str(api_error)}"
            debug_info["llm_call"]["response_metadata"] = {
                "phase": "phase_3_ultimate_fallback"
            }
            debug_info[
                "reasoning"
            ] = reasoning  # Also keep at top level for backwards compatibility
            print(
                f"[ACTION-STRING:{episode_id}] Ultimate Fallback Action: '{action}'",
                flush=True,
            )
            print(f"{'╳'*100}\n", flush=True)
            return action, debug_info
