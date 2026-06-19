# traj_retrieval/strategies/base_strategy.py
# Base strategy interface and shared components (no async)

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional
from collections import OrderedDict


class BaseExperimentStrategy(ABC):
    """
    Base class for different LLM experiment strategies.
    Each strategy handles prompt building, response parsing, and fallback logic.

    This is the abstract interface without async/sync specifics.
    Subclasses should extend this with either async or sync execution methods.
    """

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the name of this strategy."""
        pass

    @abstractmethod
    def build_user_payload(
        self,
        goal_text: str,
        recent_history_str: str,
        observation: str,
        inventory: str,
        admissible_actions: List[str],
        **kwargs,
    ) -> Dict[str, Any]:
        """Build the user payload for this experiment type."""
        pass

    @abstractmethod
    def build_schema_message(self) -> str:
        """Build the schema instruction message for this experiment type."""
        pass

    @abstractmethod
    def parse_response(
        self, response_text: str, admissible_actions: List[str], **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Parse the LLM response and return (action, additional_info_dict).

        Returns:
            Tuple of (action, additional_info_dict)
        """
        pass

    @abstractmethod
    def get_fallback_action(
        self,
        admissible_actions: List[str],
        ultimate_fallback_action: str = "look around",
        **kwargs,
    ) -> Tuple[str, str]:
        """
        Get fallback action when parsing fails.

        Args:
            admissible_actions: List of admissible actions
            ultimate_fallback_action: Action to use as ultimate fallback (from config)

        Returns:
            Tuple of (reasoning, action)
        """
        pass

    def get_debug_info_keys(self) -> List[str]:
        """Return additional debug info keys specific to this strategy."""
        return []


class HybridRerankingMixin:
    """
    Mixin class providing common BM25 + Qwen3-Reranker hybrid reranking functionality.
    Used by strategies that need intelligent action filtering.

    This mixin is SYNCHRONOUS and does not use asyncio.
    It can be used by both async and sync strategy implementations.
    """

    # Class-level cache to prevent model reloading across instances
    _global_qwen3_reranker = None
    _global_qwen3_tokenizer = None
    _global_device = None
    _global_use_gpu = False
    _global_token_false_id = None
    _global_token_true_id = None
    _global_prefix_tokens = None
    _global_suffix_tokens = None

    def __init__(self):
        """Initialize the reranking components."""
        self.qwen3_model_name = "Qwen/Qwen3-Reranker-0.6B"
        self.max_length = 512

    def _detect_device(self):
        """Detect the best available device for Qwen3-Reranker."""
        import torch

        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[HYBRID-RERANK] GPU detected: {gpu_name}", flush=True)
            print(
                f"[HYBRID-RERANK] Using GPU acceleration for Qwen3-Reranker with quantization",
                flush=True,
            )
            return device, True
        else:
            device = torch.device("cpu")
            print(
                f"[HYBRID-RERANK] No GPU available, using CPU for Qwen3-Reranker",
                flush=True,
            )
            return device, False

    def _get_qwen3_reranker(self):
        """Lazy initialization of Qwen3-Reranker model with quantization using class-level cache."""
        # Lazy imports - only load when actually using reranker
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        # Use class-level cache to prevent reloading across instances
        if HybridRerankingMixin._global_qwen3_reranker is None:
            # Detect device if not already done
            if HybridRerankingMixin._global_device is None:
                (
                    HybridRerankingMixin._global_device,
                    HybridRerankingMixin._global_use_gpu,
                ) = self._detect_device()

            print(
                f"[HYBRID-RERANK] Loading Qwen3-Reranker model: {self.qwen3_model_name} (first time only)",
                flush=True,
            )

            # Initialize tokenizer with fallback to slow tokenizer if fast tokenizer fails
            try:
                print(
                    f"[HYBRID-RERANK] Attempting to load fast tokenizer...", flush=True
                )
                HybridRerankingMixin._global_qwen3_tokenizer = (
                    AutoTokenizer.from_pretrained(
                        self.qwen3_model_name, padding_side="left", use_fast=True
                    )
                )
                print(f"[HYBRID-RERANK] Fast tokenizer loaded successfully", flush=True)
            except Exception as fast_tokenizer_error:
                print(
                    f"[HYBRID-RERANK] Fast tokenizer failed: {fast_tokenizer_error}",
                    flush=True,
                )
                print(f"[HYBRID-RERANK] Falling back to slow tokenizer...", flush=True)
                try:
                    HybridRerankingMixin._global_qwen3_tokenizer = (
                        AutoTokenizer.from_pretrained(
                            self.qwen3_model_name, padding_side="left", use_fast=False
                        )
                    )
                    print(
                        f"[HYBRID-RERANK] Slow tokenizer loaded successfully",
                        flush=True,
                    )
                except Exception as slow_tokenizer_error:
                    print(
                        f"[HYBRID-RERANK] Slow tokenizer also failed: {slow_tokenizer_error}",
                        flush=True,
                    )
                    raise Exception(
                        f"Failed to load tokenizer for {self.qwen3_model_name}. Please try: pip install --upgrade transformers tokenizers"
                    ) from slow_tokenizer_error

            # Initialize model with quantization if GPU is available
            if HybridRerankingMixin._global_use_gpu:
                print(
                    f"[HYBRID-RERANK] Loading Qwen3-Reranker with 4-bit quantization on GPU",
                    flush=True,
                )
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )

                try:
                    HybridRerankingMixin._global_qwen3_reranker = (
                        AutoModelForCausalLM.from_pretrained(
                            self.qwen3_model_name,
                            quantization_config=bnb_config,
                            device_map="auto",
                            attn_implementation="flash_attention_2",
                            torch_dtype=torch.float16,
                            low_cpu_mem_usage=True,
                            use_cache=False,
                        )
                    )
                    print(
                        f"[HYBRID-RERANK] Model loaded successfully with 4-bit quantization and Flash Attention 2",
                        flush=True,
                    )
                except Exception as fa2_error:
                    print(
                        f"[HYBRID-RERANK] Failed to load with Flash Attention 2: {fa2_error}",
                        flush=True,
                    )
                    print(
                        f"[HYBRID-RERANK] Falling back to standard attention (slower but works)...",
                        flush=True,
                    )
                    HybridRerankingMixin._global_qwen3_reranker = (
                        AutoModelForCausalLM.from_pretrained(
                            self.qwen3_model_name,
                            quantization_config=bnb_config,
                            device_map="auto",
                            torch_dtype=torch.float16,
                            low_cpu_mem_usage=True,
                            use_cache=False,
                        )
                    )
                    print(
                        f"[HYBRID-RERANK] Model loaded successfully with 4-bit quantization (standard attention)",
                        flush=True,
                    )
            else:
                # CPU mode - no quantization
                print(
                    f"[HYBRID-RERANK] Loading Qwen3-Reranker on CPU (no quantization)",
                    flush=True,
                )
                HybridRerankingMixin._global_qwen3_reranker = (
                    AutoModelForCausalLM.from_pretrained(
                        self.qwen3_model_name,
                        device_map="cpu",
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True,
                        use_cache=False,
                    )
                )
                print(f"[HYBRID-RERANK] Model loaded successfully on CPU", flush=True)

            # Pre-compute token IDs for Yes/No
            yes_text = "Yes"
            no_text = "No"

            # Tokenize and get IDs
            yes_token_ids = HybridRerankingMixin._global_qwen3_tokenizer.encode(
                yes_text, add_special_tokens=False
            )
            no_token_ids = HybridRerankingMixin._global_qwen3_tokenizer.encode(
                no_text, add_special_tokens=False
            )

            # Store the first token ID for each (most models use single token for Yes/No)
            HybridRerankingMixin._global_token_true_id = (
                yes_token_ids[0] if yes_token_ids else None
            )
            HybridRerankingMixin._global_token_false_id = (
                no_token_ids[0] if no_token_ids else None
            )

            # Pre-compute prefix/suffix tokens
            prefix = "Query: "
            suffix = " Document: "
            HybridRerankingMixin._global_prefix_tokens = (
                HybridRerankingMixin._global_qwen3_tokenizer.encode(
                    prefix, add_special_tokens=False
                )
            )
            HybridRerankingMixin._global_suffix_tokens = (
                HybridRerankingMixin._global_qwen3_tokenizer.encode(
                    suffix, add_special_tokens=False
                )
            )

            print(
                f"[HYBRID-RERANK] ✅ Qwen3-Reranker initialization complete", flush=True
            )
            print(
                f"[HYBRID-RERANK] Token IDs - Yes: {HybridRerankingMixin._global_token_true_id}, No: {HybridRerankingMixin._global_token_false_id}",
                flush=True,
            )

        return (
            HybridRerankingMixin._global_qwen3_reranker,
            HybridRerankingMixin._global_qwen3_tokenizer,
        )

    def rerank_actions_with_bm25(
        self,
        admissible_actions: List[str],
        reasoning: str,
        key_concepts: List[str] = None,
        top_k: int = 100,
    ) -> List[str]:
        """
        Rerank actions using BM25 based on reasoning and key concepts.
        Raises exception if reranking fails - caller should handle fallback.
        """
        # Lazy imports
        import re
        from rank_bm25 import BM25Okapi

        if not admissible_actions:
            return []

        # If no reranking needed (already small enough), return all actions
        if len(admissible_actions) <= top_k:
            return admissible_actions

        # Build query from reasoning and key concepts
        query_parts = []
        if reasoning:
            query_parts.append(reasoning.lower())
        if key_concepts:
            query_parts.extend([concept.lower() for concept in key_concepts])

        if not query_parts:
            # No query available, return random selection
            import random

            return random.sample(
                admissible_actions, min(top_k, len(admissible_actions))
            )

        query = " ".join(query_parts)
        query_tokens = re.findall(r"\w+", query)

        # Tokenize all actions
        action_tokens = [
            re.findall(r"\w+", action.lower()) for action in admissible_actions
        ]

        # Build BM25 index
        bm25 = BM25Okapi(action_tokens)

        # Get scores
        scores = bm25.get_scores(query_tokens)

        # Sort by score and return top_k
        action_score_pairs = list(zip(admissible_actions, scores))
        action_score_pairs.sort(key=lambda x: x[1], reverse=True)
        reranked_actions = [action for action, score in action_score_pairs[:top_k]]

        return reranked_actions

    def build_qwen3_reranker_query(
        self,
        goal_text: str,
        recent_history_str: str,
        observation: str,
        inventory: str,
        reasoning: str,
    ) -> str:
        """
        Build comprehensive query for Qwen3-Reranker with proper context.
        """
        query_parts = []

        if goal_text.strip():
            query_parts.append(f"Goal: {goal_text.strip()}")

        if observation.strip():
            query_parts.append(f"Last observation: {observation.strip()}")

        # Extract last 2 actions from recent_history_str
        if recent_history_str.strip():
            history_lines = recent_history_str.strip().split("\n")
            actions = []
            for line in history_lines:
                if "| ACTION: " in line and "| REWARD: " in line:
                    action_start = line.find("| ACTION: ") + len("| ACTION: ")
                    action_end = line.find(" | REWARD: ")
                    if action_start > 0 and action_end > action_start:
                        action = line[action_start:action_end].strip()
                        if action:
                            actions.append(action)

            if actions:
                recent_actions = actions[-2:] if len(actions) > 2 else actions
                if recent_actions:
                    query_parts.append(
                        f"Recent action history: {' | '.join(recent_actions)}"
                    )

        if inventory.strip():
            query_parts.append(f"Inventory: {inventory.strip()}")

        if reasoning.strip():
            query_parts.append(f"Analysis: {reasoning.strip()}")

        return " | ".join(query_parts)

    def rerank_actions_with_qwen3_reranker_full(
        self,
        admissible_actions: List[str],
        goal_text: str,
        recent_history_str: str,
        observation: str,
        inventory: str,
        reasoning: str,
        top_k: int = 400,
    ) -> List[str]:
        """
        Rerank all admissible actions using Qwen3-Reranker.
        Raises exception if reranking fails - caller should handle fallback.
        """
        # Lazy imports
        import time
        import torch

        if not admissible_actions:
            return []

        if len(admissible_actions) <= top_k:
            return admissible_actions

        # This method should only be called when GPU is available
        # Caller is responsible for checking GPU availability
        try:
            # Initialize debug flag for first batch
            self._debug_first_batch = False

            # Build comprehensive query
            query = self.build_qwen3_reranker_query(
                goal_text=goal_text,
                recent_history_str=recent_history_str,
                observation=observation,
                inventory=inventory,
                reasoning=reasoning,
            )

            # Get Qwen3-Reranker model
            model, tokenizer = self._get_qwen3_reranker()

            # GPU memory optimization
            if HybridRerankingMixin._global_use_gpu:
                torch.cuda.empty_cache()
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                print(
                    f"[HYBRID-RERANK] GPU memory before reranking: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved",
                    flush=True,
                )

            # Batch processing
            batch_size = 32
            all_scores = []
            total_batches = (len(admissible_actions) + batch_size - 1) // batch_size

            start_time = time.time()

            for batch_idx in range(total_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(admissible_actions))
                batch_actions = admissible_actions[batch_start:batch_end]

                # Prepare pairs for this batch
                pairs = [[query, action] for action in batch_actions]

                # Build inputs manually using pre-computed tokens
                input_ids_list = []
                attention_mask_list = []

                for query_text, doc_text in pairs:
                    # Tokenize query and document
                    query_tokens = tokenizer.encode(
                        query_text, add_special_tokens=False
                    )
                    doc_tokens = tokenizer.encode(doc_text, add_special_tokens=False)

                    # Combine: prefix + query + suffix + document
                    input_ids = (
                        HybridRerankingMixin._global_prefix_tokens
                        + query_tokens
                        + HybridRerankingMixin._global_suffix_tokens
                        + doc_tokens
                    )

                    # Truncate to max_length
                    if len(input_ids) > self.max_length:
                        input_ids = input_ids[: self.max_length]

                    # Create attention mask
                    attention_mask = [1] * len(input_ids)

                    input_ids_list.append(input_ids)
                    attention_mask_list.append(attention_mask)

                # Pad sequences to the same length within batch
                max_len_in_batch = max(len(ids) for ids in input_ids_list)

                padded_input_ids = []
                padded_attention_mask = []

                for input_ids, attention_mask in zip(
                    input_ids_list, attention_mask_list
                ):
                    padding_length = max_len_in_batch - len(input_ids)

                    # Pad with tokenizer.pad_token_id (usually 0)
                    pad_token_id = (
                        tokenizer.pad_token_id
                        if tokenizer.pad_token_id is not None
                        else 0
                    )
                    padded_input_ids.append(input_ids + [pad_token_id] * padding_length)
                    padded_attention_mask.append(attention_mask + [0] * padding_length)

                # Convert to tensors
                input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
                attention_mask_tensor = torch.tensor(
                    padded_attention_mask, dtype=torch.long
                )

                # Move to device
                device = HybridRerankingMixin._global_device
                input_ids_tensor = input_ids_tensor.to(device)
                attention_mask_tensor = attention_mask_tensor.to(device)

                # Get model predictions
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids_tensor, attention_mask=attention_mask_tensor
                    )
                    logits = outputs.logits[:, -1, :]  # Last token logits

                    # Extract scores for Yes/No tokens
                    yes_logits = logits[:, HybridRerankingMixin._global_token_true_id]
                    no_logits = logits[:, HybridRerankingMixin._global_token_false_id]

                    # Compute scores using softmax over Yes/No
                    scores_batch = torch.softmax(
                        torch.stack([no_logits, yes_logits], dim=-1), dim=-1
                    )[:, 1]

                    # Move to CPU and convert to list
                    batch_scores = scores_batch.cpu().tolist()
                    all_scores.extend(batch_scores)

                # Print progress every 10 batches or on first/last batch
                if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                    elapsed = time.time() - start_time
                    processed = batch_end
                    rate = processed / elapsed if elapsed > 0 else 0
                    eta = (
                        (len(admissible_actions) - processed) / rate if rate > 0 else 0
                    )
                    print(
                        f"[HYBRID-RERANK] Batch {batch_idx + 1}/{total_batches}: Processed {processed}/{len(admissible_actions)} actions ({rate:.1f} actions/s, ETA: {eta:.1f}s)",
                        flush=True,
                    )

            total_time = time.time() - start_time

            # GPU memory after reranking
            if HybridRerankingMixin._global_use_gpu:
                torch.cuda.empty_cache()
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                print(
                    f"[HYBRID-RERANK] GPU memory after reranking: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved",
                    flush=True,
                )

            print(
                f"[HYBRID-RERANK] Qwen3-Reranker complete: {len(admissible_actions)} actions in {total_time:.2f}s ({len(admissible_actions)/total_time:.1f} actions/s)",
                flush=True,
            )

            # Sort by score and return top_k
            action_score_pairs = list(zip(admissible_actions, all_scores))
            action_score_pairs.sort(key=lambda x: x[1], reverse=True)
            reranked_actions = [action for action, score in action_score_pairs[:top_k]]

            return reranked_actions

        except Exception as e:
            print(
                f"[HYBRID-RERANK] ❌ Qwen3-Reranker failed: {type(e).__name__} - {str(e)}",
                flush=True,
            )
            raise

    def rerank_actions_hybrid(
        self,
        admissible_actions: List[str],
        reasoning: str,
        key_concepts: List[str],
        goal_text: str,
        recent_history_str: str,
        observation: str,
        inventory: str,
        bm25_top_k: int = 100,
        qwen3_reranker_top_k: int = 400,
    ) -> List[str]:
        """
        Hybrid reranking that combines BM25 and Qwen3-Reranker (GPU only) results and deduplicates.

        Strategy:
        - Always try BM25 reranking
        - Only use Qwen3-Reranker if GPU (CUDA) is available
        - Fallback: If BM25 fails → random selection
        - Fallback: If Qwen3-Reranker fails → random selection
        """
        # Lazy imports
        import random
        import torch

        if not admissible_actions:
            return []

        # Check GPU availability upfront
        has_gpu = torch.cuda.is_available()

        print(f"\n{'┄'*100}", flush=True)
        if has_gpu:
            print(
                f"🔀 HYBRID RERANKING: BM25 + Qwen3-Reranker (GPU Available)", flush=True
            )
        else:
            print(
                f"🔀 HYBRID RERANKING: BM25 Only (No GPU - Qwen3-Reranker Skipped)",
                flush=True,
            )
        print(f"{'┄'*100}", flush=True)
        print(f"[HYBRID-RERANK] Input: {len(admissible_actions)} actions", flush=True)

        if has_gpu:
            print(
                f"[HYBRID-RERANK] Target: BM25 top {bm25_top_k} + Qwen3-Reranker top {qwen3_reranker_top_k}",
                flush=True,
            )
        else:
            print(
                f"[HYBRID-RERANK] Target: BM25 top {bm25_top_k} (Qwen3-Reranker skipped on CPU)",
                flush=True,
            )

        # Step 1: Get top actions from BM25
        bm25_actions = []
        try:
            print(f"[HYBRID-RERANK] Step 1: Running BM25 reranking...", flush=True)
            bm25_actions = self.rerank_actions_with_bm25(
                admissible_actions=admissible_actions,
                reasoning=reasoning,
                key_concepts=key_concepts,
                top_k=bm25_top_k,
            )
            print(
                f"[HYBRID-RERANK] ✓ BM25 complete: {len(bm25_actions)} actions",
                flush=True,
            )
        except Exception as bm25_error:
            print(
                f"[HYBRID-RERANK] ❌ BM25 failed: {type(bm25_error).__name__} - {str(bm25_error)[:100]}",
                flush=True,
            )
            # Fallback: Random selection
            num_to_select = min(bm25_top_k, len(admissible_actions))
            bm25_actions = random.sample(admissible_actions, num_to_select)
            print(
                f"[HYBRID-RERANK] Using BM25 fallback: random {len(bm25_actions)} actions",
                flush=True,
            )

        # Step 2: Get top actions from Qwen3-Reranker (only if GPU available)
        qwen3_reranker_actions = []
        if has_gpu:
            try:
                print(f"[HYBRID-RERANK] Step 2: Running Qwen3-Reranker...", flush=True)
                qwen3_reranker_actions = self.rerank_actions_with_qwen3_reranker_full(
                    admissible_actions=admissible_actions,
                    goal_text=goal_text,
                    recent_history_str=recent_history_str,
                    observation=observation,
                    inventory=inventory,
                    reasoning=reasoning,
                    top_k=qwen3_reranker_top_k,
                )
                print(
                    f"[HYBRID-RERANK] ✓ Qwen3-Reranker complete: {len(qwen3_reranker_actions)} actions",
                    flush=True,
                )
            except Exception as qwen3_error:
                print(
                    f"[HYBRID-RERANK] ❌ Qwen3-Reranker failed: {type(qwen3_error).__name__} - {str(qwen3_error)[:100]}",
                    flush=True,
                )
                # Fallback: Random selection
                num_to_select = min(qwen3_reranker_top_k, len(admissible_actions))
                qwen3_reranker_actions = random.sample(
                    admissible_actions, num_to_select
                )
                print(
                    f"[HYBRID-RERANK] Using Qwen3-Reranker fallback: random {len(qwen3_reranker_actions)} actions",
                    flush=True,
                )
        else:
            print(f"[HYBRID-RERANK] Skipping Qwen3-Reranker (CPU mode)", flush=True)

        # Combine and deduplicate results
        print(f"[HYBRID-RERANK] Combining and deduplicating results...", flush=True)
        combined_actions = OrderedDict()

        # Add BM25 actions first (priority in ordering)
        for action in bm25_actions:
            combined_actions[action] = True

        # Add Qwen3-Reranker actions (only new ones, only if GPU was used)
        if has_gpu:
            for action in qwen3_reranker_actions:
                combined_actions[action] = True

        final_actions = list(combined_actions.keys())

        # Calculate statistics
        if has_gpu and qwen3_reranker_actions:
            overlap = (
                len(bm25_actions) + len(qwen3_reranker_actions) - len(final_actions)
            )
            print(
                f"[HYBRID-RERANK] ✅ Complete: {len(admissible_actions)} → {len(final_actions)} actions",
                flush=True,
            )
            print(
                f"[HYBRID-RERANK] BM25-Qwen3 overlap: {overlap} actions ({overlap * 100 // max(len(final_actions), 1)}%)",
                flush=True,
            )
        else:
            print(
                f"[HYBRID-RERANK] ✅ Complete: {len(admissible_actions)} → {len(final_actions)} actions (BM25 only)",
                flush=True,
            )

        print(f"{'┄'*100}\n", flush=True)

        return final_actions
