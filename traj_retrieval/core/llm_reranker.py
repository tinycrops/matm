# LLM-based reranker wrapper.
#
# This module orchestrates pairwise LLM reranking on top of the third-party
# `llm-rankers` library (https://github.com/ielab/llm-rankers,
# Pradeep et al.). The pairwise prompt template below is adapted from that
# project; `llm-rankers` is BSD-licensed (see PyPI metadata). This wrapper is
# Apache-2.0 like the rest of MATM.

from __future__ import annotations

import importlib
import inspect
import copy
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

Candidate = Dict[str, Any]

LLMRANKERS_PAIRWISE_PROMPT = """Given a query "{query}", which of the following two passages is more relevant to the query?

Passage A: "{doc1}"

Passage B: "{doc2}"

Output Passage A or Passage B:"""


class _SimpleSearchResult:
    def __init__(self, docid: str, score: float, text: str) -> None:
        self.docid = docid
        self.score = float(score)
        self.text = text


def _canonical_pair_key(
    query: str, doc1: str, doc2: str
) -> Tuple[Tuple[str, str, str], bool]:
    if doc1 <= doc2:
        return (query, doc1, doc2), False
    return (query, doc2, doc1), True


def _extract_pairwise_choice(text: str) -> str:
    raw = str(text or "").strip().upper()
    if not raw:
        return ""

    raw = re.sub(r"\s+", " ", raw)
    patterns = (
        (r"\bPASSAGE\s*A\b", "A"),
        (r"\bPASSAGE\s*B\b", "B"),
        (r"\bOPTION\s*A\b", "A"),
        (r"\bOPTION\s*B\b", "B"),
        (r"\bANSWER\s*[:\-]?\s*A\b", "A"),
        (r"\bANSWER\s*[:\-]?\s*B\b", "B"),
    )
    for pattern, choice in patterns:
        if re.search(pattern, raw):
            return choice
    if re.fullmatch(r"[AB]", raw):
        return raw
    if re.search(r"\bA\b", raw) and not re.search(r"\bB\b", raw):
        return "A"
    if re.search(r"\bB\b", raw) and not re.search(r"\bA\b", raw):
        return "B"
    return ""


class _ParallelPairwiseBase:
    _supports_parallel_pairwise = True

    def __init__(
        self,
        method: str,
        batch_size: int,
        k: int,
        prompt_template: Optional[str],
        max_pairs: Optional[int],
        pair_batch_prompts: int,
        pair_max_inflight: int,
        pair_retry: int,
        enable_cache: bool,
    ) -> None:
        self.method = ("allpair" if method == "allpairs" else method).lower()
        self.batch_size = int(batch_size)
        self.k = int(k)
        self.prompt = prompt_template or LLMRANKERS_PAIRWISE_PROMPT
        self.max_pairs = int(max_pairs) if max_pairs is not None else None
        self.pair_batch_prompts = max(2, int(pair_batch_prompts))
        self.pair_max_inflight = max(1, int(pair_max_inflight))
        self.pair_retry = max(0, int(pair_retry))
        self.enable_cache = bool(enable_cache)
        self._allowed_methods = {"allpair", "heapsort", "bubblesort"}
        if self.method not in self._allowed_methods:
            raise ValueError(
                f"Unsupported pairwise method='{self.method}'. "
                "Supported: allpair, heapsort, bubblesort."
            )

        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0

        self._pair_cache: Dict[Tuple[str, str, str], List[str]] = {}
        self._runtime_total: Dict[str, float] = defaultdict(float)
        self._runtime_last: Dict[str, float] = {}
        self._reset_run_stats()

    def _reset_run_stats(self) -> None:
        self._run_pairs_total = 0
        self._run_cache_hit = 0
        self._run_http_calls = 0
        self._run_prompt_batches = 0
        self._run_prompt_count = 0
        self._run_retry_count = 0
        self._run_compare_calls = 0
        self._run_elapsed = 0.0

    def _finalize_run_stats(self, elapsed_s: float) -> None:
        self._run_elapsed = float(elapsed_s)
        avg_batch = 0.0
        if self._run_prompt_batches > 0:
            avg_batch = float(self._run_prompt_count) / float(self._run_prompt_batches)

        run_stats = {
            "pairs_total": int(self._run_pairs_total),
            "cache_hit": int(self._run_cache_hit),
            "http_calls": int(self._run_http_calls),
            "prompt_batches": int(self._run_prompt_batches),
            "prompt_count": int(self._run_prompt_count),
            "avg_batch_size": float(avg_batch),
            "retry_count": int(self._run_retry_count),
            "compare_calls": int(self._run_compare_calls),
            "rerank_time_s": float(self._run_elapsed),
        }
        self._runtime_last = run_stats
        for key, value in run_stats.items():
            self._runtime_total[key] += float(value)

    def get_runtime_stats(self) -> Dict[str, Any]:
        total = dict(self._runtime_total)
        prompt_batches = float(total.get("prompt_batches", 0.0))
        prompt_count = float(total.get("prompt_count", 0.0))
        if prompt_batches > 0:
            total["avg_batch_size"] = prompt_count / prompt_batches
        else:
            total["avg_batch_size"] = 0.0
        total["pairs_total"] = int(total.get("pairs_total", 0.0))
        total["cache_hit"] = int(total.get("cache_hit", 0.0))
        total["http_calls"] = int(total.get("http_calls", 0.0))
        total["prompt_batches"] = int(total.get("prompt_batches", 0.0))
        total["prompt_count"] = int(total.get("prompt_count", 0.0))
        total["retry_count"] = int(total.get("retry_count", 0.0))
        total["compare_calls"] = int(total.get("compare_calls", 0.0))
        total["cache_hit_rate"] = (
            float(total["cache_hit"]) / float(total["pairs_total"])
            if total["pairs_total"] > 0
            else 0.0
        )
        return {
            "last": dict(self._runtime_last),
            "total": total,
        }

    def _build_generation_prompt(self, input_text: str) -> str:
        if getattr(self, "tokenizer", None) is None:
            return f"{input_text}\nPassage:"
        try:
            conversation = [{"role": "user", "content": input_text}]
            prompt = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt = input_text
        return f"{prompt} Passage:"

    def _normalize_pairwise_output(self, raw_text: str) -> str:
        choice = _extract_pairwise_choice(raw_text)
        if choice == "A":
            return "Passage A"
        if choice == "B":
            return "Passage B"
        text = str(raw_text or "").strip().upper()
        return f"Passage {text}" if text else "Passage"

    def _infer_pair_once(self, query: str, doc1: str, doc2: str) -> List[str]:
        prompts = [
            self.prompt.format(query=query, doc1=doc1, doc2=doc2),
            self.prompt.format(query=query, doc1=doc2, doc2=doc1),
        ]
        raw_outputs = self.infer_prompts(prompts)
        if len(raw_outputs) != 2:
            raise RuntimeError(
                f"infer_prompts returned {len(raw_outputs)} outputs for 2 prompts."
            )
        return [
            self._normalize_pairwise_output(raw_outputs[0]),
            self._normalize_pairwise_output(raw_outputs[1]),
        ]

    def compare(self, query: str, docs: Sequence[Any]) -> List[str]:
        self.total_compare += 1
        self._run_compare_calls += 1
        if len(docs) != 2:
            raise ValueError("compare expects exactly 2 documents.")

        doc1 = str(docs[0] or "")
        doc2 = str(docs[1] or "")
        self._run_pairs_total += 1
        cache_key, is_flipped = _canonical_pair_key(query, doc1, doc2)
        if self.enable_cache and cache_key in self._pair_cache:
            self._run_cache_hit += 1
            cached = self._pair_cache[cache_key]
            return [cached[1], cached[0]] if is_flipped else list(cached)

        outputs = self._infer_pair_once(query, doc1, doc2)
        if self.enable_cache:
            self._pair_cache[cache_key] = (
                [outputs[1], outputs[0]] if is_flipped else list(outputs)
            )
        return outputs

    def heapify(self, arr, n, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        if l < n and arr[l] > arr[i]:
            largest = l
        if r < n and arr[r] > arr[largest]:
            largest = r

        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            self.heapify(arr, n, largest)

    def heapSort(self, arr, k):
        n = len(arr)
        ranked = 0
        for i in range(n // 2, -1, -1):
            self.heapify(arr, n, i)
        for i in range(n - 1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]
            ranked += 1
            if ranked == k:
                break
            self.heapify(arr, i, 0)

    def _allpair_scores(self, query: str, ranking: List[Any]) -> Dict[str, float]:
        doc_pairs = list(combinations(ranking, 2))
        if self.max_pairs is not None and self.max_pairs >= 0:
            doc_pairs = doc_pairs[: self.max_pairs]
        self._run_pairs_total += len(doc_pairs)

        pair_prompts: List[str] = []
        for doc1, doc2 in doc_pairs:
            pair_prompts.append(
                self.prompt.format(query=query, doc1=doc1.text, doc2=doc2.text)
            )
            pair_prompts.append(
                self.prompt.format(query=query, doc1=doc2.text, doc2=doc1.text)
            )

        raw_outputs = self.infer_prompts(pair_prompts)
        if len(raw_outputs) != len(pair_prompts):
            raise RuntimeError(
                f"infer_prompts returned {len(raw_outputs)} outputs for {len(pair_prompts)} prompts."
            )

        scores = defaultdict(float)
        for idx in range(0, len(raw_outputs), 2):
            doc1, doc2 = doc_pairs[idx // 2]
            output1 = self._normalize_pairwise_output(raw_outputs[idx])
            output2 = self._normalize_pairwise_output(raw_outputs[idx + 1])
            if output1 == "Passage A" and output2 == "Passage B":
                scores[doc1.docid] += 1
            elif output1 == "Passage B" and output2 == "Passage A":
                scores[doc2.docid] += 1
            else:
                scores[doc1.docid] += 0.5
                scores[doc2.docid] += 0.5

            if self.enable_cache:
                cache_key, _ = _canonical_pair_key(
                    query, str(doc1.text), str(doc2.text)
                )
                self._pair_cache[cache_key] = [output1, output2]

        return scores

    def rerank(self, query: str, ranking: Sequence[Any]) -> List[Any]:
        ranking = self._normalize_ranking(ranking)
        original_ranking = copy.deepcopy(ranking)
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        self._pair_cache = {}
        self._reset_run_stats()

        start = time.perf_counter()
        if self.method == "allpair":
            scores = self._allpair_scores(query, ranking)
            ranking = sorted(
                [
                    self._make_result(docid=docid, score=score, text=None)
                    for docid, score in scores.items()
                ],
                key=lambda x: x.score,
                reverse=True,
            )
        elif self.method == "heapsort":

            class ComparableDoc:
                def __init__(self, docid, text, ranker):
                    self.docid = docid
                    self.text = text
                    self.ranker = ranker

                def __gt__(self, other):
                    out = self.ranker.compare(query, [self.text, other.text])
                    return out[0] == "Passage A" and out[1] == "Passage B"

            arr = [
                ComparableDoc(docid=doc.docid, text=doc.text, ranker=self)
                for doc in ranking
            ]
            self.heapSort(arr, self.k)
            ranking = [
                self._make_result(docid=doc.docid, score=-i, text=None)
                for i, doc in enumerate(reversed(arr))
            ]
        elif self.method == "bubblesort":
            k = min(self.k, len(ranking))
            last_end = len(ranking) - 1
            for i in range(k):
                current_ind = last_end
                is_change = False
                while True:
                    if current_ind <= i:
                        break
                    doc1 = ranking[current_ind]
                    doc2 = ranking[current_ind - 1]
                    output = self.compare(query, [doc1.text, doc2.text])
                    if output[0] == "Passage A" and output[1] == "Passage B":
                        ranking[current_ind - 1], ranking[current_ind] = (
                            ranking[current_ind],
                            ranking[current_ind - 1],
                        )
                        if not is_change:
                            is_change = True
                            if last_end != len(ranking) - 1:
                                last_end += 1
                    if not is_change:
                        last_end -= 1
                    current_ind -= 1
        else:
            raise NotImplementedError(f"Method {self.method} is not implemented.")

        results: List[Any] = []
        top_doc_ids = set()
        rank = 1
        for doc in ranking[: self.k]:
            top_doc_ids.add(doc.docid)
            results.append(self._make_result(docid=doc.docid, score=-rank, text=None))
            rank += 1
        for doc in original_ranking:
            if doc.docid not in top_doc_ids:
                results.append(
                    self._make_result(docid=doc.docid, score=-rank, text=None)
                )
                rank += 1

        self._finalize_run_stats(time.perf_counter() - start)
        return results

    def _record_prompt_batch(self, prompt_count: int) -> None:
        self._run_prompt_batches += 1
        self._run_prompt_count += int(prompt_count)
        self._run_http_calls += 1

    def infer_prompts(self, prompts: List[str]) -> List[str]:
        raise NotImplementedError

    def _make_result(self, docid: str, score: float, text: Optional[str]):
        try:
            from llmrankers.rankers import SearchResult  # type: ignore

            return SearchResult(docid=docid, score=float(score), text=text)
        except Exception:
            return _SimpleSearchResult(docid=docid, score=float(score), text=text or "")

    def _normalize_ranking(self, ranking: Sequence[Any]) -> List[Any]:
        if not ranking:
            return []
        first = ranking[0]
        if hasattr(first, "docid") and hasattr(first, "text"):
            return list(ranking)
        if isinstance(first, dict):
            out: List[Any] = []
            for idx, item in enumerate(ranking):
                docid = str(item.get("docid", idx))
                text = str(item.get("text", "") or "")
                score = float(item.get("score", 0.0) or 0.0)
                out.append(self._make_result(docid=docid, score=score, text=text))
            return out
        if isinstance(first, str):
            return [
                self._make_result(docid=str(idx), score=0.0, text=str(text))
                for idx, text in enumerate(ranking)
            ]
        raise ValueError("Unsupported ranking item type for pairwise ranker.")

    def truncate(self, text: str, length: int) -> str:
        if length <= 0:
            return ""
        if getattr(self, "tokenizer", None) is None:
            return text[:length]
        return self.tokenizer.convert_tokens_to_string(
            self.tokenizer.tokenize(text)[:length]
        )


class VLLMPairwiseRanker(_ParallelPairwiseBase):
    """
    llmrankers-compatible pairwise reranker using a vLLM OpenAI-compatible endpoint.

    This backend uses batched prompt inference for compare/allpair and supports
    concurrent in-flight requests for allpair throughput.
    """

    def __init__(
        self,
        model_name_or_path: str,
        method: str = "heapsort",
        batch_size: int = 2,
        k: int = 10,
        tokenizer_name_or_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        prompt_template: Optional[str] = None,
        temperature: float = 0.0,
        timeout_s: int = 60,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "EMPTY",
        max_new_tokens: int = 1,
        request_timeout_s: Optional[int] = None,
        bidirectional_compare: bool = False,
        max_pairs: Optional[int] = None,
        pair_batch_prompts: int = 32,
        pair_max_inflight: int = 4,
        pair_retry: int = 2,
        enable_cache: bool = True,
    ) -> None:
        super().__init__(
            method=method,
            batch_size=batch_size,
            k=k,
            prompt_template=prompt_template,
            max_pairs=max_pairs,
            pair_batch_prompts=pair_batch_prompts,
            pair_max_inflight=pair_max_inflight,
            pair_retry=pair_retry,
            enable_cache=enable_cache,
        )
        self.llm = model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path or model_name_or_path
        self.cache_dir = cache_dir
        self.temperature = float(temperature)
        self.timeout_s = int(timeout_s)
        self.base_url = str(base_url).rstrip("/")
        self.api_key = str(api_key or "").strip()
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.request_timeout_s = (
            int(request_timeout_s) if request_timeout_s is not None else int(timeout_s)
        )
        self.bidirectional_compare = bool(bidirectional_compare)
        self.tokenizer = self._init_tokenizer()

    def _init_tokenizer(self):
        try:
            from transformers import AutoTokenizer
        except Exception:
            return None

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name_or_path,
                cache_dir=self.cache_dir,
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
        except Exception:
            return None

        if hasattr(tokenizer, "use_default_system_prompt"):
            tokenizer.use_default_system_prompt = False
        if "vicuna" and "v1.5" in self.tokenizer_name_or_path:
            tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\\'s questions.' %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{{ system_message }}{% endif %}{% if message['role'] == 'user' %}{{ ' USER: ' + message['content'].strip() }}{% elif message['role'] == 'assistant' %}{{ ' ASSISTANT: ' + message['content'].strip() + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ ' ASSISTANT:' }}{% endif %}"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
        tokenizer.padding_side = "left"
        return tokenizer

    def _extract_choice_text(self, choice: Any) -> str:
        if isinstance(choice, dict):
            text = choice.get("text")
            if isinstance(text, str):
                return text
            message = choice.get("message")
            if isinstance(message, dict):
                msg_text = message.get("content")
                if isinstance(msg_text, str):
                    return msg_text
                return str(msg_text or "")
            return str(text or "")
        return str(choice or "")

    def _completion_batch_once(self, prompts: List[str]) -> List[str]:
        endpoint = f"{self.base_url}/completions"
        payload = {
            "model": self.llm,
            "prompt": prompts,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "top_p": 1.0,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(req, timeout=self.request_timeout_s) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as error:
            detail = ""
            try:
                detail = error.read().decode("utf-8", errors="ignore")
            except Exception:
                detail = ""
            detail = detail.strip()
            if len(detail) > 500:
                detail = detail[:500] + "...(truncated)"
            raise RuntimeError(
                f"vLLM completion failed: status={error.code}, endpoint={endpoint}, detail={detail}"
            ) from error
        except URLError as error:
            raise RuntimeError(
                f"vLLM endpoint unreachable: {endpoint} ({error})"
            ) from error
        except TimeoutError as error:
            raise RuntimeError(
                f"vLLM request timed out after {self.request_timeout_s}s: {endpoint}"
            ) from error

        try:
            response_json = json.loads(raw)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"vLLM response is not valid JSON: {raw[:400]}"
            ) from error

        choices = response_json.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"vLLM response missing choices: {response_json}")

        usage = response_json.get("usage")
        usage_dict = usage if isinstance(usage, dict) else {}
        prompt_tokens = usage_dict.get("prompt_tokens")
        completion_tokens = usage_dict.get("completion_tokens")
        if isinstance(prompt_tokens, int):
            self.total_prompt_tokens += int(prompt_tokens)
        if isinstance(completion_tokens, int):
            self.total_completion_tokens += int(completion_tokens)

        outputs = [""] * len(prompts)
        for idx, choice in enumerate(choices):
            out_index = idx
            if isinstance(choice, dict) and isinstance(choice.get("index"), int):
                out_index = int(choice["index"])
            if out_index < 0 or out_index >= len(outputs):
                continue
            outputs[out_index] = self._extract_choice_text(choice).strip().upper()
        return outputs

    def _completion_batch(self, prompts: List[str]) -> List[str]:
        last_error: Optional[Exception] = None
        for attempt in range(self.pair_retry + 1):
            try:
                self._record_prompt_batch(len(prompts))
                return self._completion_batch_once(prompts)
            except Exception as error:
                last_error = error
                if attempt >= self.pair_retry:
                    raise
                self._run_retry_count += 1
                time.sleep(min(1.5, 0.2 * (2**attempt)))
        if last_error is not None:
            raise last_error
        raise RuntimeError("vLLM completion batch failed without explicit error.")

    def infer_prompts(self, prompts: List[str]) -> List[str]:
        if not prompts:
            return []
        generation_prompts = [self._build_generation_prompt(text) for text in prompts]
        chunks: List[List[str]] = []
        for offset in range(0, len(generation_prompts), self.pair_batch_prompts):
            chunks.append(generation_prompts[offset : offset + self.pair_batch_prompts])

        if len(chunks) == 1 or self.pair_max_inflight <= 1:
            outputs: List[str] = []
            for chunk in chunks:
                outputs.extend(self._completion_batch(chunk))
            return outputs

        results: List[List[str]] = [None] * len(chunks)  # type: ignore
        with ThreadPoolExecutor(max_workers=self.pair_max_inflight) as executor:
            future_map = {
                executor.submit(self._completion_batch, chunk): idx
                for idx, chunk in enumerate(chunks)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                results[idx] = future.result()

        flattened: List[str] = []
        for chunk_outputs in results:
            flattened.extend(chunk_outputs)
        return flattened


class HFParallelPairwiseRanker(_ParallelPairwiseBase):
    """
    Pairwise ranker that keeps llmrankers HF model loading behavior but replaces
    comparison/rerank with batched prompt inference.
    """

    def __init__(
        self,
        base_ranker: Any,
        method: str = "allpair",
        max_pairs: Optional[int] = None,
        pair_batch_prompts: int = 32,
        pair_max_inflight: int = 1,
        pair_retry: int = 0,
        enable_cache: bool = True,
    ) -> None:
        super().__init__(
            method=method,
            batch_size=int(getattr(base_ranker, "batch_size", 2)),
            k=int(getattr(base_ranker, "k", 10)),
            prompt_template=str(
                getattr(base_ranker, "prompt", LLMRANKERS_PAIRWISE_PROMPT)
            ),
            max_pairs=max_pairs,
            pair_batch_prompts=pair_batch_prompts,
            pair_max_inflight=pair_max_inflight,
            pair_retry=pair_retry,
            enable_cache=enable_cache,
        )
        self.base_ranker = base_ranker
        self.llm = getattr(base_ranker, "llm", None)
        self.tokenizer = getattr(base_ranker, "tokenizer", None)
        self.config = getattr(base_ranker, "config", None)
        self.decoder_input_ids = getattr(base_ranker, "decoder_input_ids", None)
        self.device = getattr(base_ranker, "device", None)

        try:
            self._hf_device = self.llm.device
        except Exception:
            try:
                import torch

                self._hf_device = next(self.llm.parameters()).device
            except Exception:
                self._hf_device = self.device

    def _infer_llama_qwen_batch(self, prompts: List[str]) -> List[str]:
        prompt_texts = [self._build_generation_prompt(text) for text in prompts]
        inputs = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding="longest",
        )
        input_ids = inputs["input_ids"].to(self._hf_device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._hf_device)
        self.total_prompt_tokens += input_ids.shape[0] * input_ids.shape[1]

        generate_kwargs = {
            "do_sample": False,
            "temperature": 0.0,
            "top_p": None,
            "max_new_tokens": 1,
        }
        if attention_mask is not None:
            generate_kwargs["attention_mask"] = attention_mask
        output_ids = self.llm.generate(input_ids, **generate_kwargs)
        self.total_completion_tokens += output_ids.shape[0] * output_ids.shape[1]

        input_len = input_ids.shape[1]
        outputs: List[str] = []
        for row in range(output_ids.shape[0]):
            text = self.tokenizer.decode(
                output_ids[row][input_len:],
                skip_special_tokens=True,
            )
            outputs.append(str(text or "").strip().upper())
        return outputs

    def _infer_t5_batch(self, prompts: List[str]) -> List[str]:
        input_ids = self.tokenizer(
            prompts,
            padding="longest",
            return_tensors="pt",
        ).input_ids.to(self.llm.device)
        self.total_prompt_tokens += input_ids.shape[0] * input_ids.shape[1]

        decoder_input_ids = self.decoder_input_ids
        if decoder_input_ids is not None and decoder_input_ids.shape[0] != len(prompts):
            decoder_input_ids = decoder_input_ids[: len(prompts), :]
        output_ids = self.llm.generate(
            input_ids,
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=2,
        )
        self.total_completion_tokens += output_ids.shape[0] * output_ids.shape[1]
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return [str(text or "").strip().upper() for text in outputs]

    def infer_prompts(self, prompts: List[str]) -> List[str]:
        if not prompts:
            return []
        model_type = str(getattr(self.config, "model_type", "") or "").lower()
        outputs: List[str] = []
        for offset in range(0, len(prompts), self.pair_batch_prompts):
            chunk = prompts[offset : offset + self.pair_batch_prompts]
            self._record_prompt_batch(len(chunk))
            if model_type == "t5":
                outputs.extend(self._infer_t5_batch(chunk))
            elif model_type in ("llama", "qwen2"):
                outputs.extend(self._infer_llama_qwen_batch(chunk))
            else:
                # Fall back to original ranker compare path if model type is unsupported.
                for start in range(0, len(chunk), 2):
                    pair_chunk = chunk[start : start + 2]
                    if len(pair_chunk) < 2:
                        pair_chunk = pair_chunk + [pair_chunk[0]]
                    inferred = self.base_ranker.compare(
                        query="",
                        docs=[pair_chunk[0], pair_chunk[1]],
                    )
                    outputs.extend([str(v or "").strip().upper() for v in inferred[:2]])
        return outputs


class LLMReranker:
    """
    Thin wrapper around llm-rankers pairwise reranking.

    Interface mirrors the retrieval pipeline needs:
    rerank(query, candidates, top_k) -> reordered candidates list.
    """

    def __init__(
        self,
        model: str,
        provider: str,
        method: str = "allpair",
        api_key_env: Optional[str] = None,
        api_key_file: Optional[str] = None,
        prompt_template: Optional[str] = None,
        tokenizer_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        debug_enabled: bool = False,
        debug_log_path: Optional[str] = None,
        debug_top_n: Optional[int] = None,
        batch_size: int = 8,
        max_pairs: Optional[int] = None,
        temperature: float = 0.0,
        timeout_s: int = 60,
        vllm_base_url: Optional[str] = None,
        vllm_api_key: Optional[str] = None,
        vllm_api_key_env: Optional[str] = None,
        vllm_api_key_file: Optional[str] = None,
        vllm_max_new_tokens: int = 1,
        vllm_request_timeout_s: Optional[int] = None,
        vllm_bidirectional_compare: bool = False,
        pair_batch_prompts: int = 32,
        pair_max_inflight: int = 4,
        pair_retry: int = 2,
        enable_pair_cache: bool = True,
        candidate_text_fields: Optional[Sequence[str]] = None,
    ) -> None:
        self.model = model
        self.provider = str(provider or "hf").lower()
        self.method = method
        self.api_key_env = api_key_env
        self.api_key_file = api_key_file
        self.vllm_base_url = vllm_base_url
        self.vllm_api_key = vllm_api_key
        self.vllm_api_key_env = vllm_api_key_env
        self.vllm_api_key_file = vllm_api_key_file
        self.vllm_max_new_tokens = int(vllm_max_new_tokens)
        self.vllm_request_timeout_s = (
            int(vllm_request_timeout_s) if vllm_request_timeout_s is not None else None
        )
        self.vllm_bidirectional_compare = bool(vllm_bidirectional_compare)
        self.pair_batch_prompts = int(pair_batch_prompts)
        self.pair_max_inflight = int(pair_max_inflight)
        self.pair_retry = int(pair_retry)
        self.enable_pair_cache = bool(enable_pair_cache)
        if prompt_template is None:
            prompt_template = (
                "You are ranking two candidate trajectories for a task.\n"
                'CURRENT STATE (query): "{query}"\n\n'
                'Trajectory A:\n"{doc1}"\n\n'
                'Trajectory B:\n"{doc2}"\n\n'
                "Which trajectory is more helpful for deciding the NEXT action from the current state?\n"
                "Prefer the trajectory that is both goal-relevant and consistent with the current state "
                "(objects/locations).\n"
                "If one trajectory assumes objects in places that contradict the current state, prefer the other.\n"
                "Output Passage A or Passage B:"
            )
        self.prompt_template = prompt_template
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.device = device
        self.cache_dir = cache_dir
        self.debug_enabled = debug_enabled
        self.debug_log_path = debug_log_path
        self.debug_top_n = debug_top_n
        self.batch_size = batch_size
        self.max_pairs = max_pairs
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.candidate_text_fields = list(candidate_text_fields or [])
        self._ranker = None

    def get_runtime_stats(self) -> Dict[str, Any]:
        ranker = self._get_ranker()
        getter = getattr(ranker, "get_runtime_stats", None)
        if callable(getter):
            try:
                stats = getter()
                if isinstance(stats, dict):
                    return stats
            except Exception:
                return {}
        return {}

    def rerank(
        self, query: str, candidates: List[Candidate], top_k: int
    ) -> List[Candidate]:
        """
        Rerank candidates with a pairwise LLM ranker.

        Args:
            query: Query string used for retrieval.
            candidates: List of raw candidate records from LanceDB.
            top_k: Number of candidates from the front of the list to rerank.

        Returns:
            Reordered candidate list (same objects, new order).
        """
        if not candidates or top_k <= 0:
            return candidates

        subset = candidates[: min(top_k, len(candidates))]
        texts = [self._build_candidate_text(c) for c in subset]
        if not any(texts):
            return candidates

        ranker = self._get_ranker()
        ranking = self._rank_texts(ranker, query, texts)
        reordered_subset = [subset[i] for i in ranking]
        if self.debug_enabled and self.debug_log_path:
            try:
                top_n = self.debug_top_n if self.debug_top_n is not None else top_k
                if self._order_changed(subset, reordered_subset):
                    self._write_debug_log(query, subset, reordered_subset, top_n)
            except Exception:
                # Best-effort logging only; do not affect reranking
                pass
        return reordered_subset + candidates[len(subset) :]

    def _build_candidate_text(self, candidate: Candidate) -> str:
        if not self.candidate_text_fields:
            return ""
        # This is the text shown to the pairwise reranker, not the final agent
        # prompt. Field order matters because it fixes the comparison template.
        return self._build_candidate_text_with_fields(
            candidate, self.candidate_text_fields
        )

    def _build_candidate_text_with_fields(
        self, candidate: Candidate, fields: Sequence[str]
    ) -> str:
        parts = []
        for field in fields:
            value = candidate.get(field, "")
            if value is None:
                continue
            parts.append(f"{field}: {value}")
        return " | ".join(parts)

    def _order_changed(self, before: List[Candidate], after: List[Candidate]) -> bool:
        return [self._candidate_id(c) for c in before] != [
            self._candidate_id(c) for c in after
        ]

    def _candidate_id(self, candidate: Candidate) -> str:
        if "thought_id" in candidate:
            return str(candidate["thought_id"])
        if "id" in candidate:
            return str(candidate["id"])
        if "task_name" in candidate and "variation_idx" in candidate:
            return f"{candidate.get('task_name')}-{candidate.get('variation_idx')}"
        return str(id(candidate))

    def _truncate(self, text: str, max_len: Optional[int]) -> str:
        if max_len is None or max_len <= 0:
            return text
        if len(text) <= max_len:
            return text
        return text[:max_len] + "...(truncated)"

    def _format_candidate(
        self, candidate: Candidate, max_len: Optional[int] = None
    ) -> str:
        parts = [f"id={self._candidate_id(candidate)}"]
        for field in self.candidate_text_fields:
            value = candidate.get(field, "")
            if value:
                parts.append(f"{field}={self._truncate(str(value), max_len)}")
        return " | ".join(parts)

    def _write_debug_log(
        self,
        query: str,
        before: List[Candidate],
        after: List[Candidate],
        top_n: int,
    ) -> None:
        import datetime

        os.makedirs(os.path.dirname(self.debug_log_path), exist_ok=True)
        top_n = min(max(top_n, 1), len(before))
        query_str = self._truncate(query, None)

        lines = [
            f"timestamp={datetime.datetime.utcnow().isoformat()}",
            f"top_n={top_n}",
            f"query_len={len(query)}",
            "query:",
            query_str,
            "prompt_template:",
            self.prompt_template or "",
            "before:",
        ]
        if self.prompt_template and before:
            doc1 = self._build_candidate_text(before[0])
            doc2 = self._build_candidate_text(before[1]) if len(before) > 1 else ""
            try:
                prompt_example = self.prompt_template.format(
                    query=query, doc1=doc1, doc2=doc2
                )
            except Exception:
                prompt_example = (
                    f"{self.prompt_template}\n\n[formatting failed for query/doc1/doc2]"
                )
            lines.extend(["prompt_example:", prompt_example])
        for idx, cand in enumerate(before[:top_n], 1):
            lines.append(f"  {idx}. {self._format_candidate(cand)}")
        lines.append("after:")
        for idx, cand in enumerate(after[:top_n], 1):
            lines.append(f"  {idx}. {self._format_candidate(cand)}")
        lines.append("order_changed: true")
        lines.append("-" * 80)

        with open(self.debug_log_path, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

    def _get_ranker(self):
        if self._ranker is not None:
            return self._ranker

        # vLLM uses the local pairwise backend directly; other providers go
        # through llm-rankers-compatible wrappers so the retrieval code stays uniform.
        if self.provider == "vllm":
            self._ranker = VLLMPairwiseRanker(
                model_name_or_path=self.model,
                method=self.method,
                batch_size=self.batch_size,
                tokenizer_name_or_path=(self.tokenizer_name_or_path or self.model),
                cache_dir=self.cache_dir,
                prompt_template=self.prompt_template,
                temperature=self.temperature,
                timeout_s=self.timeout_s,
                base_url=self.vllm_base_url or "http://127.0.0.1:8000/v1",
                api_key=self._resolve_vllm_api_key(),
                max_new_tokens=self.vllm_max_new_tokens,
                request_timeout_s=self.vllm_request_timeout_s,
                bidirectional_compare=self.vllm_bidirectional_compare,
                max_pairs=self.max_pairs,
                pair_batch_prompts=self.pair_batch_prompts,
                pair_max_inflight=self.pair_max_inflight,
                pair_retry=self.pair_retry,
                enable_cache=self.enable_pair_cache,
            )
            return self._ranker

        module = self._import_llm_rankers()
        pairwise_cls = self._resolve_pairwise_ranker(module)
        kwargs = self._build_ranker_kwargs(pairwise_cls)

        # Keep legacy openai path unchanged for compatibility.
        if self.provider == "openai":
            self._ranker = pairwise_cls(**kwargs)
            return self._ranker

        base_ranker = pairwise_cls(**kwargs)
        self._ranker = HFParallelPairwiseRanker(
            base_ranker=base_ranker,
            method=self.method,
            max_pairs=self.max_pairs,
            pair_batch_prompts=self.pair_batch_prompts,
            pair_max_inflight=self.pair_max_inflight,
            pair_retry=self.pair_retry,
            enable_cache=self.enable_pair_cache,
        )
        return self._ranker

    def _import_llm_rankers(self):
        try:
            import llmrankers as module  # type: ignore

            return module
        except Exception:
            try:
                import llm_rankers as module  # type: ignore

                return module
            except Exception as exc:
                raise ImportError(
                    "llm-rankers is required for LLM reranking. Install it and try again."
                ) from exc

    def _resolve_pairwise_ranker(self, module):
        pairwise = importlib.import_module(f"{module.__name__}.pairwise")
        if self.provider == "openai":
            self._ensure_openai_compatible()
            return pairwise.OpenAiPairwiseLlmRanker
        return pairwise.PairwiseLlmRanker

    def _build_ranker_kwargs(self, pairwise_cls) -> Dict[str, Any]:
        sig = inspect.signature(pairwise_cls.__init__)
        params = sig.parameters

        kwargs: Dict[str, Any] = {}
        method = "allpair" if self.method == "allpairs" else self.method
        if self.provider == "openai" and method in ("allpair", "allpairs"):
            raise ValueError(
                "OpenAI pairwise reranker does not support method='allpair'. Use heapsort or bubblesort."
            )

        self._maybe_set(
            kwargs,
            params,
            "model_name_or_path",
            self.model,
            alt_keys=["model", "model_name"],
        )
        tokenizer_name = self.tokenizer_name_or_path or self.model
        self._maybe_set(kwargs, params, "tokenizer_name_or_path", tokenizer_name)
        device = self.device
        if device is None and "device" in params:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self._maybe_set(kwargs, params, "device", device)
        self._maybe_set(kwargs, params, "method", method, alt_keys=["sort_method"])
        self._maybe_set(
            kwargs, params, "prompt_template", self.prompt_template, alt_keys=["prompt"]
        )
        self._maybe_set(kwargs, params, "batch_size", self.batch_size)
        self._maybe_set(kwargs, params, "max_pairs", self.max_pairs)
        self._maybe_set(kwargs, params, "temperature", self.temperature)
        self._maybe_set(
            kwargs, params, "timeout_s", self.timeout_s, alt_keys=["timeout"]
        )
        self._maybe_set(kwargs, params, "cache_dir", self.cache_dir)

        api_key = os.getenv(self.api_key_env) if self.api_key_env else None
        if self.api_key_file:
            api_key = self._read_api_key_file(self.api_key_file)
        if api_key:
            self._maybe_set(kwargs, params, "api_key", api_key)

        return kwargs

    def _maybe_set(
        self,
        kwargs: Dict[str, Any],
        params: Dict[str, inspect.Parameter],
        key: str,
        value: Any,
        alt_keys: Optional[Sequence[str]] = None,
    ) -> None:
        if value is None:
            return
        if key in params:
            kwargs[key] = value
            return
        for alt_key in alt_keys or []:
            if alt_key in params:
                kwargs[alt_key] = value
                return

    def _rank_texts(self, ranker, query: str, texts: List[str]) -> List[int]:
        if hasattr(ranker, "rerank"):
            try:
                # Some llm-rankers backends expect SearchResult objects with doc ids,
                # while local wrappers accept raw text lists. Try the richer path first.
                from llmrankers.rankers import SearchResult

                ranking = [
                    SearchResult(docid=str(i), score=0.0, text=text)
                    for i, text in enumerate(texts)
                ]
                if hasattr(ranker, "k"):
                    ranker.k = len(ranking)
                result = ranker.rerank(query, ranking)
            except Exception:
                result = ranker.rerank(query, texts)
        elif hasattr(ranker, "rank"):
            result = ranker.rank(query, texts)
        else:
            raise AttributeError(
                "PairwiseRanker does not provide rerank or rank method."
            )

        return self._normalize_ranking_result(result, texts)

    def _normalize_ranking_result(self, result: Any, texts: List[str]) -> List[int]:
        # Different reranker backends return rankings in different shapes
        # (SearchResult objects, indices, tuples, dicts). Normalize everything
        # into a single list of integer positions before reordering candidates.
        if isinstance(result, list):
            if result and hasattr(result[0], "docid"):
                indices = [int(r.docid) for r in result]
                return self._sanitize_indices(indices, len(texts))
            if result and all(isinstance(x, int) for x in result):
                return self._sanitize_indices(result, len(texts))
            if result and all(isinstance(x, tuple) and len(x) >= 1 for x in result):
                indices = []
                for item in result:
                    if isinstance(item[0], int):
                        indices.append(item[0])
                    else:
                        indices.append(self._index_from_text(item[0], texts))
                return self._sanitize_indices(indices, len(texts))
            if result and all(isinstance(x, dict) for x in result):
                indices = []
                for item in result:
                    if "doc_id" in item:
                        indices.append(int(item["doc_id"]))
                    elif "index" in item:
                        indices.append(int(item["index"]))
                    elif "text" in item:
                        indices.append(self._index_from_text(item["text"], texts))
                return self._sanitize_indices(indices, len(texts))

        if (
            isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[0], list)
        ):
            return self._normalize_ranking_result(result[0], texts)

        raise ValueError("Unrecognized reranker output format.")

    def _sanitize_indices(self, indices: List[int], size: int) -> List[int]:
        seen = set()
        final = []
        for idx in indices:
            if idx < 0 or idx >= size:
                continue
            if idx in seen:
                continue
            seen.add(idx)
            final.append(idx)

        # Append any missing indices to keep a total order
        for idx in range(size):
            if idx not in seen:
                final.append(idx)
        return final

    def _index_from_text(self, text: Any, texts: List[str]) -> int:
        try:
            return texts.index(text)
        except ValueError:
            return 0

    def _read_api_key_file(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return handle.read().strip()
        except OSError as exc:
            raise RuntimeError(f"Failed to read api_key_file: {path}") from exc

    def _resolve_vllm_api_key(self) -> str:
        if self.vllm_api_key:
            return str(self.vllm_api_key)
        if self.vllm_api_key_file:
            return self._read_api_key_file(self.vllm_api_key_file)
        if self.vllm_api_key_env:
            value = os.getenv(self.vllm_api_key_env)
            if value:
                return value
        if self.api_key_file:
            value = self._read_api_key_file(self.api_key_file)
            if value:
                return value
        if self.api_key_env:
            value = os.getenv(self.api_key_env)
            if value:
                return value
        return "EMPTY"

    def _ensure_openai_compatible(self) -> None:
        try:
            from importlib import metadata

            version = metadata.version("openai")
        except Exception:
            return

        parts = version.split(".")
        if parts and parts[0].isdigit() and int(parts[0]) >= 1:
            raise RuntimeError(
                "llm-rankers 0.0.2 expects the legacy OpenAI SDK (openai<1). "
                "This environment pins openai 1.x, which is incompatible with "
                "OpenAiPairwiseLlmRanker. Options: (1) run reranking with a local HF "
                "model (provider != 'openai'), (2) use a separate env with openai<1, "
                "or (3) update llm-rankers to a version that supports openai>=1."
            )
