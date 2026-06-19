#!/usr/bin/env python3
"""
Shared helpers for baseline retrieval row -> candidate conversion and
LLM allpair ranking/feature computation.

Both:
  - scripts/convert_raw_json_to_ltr_llm_with_llm_features.py
  - scripts/eval_lancedb_with_cascade_rerankers_llm_features.py
should call this module to keep runtime feature generation aligned.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from .llm_allpair_feature_utils import compute_llm_allpair_rank_and_features

Candidate = Dict[str, Any]
Row = Dict[str, Any]
GetRowDocIdFn = Callable[[str, Row], Optional[str]]
SearchRowsFn = Callable[[str, int], Sequence[Row]]
RowToCandidateFn = Callable[[str, Row, int], Optional[Candidate]]


def build_candidate_from_row(
    dataset: str,
    row: Row,
    retrieved_rank: int,
    get_row_doc_id: GetRowDocIdFn,
) -> Optional[Candidate]:
    doc_id = get_row_doc_id(dataset, row)
    if not isinstance(doc_id, str) or not doc_id:
        return None
    candidate = dict(row)
    candidate["docid"] = doc_id
    candidate["retrieved_rank"] = int(retrieved_rank)
    return candidate


def compute_allpair_for_candidates(
    query_text: str,
    candidates: Sequence[Candidate],
    llm_reranker: Any,
    ranker: Any,
    allpair_top_k: Optional[int] = None,
) -> Dict[str, Any]:
    total = len(candidates)
    if total == 0:
        return {
            "n": 0,
            "ordered_docids": [],
            "reranked_prefix": [],
            "prefix_candidates": [],
            "features_by_docid": {},
            "rank_initial": {},
            "rank_llm": {},
            "total_points": {},
            "winrate_raw": {},
            "winrate_norm": {},
        }

    if allpair_top_k is None:
        allpair_top_k = total
    allpair_top_k = max(0, int(allpair_top_k))
    prefix = list(candidates[: min(allpair_top_k, total)])
    if not prefix:
        return {
            "n": 0,
            "ordered_docids": [],
            "reranked_prefix": [],
            "prefix_candidates": [],
            "features_by_docid": {},
            "rank_initial": {},
            "rank_llm": {},
            "total_points": {},
            "winrate_raw": {},
            "winrate_norm": {},
        }

    computed = compute_llm_allpair_rank_and_features(
        query_text=query_text,
        candidates=prefix,
        llm_reranker=llm_reranker,
        ranker=ranker,
    )

    ordered_docids = list(computed.get("ordered_docids", []))
    # Keep first seen candidate for each docid.
    by_docid: Dict[str, Candidate] = {}
    for candidate in prefix:
        doc_id = candidate.get("docid")
        if isinstance(doc_id, str) and doc_id and doc_id not in by_docid:
            by_docid[doc_id] = candidate

    reranked_prefix: List[Candidate] = []
    seen_docids = set()
    for doc_id in ordered_docids:
        candidate = by_docid.get(doc_id)
        if candidate is None:
            continue
        reranked_prefix.append(candidate)
        seen_docids.add(doc_id)
    # Append any missing docs in original prefix order.
    for candidate in prefix:
        doc_id = candidate.get("docid")
        if not isinstance(doc_id, str) or not doc_id:
            continue
        if doc_id in seen_docids:
            continue
        reranked_prefix.append(candidate)
        seen_docids.add(doc_id)

    out = dict(computed)
    out["prefix_candidates"] = prefix
    out["reranked_prefix"] = reranked_prefix
    out["ordered_docids"] = ordered_docids
    return out


def retrieve_and_compute_allpair(
    dataset: str,
    query_text: str,
    retrieval_top_k: int,
    llm_reranker: Any,
    ranker: Any,
    get_row_doc_id: GetRowDocIdFn,
    search_rows_fn: SearchRowsFn,
    allpair_top_k: Optional[int] = None,
    row_to_candidate: Optional[RowToCandidateFn] = None,
) -> Dict[str, Any]:
    rows = list(search_rows_fn(query_text, int(retrieval_top_k)))
    builder = row_to_candidate
    if builder is None:
        builder = lambda ds, row, rr: build_candidate_from_row(
            dataset=ds,
            row=row,
            retrieved_rank=rr,
            get_row_doc_id=get_row_doc_id,
        )

    candidates: List[Candidate] = []
    for retrieved_rank, row in enumerate(rows, start=1):
        candidate = builder(dataset, row, int(retrieved_rank))
        if candidate is None:
            continue
        candidates.append(candidate)

    allpair = compute_allpair_for_candidates(
        query_text=query_text,
        candidates=candidates,
        llm_reranker=llm_reranker,
        ranker=ranker,
        allpair_top_k=allpair_top_k,
    )
    allpair["retrieved_rows"] = rows
    allpair["candidates"] = candidates
    allpair["retrieved_topk"] = int(len(candidates))
    return allpair
