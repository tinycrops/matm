#!/usr/bin/env python3
"""
Shared helper for LLM allpair stage outputs and derived runtime features.

This module intentionally mirrors the allpair scoring/ordering semantics used
in scripts/add_llm_features_to_splits.py and
scripts/convert_raw_json_to_ltr_llm_with_llm_features.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple


LLM_ALLPAIR_FEATURE_NAMES: Tuple[str, str, str] = (
    "llm_allpair_winrate_norm_top20",
    "llm_recip_rank_top20",
    "llm_recip_rank_gain_top20",
)


def zero_llm_allpair_features() -> Dict[str, float]:
    return {
        "llm_allpair_winrate_norm_top20": 0.0,
        "llm_recip_rank_top20": 0.0,
        "llm_recip_rank_gain_top20": 0.0,
    }


def compute_llm_allpair_rank_and_features(
    query_text: str,
    candidates: Sequence[Dict[str, Any]],
    llm_reranker: Any,
    ranker: Any,
) -> Dict[str, Any]:
    """
    Compute allpair ranking and 3 LLM runtime features in one pass.

    Returns:
      {
        "n": int,
        "ordered_docids": List[str],  # LLM reranked order
        "rank_initial": Dict[str, int],
        "rank_llm": Dict[str, int],
        "total_points": Dict[str, float],
        "winrate_raw": Dict[str, float],
        "winrate_norm": Dict[str, float],
        "features_by_docid": Dict[str, Dict[str, float]],
      }
    """
    n = len(candidates)
    if n == 0:
        return {
            "n": 0,
            "ordered_docids": [],
            "rank_initial": {},
            "rank_llm": {},
            "total_points": {},
            "winrate_raw": {},
            "winrate_norm": {},
            "features_by_docid": {},
        }

    texts = [llm_reranker._build_candidate_text(candidate) for candidate in candidates]
    initial_docids = [str(candidate["docid"]) for candidate in candidates]
    rank_initial = {doc_id: idx + 1 for idx, doc_id in enumerate(initial_docids)}

    # Mirror wrapper behavior: if no candidate text can be built, keep order.
    if not any(texts):
        ordered_docids = list(initial_docids)
        rank_llm = dict(rank_initial)
        total_points = {doc_id: 0.0 for doc_id in initial_docids}
    else:
        ranking = [
            ranker._make_result(docid=doc_id, score=0.0, text=text)
            for doc_id, text in zip(initial_docids, texts)
        ]
        if hasattr(ranker, "k"):
            ranker.k = len(ranking)

        if len(ranking) >= 2:
            scores = ranker._allpair_scores(query_text, ranking)
        else:
            scores = {}

        # Reproduce _ParallelPairwiseBase.rerank(allpair) ordering exactly.
        scored = sorted(
            [
                ranker._make_result(docid=docid, score=score, text=None)
                for docid, score in scores.items()
            ],
            key=lambda x: x.score,
            reverse=True,
        )

        ordered_docids = []
        top_doc_ids = set()
        for doc in scored[: len(ranking)]:
            docid = str(doc.docid)
            top_doc_ids.add(docid)
            ordered_docids.append(docid)
        for doc in ranking:
            docid = str(doc.docid)
            if docid not in top_doc_ids:
                ordered_docids.append(docid)

        rank_llm = {doc_id: idx + 1 for idx, doc_id in enumerate(ordered_docids)}
        total_points = {
            doc_id: float(scores.get(doc_id, 0.0)) for doc_id in initial_docids
        }

    if n <= 1:
        winrate_raw = {doc_id: 0.5 for doc_id in initial_docids}
    else:
        denom = float(n - 1)
        winrate_raw = {
            doc_id: float(total_points[doc_id] / denom) for doc_id in initial_docids
        }

    min_q = min(winrate_raw.values())
    max_q = max(winrate_raw.values())
    if max_q == min_q:
        winrate_norm = {doc_id: 0.5 for doc_id in initial_docids}
    else:
        span = max_q - min_q
        winrate_norm = {
            doc_id: float((winrate_raw[doc_id] - min_q) / span)
            for doc_id in initial_docids
        }

    features_by_docid: Dict[str, Dict[str, float]] = {}
    for doc_id in initial_docids:
        r_initial = float(rank_initial[doc_id])
        r_llm = float(rank_llm[doc_id])
        recip_rank = 1.0 / r_llm
        recip_gain = recip_rank - (1.0 / r_initial)
        features_by_docid[doc_id] = {
            "llm_allpair_winrate_norm_top20": float(winrate_norm[doc_id]),
            "llm_recip_rank_top20": float(recip_rank),
            "llm_recip_rank_gain_top20": float(recip_gain),
        }

    return {
        "n": int(n),
        "ordered_docids": list(ordered_docids),
        "rank_initial": rank_initial,
        "rank_llm": rank_llm,
        "total_points": total_points,
        "winrate_raw": winrate_raw,
        "winrate_norm": winrate_norm,
        "features_by_docid": features_by_docid,
    }
