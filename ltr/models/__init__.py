"""Learning to Rank model implementations."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

from ltr.models.base import BaseRanker

__all__ = [
    "BaseRanker",
    "XGBoostRanker",
    "LambdaMARTRanker",
    "FFNRanker",
    "ListNetRanker",
    "SVMRanker",
]

_LAZY_IMPORTS: Dict[str, Tuple[str, str]] = {
    "XGBoostRanker": ("ltr.models.xgboost_ranker", "XGBoostRanker"),
    "LambdaMARTRanker": ("ltr.models.lambdamart_ranker", "LambdaMARTRanker"),
    "FFNRanker": ("ltr.models.ffn_ranker", "FFNRanker"),
    "ListNetRanker": ("ltr.models.listnet_ranker", "ListNetRanker"),
    "SVMRanker": ("ltr.models.svmrank_ranker", "SVMRanker"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(_LAZY_IMPORTS.keys()))
