#!/usr/bin/env python3
"""Train a learning-to-rank (LTR) reranker for MATM.

Trains one of the supported ranker families on the chunk-aligned LTR feature
table produced by the LTR data pipeline
(`ltr/data_out/<environment>/ltr_train.tsv`) and writes the trained model to
`ltr/ltr_models/<environment>/<model_type>_model.<ext>`, using the exact
filenames the retrieval runtime expects (see
`traj_retrieval/core/reranker_orchestrator.py`).

The TSV is expected in the format `qid, label, feature_1, ..., feature_n`,
which is the output of the LTR data pipeline described in the README
("LTR Pipeline" section).

Examples:
    python -m ltr.train --environment alfworld --model-type svmrank
    python -m ltr.train --environment webarena --model-type lambdamart
    python -m ltr.train --environment alfworld --model-type ffn --epochs 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the repo root is importable regardless of the current working dir.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import importlib

from ltr.data import loader as ltr_loader

# Mirrors the registry/extension map used by the retrieval runtime so that the
# produced filenames are loadable at evaluation time without extra config.
MODEL_REGISTRY = {
    "xgboost": ("ltr.models.xgboost_ranker", "XGBoostRanker"),
    "lambdamart": ("ltr.models.lambdamart_ranker", "LambdaMARTRanker"),
    "ffn": ("ltr.models.ffn_ranker", "FFNRanker"),
    "listnet": ("ltr.models.listnet_ranker", "ListNetRanker"),
    "svmrank": ("ltr.models.svmrank_ranker", "SVMRanker"),
}

EXTENSION_MAP = {
    "xgboost": ".pkl",
    "lambdamart": ".pkl",
    "svmrank": ".pkl",
    "ffn": ".pt",
    "listnet": ".pt",
}

DEFAULT_DATA_DIR = REPO_ROOT / "ltr" / "data_out"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "ltr" / "ltr_models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--environment",
        required=True,
        choices=["alfworld", "webarena"],
        help="Benchmark whose LTR data to train on.",
    )
    parser.add_argument(
        "--model-type",
        required=True,
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Ranker family to train.",
    )
    parser.add_argument(
        "--train-file",
        default=None,
        help="Explicit training TSV. Defaults to "
        "<data-dir>/<environment>/ltr_train.tsv.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Root directory holding per-environment LTR TSVs.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Root output directory for trained models.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Explicit output path (without runtime-managed suffixes). "
        "Overrides --output-dir.",
    )
    parser.add_argument(
        "--ltr-style",
        default=None,
        help="Override the ranker's learning-to-rank style "
        "(pointwise/pairwise/listwise). Defaults to the model's own default.",
    )
    parser.add_argument(
        "--normalize-features",
        action="store_true",
        help="Apply query-level feature normalization when loading the TSV.",
    )
    # Optional hyperparameters (passed through only to relevant model families).
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="FFN/ListNet only.")
    parser.add_argument("--batch-size", type=int, default=None, help="FFN/ListNet only.")
    parser.add_argument("--device", default=None, help="FFN/ListNet only (cpu/cuda).")
    parser.add_argument(
        "--n-estimators", type=int, default=None, help="LambdaMART/XGBoost only."
    )
    parser.add_argument(
        "--max-depth", type=int, default=None, help="LambdaMART/XGBoost only."
    )
    parser.add_argument("--c", type=float, default=None, help="SVMRank only.")
    parser.add_argument("--epsilon", type=float, default=None, help="SVMRank only.")
    return parser.parse_args()


def build_model_kwargs(args: argparse.Namespace) -> dict:
    """Collect only the hyperparameters relevant to the chosen model family.

    Passing irrelevant kwargs (e.g. ``epochs`` to an XGBoost-backed ranker)
    would raise, so each family receives a curated subset.
    """
    kwargs: dict = {}
    if args.ltr_style is not None:
        kwargs["ltr_style"] = args.ltr_style

    model_type = args.model_type
    if model_type in ("ffn", "listnet"):
        if args.learning_rate is not None:
            kwargs["learning_rate"] = args.learning_rate
        if args.epochs is not None:
            kwargs["epochs"] = args.epochs
        if args.batch_size is not None:
            kwargs["batch_size"] = args.batch_size
        if args.device is not None:
            kwargs["device"] = args.device
    elif model_type in ("lambdamart", "xgboost"):
        if args.n_estimators is not None:
            kwargs["n_estimators"] = args.n_estimators
        if args.max_depth is not None:
            kwargs["max_depth"] = args.max_depth
        if args.learning_rate is not None:
            kwargs["learning_rate"] = args.learning_rate
    elif model_type == "svmrank":
        if args.c is not None:
            kwargs["c"] = args.c
        if args.epsilon is not None:
            kwargs["epsilon"] = args.epsilon
    return kwargs


def resolve_train_file(args: argparse.Namespace) -> Path:
    if args.train_file:
        return Path(args.train_file)
    return Path(args.data_dir) / args.environment / "ltr_train.tsv"


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output_path:
        return Path(args.output_path)
    suffix = EXTENSION_MAP[args.model_type]
    return Path(args.output_dir) / args.environment / f"{args.model_type}_model{suffix}"


def main() -> int:
    args = parse_args()

    train_file = resolve_train_file(args)
    if not train_file.is_file():
        raise FileNotFoundError(
            f"Training TSV not found: {train_file}\n"
            "Generate it first via the LTR data pipeline (see README, "
            "'LTR Pipeline'). Note that ltr/data_out/ is gitignored."
        )

    output_path = resolve_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Environment : {args.environment}")
    print(f"Model type  : {args.model_type}")
    print(f"Train file  : {train_file}")
    print(f"Output path : {output_path}")

    X, y, qid = ltr_loader.load_tsv_data(
        str(train_file), normalize_features=args.normalize_features
    )
    print(
        f"Loaded {X.shape[0]} rows, {X.shape[1]} features, "
        f"{len(set(qid.tolist()))} query groups."
    )

    module_path, class_name = MODEL_REGISTRY[args.model_type]
    model_class = getattr(importlib.import_module(module_path), class_name)
    model = model_class(**build_model_kwargs(args))

    print("Training...")
    model.fit(X, y, qid)

    model.save(str(output_path))
    print(f"Saved model to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
