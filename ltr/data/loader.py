"""Data loading utilities for TSV files."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def normalize_features_query_level(X: np.ndarray, qid: np.ndarray) -> np.ndarray:
    """
    Normalize features query-wise (per qid group) - STATELESS operation.

    For each query, normalize that query's documents using that query's own
    statistics (mean and std). This is the standard "query-level normalization"
    used in Learning to Rank.

    This is a purely local operation per query - no cross-query sharing of
    statistics. Each split (train/val/test) should be normalized independently
    using this function.

    Args:
        X: Feature matrix (n_samples, n_features)
        qid: Query IDs (n_samples,)

    Returns:
        Normalized feature matrix with same shape as X
    """
    X_normalized = X.astype(float).copy()

    for q in np.unique(qid):
        mask = qid == q
        X_q = X_normalized[mask]

        # Compute mean and std for this query's documents
        mean = X_q.mean(axis=0, keepdims=True)
        std = X_q.std(axis=0, keepdims=True)

        # Avoid division by zero (if all values are the same, std=0)
        std[std == 0] = 1.0

        X_normalized[mask] = (X_q - mean) / std

    return X_normalized


def load_tsv_data(
    file_path: str,
    normalize_features: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load training data from TSV file.

    Expected format: qid, label, feature_1, feature_2, ..., feature_n

    Note: If normalize_features=True, normalization is applied here. However,
    for proper train/val splits, you may want to load without normalization
    and apply normalize_features_query_level() separately after splitting.

    Args:
        file_path: Path to the TSV file
        normalize_features: If True, apply query-level feature normalization

    Returns:
        Tuple of (X, y, qid) where:
        - X: Feature matrix (n_samples, n_features)
        - y: Labels (n_samples,)
        - qid: Query IDs (n_samples,)
    """
    df = pd.read_csv(file_path, sep="\t")

    # Validate required columns
    if "qid" not in df.columns:
        raise ValueError("TSV file must contain 'qid' column")
    if "label" not in df.columns:
        raise ValueError("TSV file must contain 'label' column")

    # Extract qid and label
    qid = df["qid"].values
    y = df["label"].values

    # Extract features (all columns except qid and label)
    feature_cols = [col for col in df.columns if col not in ["qid", "label"]]
    X = df[feature_cols].values

    # Sort by qid to ensure proper grouping
    sorted_idx = np.argsort(qid)
    X = X[sorted_idx]
    y = y[sorted_idx]
    qid = qid[sorted_idx]

    # Apply query-level feature normalization if requested
    if normalize_features:
        X = normalize_features_query_level(X, qid)

    return X, y, qid


def load_test_tsv_data(
    file_path: str,
    normalize_features: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load test data from TSV file (no label column).

    Expected format: qid, feature_1, feature_2, ..., feature_n

    Args:
        file_path: Path to the TSV file
        normalize_features: If True, apply query-level feature normalization

    Returns:
        Tuple of (X, qid) where:
        - X: Feature matrix (n_samples, n_features)
        - qid: Query IDs (n_samples,)
    """
    df = pd.read_csv(file_path, sep="\t")

    # Validate required columns
    if "qid" not in df.columns:
        raise ValueError("TSV file must contain 'qid' column")

    # Extract qid
    qid = df["qid"].values

    # Extract features (all columns except qid)
    feature_cols = [col for col in df.columns if col != "qid"]
    X = df[feature_cols].values

    # Sort by qid to ensure proper grouping
    sorted_idx = np.argsort(qid)
    X = X[sorted_idx]
    qid = qid[sorted_idx]

    # Apply query-level feature normalization if requested
    if normalize_features:
        X = normalize_features_query_level(X, qid)

    return X, qid
