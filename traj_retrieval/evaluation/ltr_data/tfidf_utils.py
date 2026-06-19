#!/usr/bin/env python3
"""
Utility functions for loading and using TF-IDF vectorizers.

This module provides helper functions to:
1. Load pre-computed TF-IDF vectorizers from disk
2. Compute TF-IDF cosine similarities between texts
3. Compute bigram overlap similarities

Usage example in enrich_disagreements.py:

    from tfidf_utils import load_tfidf_vectorizers, compute_tfidf_similarity

    # During initialization
    vectorizers = load_tfidf_vectorizers(environment="alfworld")

    # During feature computation
    similarity = compute_tfidf_similarity(
        text1=query,
        text2=retrieved_key,
        vectorizer=vectorizers['full']
    )
"""

import pickle
import sys
from pathlib import Path
from typing import Dict, Optional, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def _install_numpy_pickle_compat() -> None:
    """
    Bridge numpy 2.x pickle module paths when running under numpy 1.x.

    Some precomputed vectorizer pickles reference `numpy._core.*`.
    numpy 1.x exposes `numpy.core.*`, so we alias those module names
    before unpickling to keep the artifacts loadable.
    """
    if "numpy._core" in sys.modules and "numpy._core.numeric" in sys.modules:
        return

    # Lazy imports keep startup behavior unchanged when aliases are unnecessary.
    import numpy.core as numpy_core  # pylint: disable=import-outside-toplevel
    import numpy.core.numeric as numpy_core_numeric  # pylint: disable=import-outside-toplevel

    sys.modules.setdefault("numpy._core", numpy_core)
    sys.modules.setdefault("numpy._core.numeric", numpy_core_numeric)


def load_tfidf_vectorizers(
    environment: str, base_dir: Optional[Path] = None
) -> Dict[str, TfidfVectorizer]:
    """
    Load all TF-IDF vectorizers for a given environment.

    Args:
        environment: Environment name ("alfworld", "webarena")
        base_dir: Base directory containing environment folders.
                 If None, uses directory of this script.

    Returns:
        Dictionary mapping component name to TfidfVectorizer:
        {
            'full': TfidfVectorizer,
            'goal': TfidfVectorizer,
            'state': TfidfVectorizer,
            'context': TfidfVectorizer,
        }

    Raises:
        FileNotFoundError: If vectorizer files don't exist
        ValueError: If environment is invalid
    """
    valid_environments = ["alfworld", "webarena"]
    if environment not in valid_environments:
        raise ValueError(
            f"Invalid environment: {environment}. Must be one of {valid_environments}"
        )

    if base_dir is None:
        base_dir = Path(__file__).parent

    env_dir = base_dir / environment

    if not env_dir.exists():
        raise FileNotFoundError(f"Environment directory not found: {env_dir}")

    vectorizers = {}
    components = ["full", "goal", "state", "context"]

    for component in components:
        vectorizer_path = env_dir / f"tfidf_vectorizer_{component}.pkl"

        if not vectorizer_path.exists():
            raise FileNotFoundError(
                f"TF-IDF vectorizer not found: {vectorizer_path}\n"
                f"Run calculate_tf_idf_corpus.py first to build vectorizers."
            )

        _install_numpy_pickle_compat()
        with open(vectorizer_path, "rb") as f:
            vectorizers[component] = pickle.load(f)

    return vectorizers


def compute_tfidf_similarity(
    text1: str, text2: str, vectorizer: TfidfVectorizer
) -> float:
    """
    Compute TF-IDF cosine similarity between two texts.

    Args:
        text1: First text
        text2: Second text
        vectorizer: Fitted TfidfVectorizer

    Returns:
        Cosine similarity score in range [0, 1]
        Returns 0.0 if either text is empty

    Note:
        Higher values indicate more similarity (1.0 = identical, 0.0 = no overlap)
    """
    # Handle empty texts
    if not text1.strip() or not text2.strip():
        return 0.0

    try:
        # Transform texts to TF-IDF vectors
        vec1 = vectorizer.transform([text1])
        vec2 = vectorizer.transform([text2])

        # Compute cosine similarity
        similarity = cosine_similarity(vec1, vec2)[0, 0]

        # Ensure valid range [0, 1]
        return float(np.clip(similarity, 0.0, 1.0))

    except Exception as e:
        # If transformation fails, return 0
        print(f"Warning: TF-IDF similarity computation failed: {e}")
        return 0.0


def extract_bigrams(text: str) -> Set[Tuple[str, str]]:
    """
    Extract bigrams (consecutive word pairs) from text.

    Args:
        text: Input text

    Returns:
        Set of (word1, word2) tuples

    Example:
        >>> extract_bigrams("go to the shelf")
        {('go', 'to'), ('to', 'the'), ('the', 'shelf')}
    """
    tokens = text.lower().split()

    if len(tokens) < 2:
        return set()

    bigrams = set()
    for i in range(len(tokens) - 1):
        bigrams.add((tokens[i], tokens[i + 1]))

    return bigrams


def compute_bigram_overlap(text1: str, text2: str) -> float:
    """
    Compute bigram Jaccard similarity between two texts.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Jaccard similarity of bigram sets in range [0, 1]
        Returns 0.0 if either text has no bigrams

    Formula:
        Jaccard = |intersection| / |union|

    Note:
        Higher values indicate more similarity (1.0 = identical, 0.0 = no overlap)
    """
    bigrams1 = extract_bigrams(text1)
    bigrams2 = extract_bigrams(text2)

    if not bigrams1 or not bigrams2:
        return 0.0

    intersection = bigrams1.intersection(bigrams2)
    union = bigrams1.union(bigrams2)

    if not union:
        return 0.0

    return len(intersection) / len(union)


def compute_all_tfidf_features(
    query: str,
    retrieved_key: str,
    query_goal: str,
    retrieved_goal: str,
    query_state: str,
    retrieved_state: str,
    query_context: str,
    retrieved_context: str,
    vectorizers: Dict[str, TfidfVectorizer],
) -> Dict[str, float]:
    """
    Compute all TF-IDF and bigram overlap features for a query-trajectory pair.

    Args:
        query: Full query text
        retrieved_key: Full retrieved trajectory key text
        query_goal: Query goal component
        retrieved_goal: Retrieved trajectory goal component
        query_state: Query state component
        retrieved_state: Retrieved trajectory state component
        query_context: Query context component
        retrieved_context: Retrieved trajectory context component
        vectorizers: Dictionary of TfidfVectorizers

    Returns:
        Dictionary of features:
        {
            'text_overlap_tfidf': float,
            'text_overlap_bigram': float,
            'goal_only_overlap_tfidf': float,
            'goal_only_overlap_bigram': float,
            'state_only_overlap_tfidf': float,
            'state_only_overlap_bigram': float,
            'context_only_overlap_tfidf': float or None,
            'context_only_overlap_bigram': float or None,
        }
    """
    features = {}

    # Full text features
    features["text_overlap_tfidf"] = compute_tfidf_similarity(
        query, retrieved_key, vectorizers["full"]
    )
    features["text_overlap_bigram"] = compute_bigram_overlap(query, retrieved_key)

    # Goal-only features
    features["goal_only_overlap_tfidf"] = compute_tfidf_similarity(
        query_goal, retrieved_goal, vectorizers["goal"]
    )
    features["goal_only_overlap_bigram"] = compute_bigram_overlap(
        query_goal, retrieved_goal
    )

    # State-only features
    features["state_only_overlap_tfidf"] = compute_tfidf_similarity(
        query_state, retrieved_state, vectorizers["state"]
    )
    features["state_only_overlap_bigram"] = compute_bigram_overlap(
        query_state, retrieved_state
    )

    # Context-only features (None if either context is empty)
    query_context_stripped = query_context.strip()
    retrieved_context_stripped = retrieved_context.strip()

    if query_context_stripped and retrieved_context_stripped:
        features["context_only_overlap_tfidf"] = compute_tfidf_similarity(
            query_context_stripped, retrieved_context_stripped, vectorizers["context"]
        )
        features["context_only_overlap_bigram"] = compute_bigram_overlap(
            query_context_stripped, retrieved_context_stripped
        )
    else:
        features["context_only_overlap_tfidf"] = None
        features["context_only_overlap_bigram"] = None

    return features


# Example usage and testing
if __name__ == "__main__":
    print("TF-IDF Utils - Example Usage\n")

    # Example 1: Load vectorizers
    print("Example 1: Loading vectorizers")
    print("-" * 50)
    try:
        vectorizers = load_tfidf_vectorizers("alfworld")
        print(f"✓ Loaded vectorizers for alfworld")
        print(f"  Components: {list(vectorizers.keys())}")
        print(f"  Full vocabulary size: {len(vectorizers['full'].vocabulary_)}")
    except FileNotFoundError as e:
        print(f"✗ {e}")
        print("  Run calculate_tf_idf_corpus.py first!")

    print("\n")

    # Example 2: Compute similarities
    print("Example 2: Computing similarities")
    print("-" * 50)

    text1 = "go to the shelf and take the statue"
    text2 = "go to the desk and take the book"

    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    print()

    # Bigram overlap (doesn't need vectorizer)
    bigram_sim = compute_bigram_overlap(text1, text2)
    print(f"Bigram overlap: {bigram_sim:.4f}")

    # TF-IDF similarity (needs vectorizer)
    if "vectorizers" in locals():
        tfidf_sim = compute_tfidf_similarity(text1, text2, vectorizers["full"])
        print(f"TF-IDF cosine similarity: {tfidf_sim:.4f}")

    print("\n")

    # Example 3: Extract bigrams
    print("Example 3: Bigram extraction")
    print("-" * 50)

    text = "go to shelf"
    bigrams = extract_bigrams(text)
    print(f"Text: '{text}'")
    print(f"Bigrams: {bigrams}")
