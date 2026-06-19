"""Base class for ranking models."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class BaseRanker(ABC):
    """Base class for all ranking models."""

    def __init__(self, ltr_style: str = "pairwise", **kwargs):
        """
        Initialize the ranker.

        Args:
            ltr_style: Learning to rank style ('pointwise', 'pairwise', 'listwise')
            **kwargs: Additional model-specific parameters
        """
        if ltr_style not in ["pointwise", "pairwise", "listwise"]:
            raise ValueError(
                f"Invalid ltr_style: {ltr_style}. Must be 'pointwise', 'pairwise', or 'listwise'"
            )

        self.ltr_style = ltr_style
        self.is_fitted = False
        self.wandb_run = None

    def set_wandb_run(self, wandb_run):
        """Set the wandb run for logging."""
        self.wandb_run = wandb_run

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, qid: np.ndarray, **kwargs):
        """
        Train the ranking model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            qid: Query IDs (n_samples,)
            **kwargs: Additional training parameters
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, qid: np.ndarray) -> np.ndarray:
        """
        Predict relevance scores for documents.

        Args:
            X: Feature matrix (n_samples, n_features)
            qid: Query IDs (n_samples,)

        Returns:
            Relevance scores (n_samples,)
        """
        pass

    @abstractmethod
    def save(self, filepath: str):
        """Save the model to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath: str):
        """Load the model from disk."""
        pass

    def rank(self, X: np.ndarray, qid: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Rank documents within each query group.

        Args:
            X: Feature matrix (n_samples, n_features)
            qid: Query IDs (n_samples,)

        Returns:
            Dictionary mapping query ID to ranked document indices
        """
        scores = self.predict(X, qid)

        # Group by query ID and sort by score
        unique_qids = np.unique(qid)
        rankings = {}

        for q in unique_qids:
            mask = qid == q
            indices = np.where(mask)[0]
            query_scores = scores[mask]

            # Sort by score (descending)
            sorted_indices = indices[np.argsort(query_scores)[::-1]]
            rankings[q] = sorted_indices

        return rankings
