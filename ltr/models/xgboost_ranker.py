"""XGBoost-based ranking model."""

import numpy as np
import xgboost as xgb
from typing import Optional, Dict, Any
import pickle
import os

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from ltr.models.base import BaseRanker


class XGBoostRanker(BaseRanker):
    """XGBoost ranker using pointwise approach."""

    def __init__(
        self,
        ltr_style: str = "pointwise",
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        **kwargs,
    ):
        """
        Initialize XGBoost ranker.

        Args:
            ltr_style: Learning to rank style (default: 'pointwise')
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            **kwargs: Additional XGBoost parameters
        """
        super().__init__(ltr_style=ltr_style)

        # For pointwise, we use binary classification
        if ltr_style == "pointwise":
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                **kwargs,
            )
        else:
            raise ValueError(
                f"XGBoostRanker only supports 'pointwise' style. Use LambdaMARTRanker for 'pairwise' or 'listwise'."
            )

        self.n_features = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        qid: np.ndarray,
        X_val=None,
        y_val=None,
        qid_val=None,
        **kwargs,
    ):
        """Train the XGBoost model."""
        self.n_features = X.shape[1]

        # Prepare callbacks for wandb logging
        callbacks = kwargs.pop("callbacks", [])

        if (
            self.wandb_run
            and WANDB_AVAILABLE
            and (X_val is not None and y_val is not None)
        ):
            try:
                # Try to use wandb's XGBoost integration callback
                from wandb.integration.xgboost import (
                    WandbCallback as WandbXGBoostCallback,
                )

                callbacks.append(WandbXGBoostCallback(log_model=False))
            except ImportError:
                # Fallback: Create a simple callback that logs metrics
                class SimpleWandbCallback:
                    def __init__(self, wandb_run):
                        self.wandb_run = wandb_run
                        self.iteration = 0

                    def __call__(self, env):
                        # This is called during XGBoost training
                        if env.evaluation_result_list:
                            log_dict = {"iteration": self.iteration}
                            for (
                                dataset_name,
                                metric_name,
                                metric_value,
                            ) in env.evaluation_result_list:
                                log_dict[f"{dataset_name}-{metric_name}"] = metric_value
                            wandb.log(log_dict)
                            self.iteration += 1

                callbacks.append(SimpleWandbCallback(self.wandb_run))

        # Add validation set if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            kwargs["eval_set"] = eval_set
            kwargs["verbose"] = kwargs.get("verbose", False)

        if callbacks:
            kwargs["callbacks"] = callbacks

        self.model.fit(X, y, **kwargs)
        self.is_fitted = True

    def predict(self, X: np.ndarray, qid: np.ndarray) -> np.ndarray:
        """Predict relevance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # For binary classification, use predict_proba for scores
        scores = self.model.predict_proba(X)[:, 1]
        return scores

    def save(self, filepath: str):
        """Save the model."""
        model_data = {
            "model": self.model,
            "ltr_style": self.ltr_style,
            "n_features": self.n_features,
        }
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, filepath: str):
        """Load the model."""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        instance = cls(ltr_style=model_data["ltr_style"])
        instance.model = model_data["model"]
        instance.n_features = model_data["n_features"]
        instance.is_fitted = True

        return instance
