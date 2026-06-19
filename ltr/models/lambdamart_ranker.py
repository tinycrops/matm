"""LambdaMART ranking model using XGBoost."""

import os
import numpy as np
import xgboost as xgb
from typing import Optional, Dict, Any
import pickle

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from ltr.models.base import BaseRanker


class LambdaMARTRanker(BaseRanker):
    """LambdaMART ranker using XGBoost's pairwise/listwise objectives."""

    def __init__(
        self,
        ltr_style: str = "pairwise",
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        objective: str = "rank:ndcg",
        lambdarank_pair_method: str = "topk",
        lambdarank_num_pair_per_sample: int = 8,
        **kwargs,
    ):
        """
        Initialize LambdaMART ranker.

        Args:
            ltr_style: Learning to rank style ('pairwise' or 'listwise')
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            objective: Ranking objective ('rank:ndcg', 'rank:map', 'rank:pairwise')
            lambdarank_pair_method: Pair construction method ('topk' or 'mean')
            lambdarank_num_pair_per_sample: Number of pairs per sample
            **kwargs: Additional XGBoost parameters
        """
        if ltr_style not in ["pairwise", "listwise"]:
            raise ValueError("LambdaMARTRanker supports 'pairwise' or 'listwise' style")

        super().__init__(ltr_style=ltr_style)

        self.objective = objective
        self.lambdarank_pair_method = lambdarank_pair_method
        self.lambdarank_num_pair_per_sample = lambdarank_num_pair_per_sample

        self.model = xgb.XGBRanker(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective=objective,
            lambdarank_pair_method=lambdarank_pair_method,
            lambdarank_num_pair_per_sample=lambdarank_num_pair_per_sample,
            tree_method="hist",
            **kwargs,
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
        """Train the LambdaMART model."""
        self.n_features = X.shape[1]

        # Prepare callbacks for wandb logging
        callbacks = kwargs.pop("callbacks", [])

        if (
            self.wandb_run
            and WANDB_AVAILABLE
            and (X_val is not None and y_val is not None and qid_val is not None)
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
        if X_val is not None and y_val is not None and qid_val is not None:
            eval_set = [(X_val, y_val)]
            kwargs["eval_set"] = eval_set
            # eval_group needs to be a list of arrays containing group sizes (docs per query)
            # Compute group sizes for validation set
            val_unique_qids, val_group_sizes = np.unique(qid_val, return_counts=True)
            # Sort by qid to match the order of the data
            val_sort_idx = np.argsort(val_unique_qids)
            val_group_sizes = val_group_sizes[val_sort_idx]
            kwargs["eval_group"] = [val_group_sizes.tolist()]
            kwargs["verbose"] = kwargs.get("verbose", False)

        if callbacks:
            kwargs["callbacks"] = callbacks

        self.model.fit(X, y, qid=qid, **kwargs)
        self.is_fitted = True

    def predict(self, X: np.ndarray, qid: np.ndarray) -> np.ndarray:
        """Predict relevance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        scores = self.model.predict(X)
        return scores

    def save(self, filepath: str):
        """Save the model."""
        import os

        # XGBoost models can be saved directly
        xgb_path = filepath + ".xgb"
        self.model.save_model(xgb_path)

        # Save metadata separately
        metadata = {
            "ltr_style": self.ltr_style,
            "n_features": self.n_features,
            "objective": self.objective,
            "lambdarank_pair_method": self.lambdarank_pair_method,
            "lambdarank_num_pair_per_sample": self.lambdarank_num_pair_per_sample,
            "xgb_path": xgb_path,
        }
        with open(filepath + ".meta", "wb") as f:
            pickle.dump(metadata, f)

    @classmethod
    def load(cls, filepath: str):
        """Load the model."""
        # Load metadata
        meta_path = filepath + ".meta"
        if not os.path.exists(meta_path):
            # Try loading without extension
            meta_path = filepath
            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"Model metadata not found: {meta_path}")

        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        # Create instance
        instance = cls(
            ltr_style=metadata["ltr_style"],
            objective=metadata["objective"],
            lambdarank_pair_method=metadata["lambdarank_pair_method"],
            lambdarank_num_pair_per_sample=metadata["lambdarank_num_pair_per_sample"],
        )

        # Load XGBoost model.
        # Some legacy metadata stores a relative xgb_path (e.g. "models/..."),
        # which may be invalid under a different runtime cwd.
        meta_dir = os.path.dirname(os.path.abspath(meta_path))
        filepath_abs = os.path.abspath(filepath)
        xgb_path_meta = metadata.get("xgb_path")

        candidates = []
        if xgb_path_meta:
            candidates.append(str(xgb_path_meta))
            if not os.path.isabs(str(xgb_path_meta)):
                candidates.append(os.path.join(meta_dir, str(xgb_path_meta)))
                candidates.append(
                    os.path.join(
                        os.path.dirname(filepath_abs),
                        os.path.basename(str(xgb_path_meta)),
                    )
                )

        candidates.extend(
            [
                filepath + ".xgb",
                filepath_abs + ".xgb",
                filepath,
                filepath_abs,
            ]
        )

        xgb_path = None
        tried = []
        for candidate in candidates:
            if not candidate or candidate in tried:
                continue
            tried.append(candidate)
            if os.path.exists(candidate):
                xgb_path = candidate
                break

        if xgb_path is None:
            raise FileNotFoundError(
                "XGBoost model not found. Tried: " + ", ".join(tried)
            )

        instance.model.load_model(xgb_path)
        instance.n_features = metadata["n_features"]
        instance.is_fitted = True

        return instance
