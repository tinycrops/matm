"""SVMRank ranking model."""

import numpy as np
from typing import Optional, Dict, Any
import pickle
import os
import subprocess
import tempfile
from pathlib import Path

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from ltr.models.base import BaseRanker


class SVMRanker(BaseRanker):
    """SVMRank ranker (pairwise approach)."""

    def __init__(
        self,
        ltr_style: str = "pairwise",
        c: float = 1.0,
        epsilon: float = 0.001,
        **kwargs,
    ):
        """
        Initialize SVMRank ranker.

        Args:
            ltr_style: Learning to rank style (default: 'pairwise')
            c: Regularization parameter
            epsilon: Tolerance parameter
            **kwargs: Additional parameters
        """
        super().__init__(ltr_style=ltr_style)

        self.c = c
        self.epsilon = epsilon
        self.model_path = None
        self.n_features = None

        candidate_dirs = self._candidate_binary_dirs()
        # Check if svm_rank_learn and svm_rank_classify are available
        self.svm_rank_learn_path = self._find_svm_rank_binary(
            "svm_rank_learn", candidate_dirs
        )
        self.svm_rank_classify_path = self._find_svm_rank_binary(
            "svm_rank_classify", candidate_dirs
        )

        if not self.svm_rank_learn_path or not self.svm_rank_classify_path:
            env_hint = os.environ.get("SVMRANK_BIN_DIR", "<unset>")
            raise RuntimeError(
                "SVMRank binaries not found. Please install SVMRank and ensure "
                "svm_rank_learn and svm_rank_classify are in your PATH "
                "(or set SVMRANK_BIN_DIR). "
                f"SVMRANK_BIN_DIR={env_hint}. "
                "See: https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html"
            )

    def _candidate_binary_dirs(self) -> list[str]:
        """Collect directories to probe for SVMRank binaries."""
        dirs: list[str] = []

        env_dir = str(os.environ.get("SVMRANK_BIN_DIR", "") or "").strip()
        if env_dir:
            for entry in env_dir.split(os.pathsep):
                entry = entry.strip()
                if entry:
                    dirs.append(entry)

        repo_default = Path(__file__).resolve().parents[1] / "svm_rank"
        dirs.append(str(repo_default))

        for entry in os.environ.get("PATH", "").split(os.pathsep):
            entry = entry.strip()
            if entry:
                dirs.append(entry)

        deduped: list[str] = []
        seen = set()
        for entry in dirs:
            if entry in seen:
                continue
            seen.add(entry)
            deduped.append(entry)
        return deduped

    def _find_svm_rank_binary(
        self, binary_name: str, candidate_dirs: Optional[list[str]] = None
    ) -> Optional[str]:
        """Find SVMRank binary in SVMRANK_BIN_DIR/repo default/PATH."""
        search_dirs = candidate_dirs or self._candidate_binary_dirs()
        for path in search_dirs:
            binary_path = os.path.join(path, binary_name)
            if os.path.isfile(binary_path) and os.access(binary_path, os.X_OK):
                return binary_path
        return None

    def _convert_to_svm_rank_format(
        self, X: np.ndarray, y: np.ndarray, qid: np.ndarray, filepath: str
    ):
        """Convert data to SVMRank format."""
        with open(filepath, "w") as f:
            unique_qids = np.unique(qid)
            for q in unique_qids:
                mask = qid == q
                X_query = X[mask]
                y_query = y[mask]

                for i, (features, label) in enumerate(zip(X_query, y_query)):
                    # SVMRank format: qid:qid label:label 1:value1 2:value2 ...
                    line = f"{int(label)} qid:{int(q)}"
                    for j, val in enumerate(features, 1):
                        if val != 0:  # Skip zero values
                            line += f" {j}:{val}"
                    f.write(line + "\n")

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
        """Train the SVMRank model."""
        self.n_features = X.shape[1]

        # Create temporary files
        with tempfile.TemporaryDirectory() as tmpdir:
            train_file = os.path.join(tmpdir, "train.dat")
            model_file = os.path.join(tmpdir, "model.dat")

            # Convert to SVMRank format
            self._convert_to_svm_rank_format(X, y, qid, train_file)

            # Train model
            cmd = [
                self.svm_rank_learn_path,
                "-c",
                str(self.c),
                "-e",
                str(self.epsilon),
                train_file,
                model_file,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"SVMRank training failed: {result.stderr}")

            # Read model file
            with open(model_file, "rb") as f:
                self.model_data = f.read()

        self.is_fitted = True

        # Log training completion to wandb
        if self.wandb_run and WANDB_AVAILABLE:
            wandb.log({"train_completed": 1})

            # If validation data is provided, evaluate and log
            if X_val is not None and y_val is not None and qid_val is not None:
                try:
                    y_pred = self.predict(X_val, qid_val)
                    # Calculate a simple MSE loss for validation
                    val_loss = np.mean((y_val - y_pred) ** 2)
                    wandb.log({"val_loss": val_loss})
                except Exception as e:
                    # If prediction fails, just log that validation was attempted
                    pass

    def predict(self, X: np.ndarray, qid: np.ndarray) -> np.ndarray:
        """Predict relevance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Create temporary files
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.dat")
            model_file = os.path.join(tmpdir, "model.dat")
            predictions_file = os.path.join(tmpdir, "predictions.dat")

            # Write model
            with open(model_file, "wb") as f:
                f.write(self.model_data)

            # Convert test data to SVMRank format (use dummy labels)
            dummy_y = np.zeros(len(X))
            self._convert_to_svm_rank_format(X, dummy_y, qid, test_file)

            # Predict
            cmd = [self.svm_rank_classify_path, test_file, model_file, predictions_file]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"SVMRank prediction failed: {result.stderr}")

            # Read predictions
            scores = []
            with open(predictions_file, "r") as f:
                for line in f:
                    scores.append(float(line.strip()))

            return np.array(scores)

    def save(self, filepath: str):
        """Save the model."""
        model_data = {
            "model_data": self.model_data,
            "ltr_style": self.ltr_style,
            "n_features": self.n_features,
            "c": self.c,
            "epsilon": self.epsilon,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, filepath: str):
        """Load the model."""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        instance = cls(
            ltr_style=model_data["ltr_style"],
            c=model_data["c"],
            epsilon=model_data["epsilon"],
        )

        instance.model_data = model_data["model_data"]
        instance.n_features = model_data["n_features"]
        instance.is_fitted = True

        return instance
