"""ListNet ranking model."""

import numpy as np
from typing import Optional, Dict, Any
import pickle

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from ltr.models.base import BaseRanker


class ListNetModel(nn.Module):
    """Neural network for ListNet."""

    def __init__(self, input_dim: int, hidden_dims: list = [128, 64]):
        super(ListNetModel, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()


class ListNetRanker(BaseRanker):
    """ListNet ranker (listwise approach)."""

    def __init__(
        self,
        ltr_style: str = "listwise",
        hidden_dims: list = [128, 64],
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Initialize ListNet ranker.

        Args:
            ltr_style: Learning to rank style (default: 'listwise')
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate
            epochs: Number of training epochs
            batch_size: Batch size
            device: Device to use ('cpu' or 'cuda')
            **kwargs: Additional parameters
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for ListNetRanker. Install with: pip install torch"
            )

        super().__init__(ltr_style=ltr_style)

        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

        self.model = None
        self.n_features = None
        self.optimizer = None

    def _top_one_probability(self, scores: torch.Tensor) -> torch.Tensor:
        """Compute top-one probability distribution."""
        exp_scores = torch.exp(scores)
        return exp_scores / exp_scores.sum()

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
        """Train the ListNet model."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")

        self.n_features = X.shape[1]
        self.model = ListNetModel(self.n_features, self.hidden_dims)
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Group data by query
        unique_qids = np.unique(qid)

        # Prepare validation data if provided
        has_val = X_val is not None and y_val is not None and qid_val is not None
        if has_val:
            val_unique_qids = np.unique(qid_val)

        # Training loop
        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0.0
            n_queries = 0

            # Shuffle queries
            shuffled_qids = np.random.permutation(unique_qids)

            for q in shuffled_qids:
                mask = qid == q
                X_query = X[mask]
                y_query = y[mask]

                if len(X_query) == 0:
                    continue

                # Convert to tensors
                X_tensor = torch.FloatTensor(X_query).to(self.device)
                y_tensor = torch.FloatTensor(y_query).to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                scores = self.model(X_tensor)

                # Compute top-one probabilities
                pred_probs = self._top_one_probability(scores)
                true_probs = self._top_one_probability(y_tensor)

                # Cross-entropy loss
                loss = -torch.sum(true_probs * torch.log(pred_probs + 1e-10))

                # Backward pass
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                n_queries += 1

            avg_train_loss = total_loss / n_queries if n_queries > 0 else 0.0

            # Validation loss
            val_loss = None
            if has_val:
                self.model.eval()
                total_val_loss = 0.0
                n_val_queries = 0

                with torch.no_grad():
                    for q in val_unique_qids:
                        mask = qid_val == q
                        X_val_query = X_val[mask]
                        y_val_query = y_val[mask]

                        if len(X_val_query) == 0:
                            continue

                        X_val_tensor = torch.FloatTensor(X_val_query).to(self.device)
                        y_val_tensor = torch.FloatTensor(y_val_query).to(self.device)

                        scores = self.model(X_val_tensor)
                        pred_probs = self._top_one_probability(scores)
                        true_probs = self._top_one_probability(y_val_tensor)
                        loss = -torch.sum(true_probs * torch.log(pred_probs + 1e-10))

                        total_val_loss += loss.item()
                        n_val_queries += 1

                val_loss = total_val_loss / n_val_queries if n_val_queries > 0 else 0.0
                self.model.train()

            # Log to wandb
            if self.wandb_run and WANDB_AVAILABLE:
                log_dict = {"epoch": epoch, "train_loss": avg_train_loss}
                if val_loss is not None:
                    log_dict["val_loss"] = val_loss
                wandb.log(log_dict)

            if n_queries > 0 and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{self.epochs}, Average Loss: {avg_train_loss:.4f}"
                )

        self.is_fitted = True

    def predict(self, X: np.ndarray, qid: np.ndarray) -> np.ndarray:
        """Predict relevance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")

        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            scores = self.model(X_tensor).cpu().numpy()

        return scores

    def save(self, filepath: str):
        """Save the model."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")

        model_data = {
            "model_state_dict": self.model.state_dict(),
            "ltr_style": self.ltr_style,
            "n_features": self.n_features,
            "hidden_dims": self.hidden_dims,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "device": self.device,
        }

        torch.save(model_data, filepath)

    @classmethod
    def load(cls, filepath: str):
        """Load the model."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")

        model_data = torch.load(filepath, map_location="cpu")

        instance = cls(
            ltr_style=model_data["ltr_style"],
            hidden_dims=model_data["hidden_dims"],
            learning_rate=model_data["learning_rate"],
            epochs=model_data["epochs"],
            batch_size=model_data["batch_size"],
            device=model_data["device"],
        )

        instance.n_features = model_data["n_features"]
        instance.model = ListNetModel(instance.n_features, instance.hidden_dims)
        instance.model.load_state_dict(model_data["model_state_dict"])
        instance.model.to(instance.device)
        instance.is_fitted = True

        return instance
