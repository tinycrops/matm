"""Feed Forward Network ranking model."""

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


class FFNModel(nn.Module):
    """Five-layer feed-forward network."""

    def __init__(self, input_dim: int, hidden_dims: list = [128, 64, 32, 16]):
        super(FFNModel, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        # Output layer (single score)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()


class FFNRanker(BaseRanker):
    """Feed Forward Network ranker (pointwise approach)."""

    def __init__(
        self,
        ltr_style: str = "pointwise",
        hidden_dims: list = [128, 64, 32, 16],
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        device: str = "cpu",
        **kwargs
    ):
        """
        Initialize FFN ranker.

        Args:
            ltr_style: Learning to rank style (default: 'pointwise')
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate
            epochs: Number of training epochs
            batch_size: Batch size
            device: Device to use ('cpu' or 'cuda')
            **kwargs: Additional parameters
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for FFNRanker. Install with: pip install torch"
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
        self.criterion = nn.BCEWithLogitsLoss()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        qid: np.ndarray,
        X_val=None,
        y_val=None,
        qid_val=None,
        **kwargs
    ):
        """Train the FFN model."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")

        self.n_features = X.shape[1]
        self.model = FFNModel(self.n_features, self.hidden_dims)
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        # Convert validation data if provided
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)

        # Training loop
        self.model.train()
        n_samples = len(X)

        for epoch in range(self.epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_tensor[indices]
            y_shuffled = y_tensor[indices]

            # Mini-batch training
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, self.batch_size):
                batch_X = X_shuffled[i : i + self.batch_size]
                batch_y = y_shuffled[i : i + self.batch_size]

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / n_batches if n_batches > 0 else 0.0

            # Validation loss
            val_loss = None
            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = self.criterion(val_outputs, y_val_tensor).item()
                self.model.train()

            # Log to wandb
            if self.wandb_run and WANDB_AVAILABLE:
                log_dict = {"epoch": epoch, "train_loss": avg_train_loss}
                if val_loss is not None:
                    log_dict["val_loss"] = val_loss
                wandb.log(log_dict)

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
            # Apply sigmoid to get probabilities
            scores = 1 / (1 + np.exp(-scores))

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
        instance.model = FFNModel(instance.n_features, instance.hidden_dims)
        instance.model.load_state_dict(model_data["model_state_dict"])
        instance.model.to(instance.device)
        instance.is_fitted = True

        return instance
