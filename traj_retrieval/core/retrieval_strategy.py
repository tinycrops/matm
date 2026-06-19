# traj_retrieval/core/retrieval_strategy.py
# Generic retrieval strategies for memory-augmented agents

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from .base_handler import EnvironmentHandler


class RetrievalResult:
    """
    Generic container for retrieval results.
    Can represent trajectories, documents, or any other retrieved memory.
    """

    def __init__(
        self,
        raw_data: Dict[str, Any],
        score: float,
        formatted_context: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            raw_data: Raw retrieved data (trajectory, document, etc.)
            score: Similarity/relevance score
            formatted_context: Human-readable formatted text ready for LLM prompt
            metadata: Additional metadata about the retrieval
        """
        self.raw_data = raw_data
        self.score = score
        self.formatted_context = formatted_context
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return (
            f"RetrievalResult(score={self.score:.4f}, "
            f"context_len={len(self.formatted_context)}, "
            f"metadata={self.metadata})"
        )


class RetrievalStrategy(ABC):
    """
    Abstract base class for retrieval strategies.

    A retrieval strategy handles:
    1. Loading/managing indices
    2. Encoding queries
    3. Searching for relevant memories
    4. Coordinating with environment handler for formatting
    """

    @abstractmethod
    def retrieve(
        self,
        query: str = None,
        env_handler: EnvironmentHandler = None,
        task_name: str = None,
        k: int = 1,
        **kwargs,
    ) -> Optional[RetrievalResult]:
        """
        Retrieve relevant memory for the given query.

        Args:
            query: Optional pre-built query string (if None, build from kwargs)
            env_handler: Environment handler to help with formatting
            task_name: Name of the task (for index selection)
            k: Number of results to retrieve
            **kwargs: Additional strategy-specific parameters (goal, observation, etc.)

        Returns:
            RetrievalResult containing formatted context, or None if retrieval fails
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the name of this retrieval strategy."""
        pass

    def refresh(self) -> None:
        """Refresh any cached retrieval resources if the strategy supports it."""
        return None


class DocumentRetrievalStrategy(RetrievalStrategy):
    """
    Placeholder for document-based retrieval (e.g., task descriptions, hints).

    This could be implemented in the future to retrieve relevant documents
    instead of full trajectories.
    """

    def __init__(self, documents_dir: str, model_name: str = "intfloat/e5-base"):
        self.documents_dir = documents_dir
        self.model_name = model_name
        print(f"[Retrieval] Initialized DocumentRetrievalStrategy (not implemented)")

    def retrieve(
        self,
        query: str = None,
        env_handler: EnvironmentHandler = None,
        task_name: str = None,
        k: int = 3,
        **kwargs,
    ) -> Optional[RetrievalResult]:
        """Retrieve relevant documents (not implemented)."""
        print(f"[Retrieval] DocumentRetrievalStrategy not implemented yet")
        return None

    def get_strategy_name(self) -> str:
        return "document"


class RetrievalManager:
    """
    High-level manager for retrieval operations.

    Coordinates between retrieval strategies and environment handlers,
    managing when and how to retrieve memories.
    """

    def __init__(self, strategy: RetrievalStrategy, enabled: bool = True):
        """
        Initialize retrieval manager.

        Args:
            strategy: Retrieval strategy to use
            enabled: Whether retrieval is enabled (False = no-op)
        """
        self.strategy = strategy
        self.enabled = enabled
        self._retrieval_count = 0

        if enabled:
            print(
                f"[RetrievalManager] Initialized with strategy: {strategy.get_strategy_name()}"
            )
        else:
            print(f"[RetrievalManager] Retrieval disabled (strategy='none')")

    def retrieve(
        self,
        env_handler: EnvironmentHandler,
        task_name: str,
        query: str = None,
        **kwargs,
    ) -> Optional[RetrievalResult]:
        """
        Perform retrieval using the configured strategy.

        Args:
            env_handler: Environment handler for formatting
            task_name: Task name
            query: Optional pre-built query string (if None, strategy builds it from kwargs)
            **kwargs: Context parameters for query building (goal, observation, etc.)

        Returns:
            RetrievalResult or None if retrieval is disabled/fails
        """
        if not self.enabled:
            return None

        result = self.strategy.retrieve(
            query=query, env_handler=env_handler, task_name=task_name, **kwargs
        )

        if result is not None:
            self._retrieval_count += 1

        return result

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about retrieval operations."""
        return {
            "enabled": self.enabled,
            "strategy": self.strategy.get_strategy_name() if self.enabled else "none",
            "total_retrievals": self._retrieval_count,
        }

    def refresh(self) -> None:
        """Refresh cached retrieval resources for subsequent retrieval calls."""
        self.strategy.refresh()

    @staticmethod
    def create_from_config(
        frequency_strategy: str,
        retrieval_type: str,
        indices_dir: str,
        search_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> "RetrievalManager":
        """
        Factory method to create a retrieval manager from configuration.

        This is the new preferred method that supports the updated config structure.
        Uses LanceDB as the vector store.

        Args:
            frequency_strategy: When to retrieve ("none", "t0", "every_10", "agentic")
            retrieval_type: What to retrieve ("trajectory", "document")
            indices_dir: Directory for indices
            search_config: Optional search configuration for filters and query params
            **kwargs: Additional parameters for strategy

        Returns:
            RetrievalManager instance
        """
        # Import strategies here to avoid circular dependencies
        from .lancedb_retrieval import LanceDBRetrievalStrategy

        # If frequency_strategy is "none", create disabled manager
        if frequency_strategy == "none":
            # Create a dummy strategy (doesn't matter which one)
            dummy_strategy = LanceDBRetrievalStrategy(
                indices_dir=indices_dir,
                table_name=kwargs.get("table_name", "alfworld"),
                search_config=search_config,
            )
            return RetrievalManager(strategy=dummy_strategy, enabled=False)

        # Select strategy based on retrieval_type (always uses LanceDB)
        if retrieval_type == "trajectory":
            strategy = LanceDBRetrievalStrategy(
                indices_dir=indices_dir,
                model_name=kwargs.get("model_name", "intfloat/e5-base"),
                table_name=kwargs.get("table_name", "alfworld"),
                search_config=search_config,
            )
        elif retrieval_type == "document":
            # Document retrieval not yet implemented
            strategy = DocumentRetrievalStrategy(
                documents_dir=indices_dir,
                model_name=kwargs.get("model_name", "intfloat/e5-base"),
            )
        else:
            raise ValueError(
                f"Unknown retrieval_type: '{retrieval_type}'. "
                f"Available: ['trajectory', 'document']"
            )

        return RetrievalManager(strategy=strategy, enabled=True)
