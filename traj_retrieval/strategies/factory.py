# traj_retrieval/strategies/factory.py
# Factory for creating experiment strategy instances

from typing import Dict, List, Union
from .base_strategy import BaseExperimentStrategy

# DO NOT import async_strategies or sync_strategies at module level
# This prevents asyncio contamination for sync-only usage
# Instead, use lazy imports in the factory methods


class ExperimentFactory:
    """
    Factory class for creating experiment strategy instances.
    Supports both async and sync strategies.
    """

    # Registry of available async experiment strategies (class names for lazy loading)
    _async_strategies = {
        "action_string_direct": "ActionStringAsyncStrategy",
    }

    # Registry of available sync experiment strategies (class names for lazy loading)
    _sync_strategies = {
        "action_string_direct": "ActionStringSyncStrategy",
    }

    @classmethod
    def get_available_experiments(cls) -> List[str]:
        """Get list of available experiment types (same for async and sync)."""
        return list(cls._async_strategies.keys())

    @classmethod
    def create_async_strategy(cls, experiment_type: str) -> BaseExperimentStrategy:
        """
        Create an async experiment strategy instance.

        Args:
            experiment_type: The type of experiment strategy to create

        Returns:
            Instance of the requested async strategy

        Raises:
            ValueError: If experiment_type is not supported
        """
        if experiment_type not in cls._async_strategies:
            available = ", ".join(cls.get_available_experiments())
            raise ValueError(
                f"Unknown experiment type: '{experiment_type}'. "
                f"Available types: {available}"
            )

        # Lazy import to avoid importing asyncio when not needed
        from .async_strategies import ActionStringAsyncStrategy

        strategy_classes = {
            "action_string_direct": ActionStringAsyncStrategy,
        }

        strategy_class = strategy_classes[experiment_type]
        return strategy_class()

    @classmethod
    def create_sync_strategy(cls, experiment_type: str) -> BaseExperimentStrategy:
        """
        Create a sync experiment strategy instance.

        Args:
            experiment_type: The type of experiment strategy to create

        Returns:
            Instance of the requested sync strategy

        Raises:
            ValueError: If experiment_type is not supported
        """
        if experiment_type not in cls._sync_strategies:
            available = ", ".join(cls.get_available_experiments())
            raise ValueError(
                f"Unknown experiment type: '{experiment_type}'. "
                f"Available types: {available}"
            )

        # Lazy import - CRITICAL: does not import async_strategies
        from .sync_strategies import ActionStringSyncStrategy

        strategy_classes = {
            "action_string_direct": ActionStringSyncStrategy,
        }

        strategy_class = strategy_classes[experiment_type]
        return strategy_class()

    @classmethod
    def create_strategy(
        cls, experiment_type: str, use_sync: bool = False
    ) -> BaseExperimentStrategy:
        """
        Create an experiment strategy instance (async or sync).

        Args:
            experiment_type: The type of experiment strategy to create
            use_sync: If True, create sync strategy; if False, create async strategy

        Returns:
            Instance of the requested strategy

        Raises:
            ValueError: If experiment_type is not supported
        """
        if use_sync:
            return cls.create_sync_strategy(experiment_type)
        else:
            return cls.create_async_strategy(experiment_type)

    @classmethod
    def get_strategy_info(cls) -> Dict[str, str]:
        """
        Get information about all available strategies.

        Returns:
            Dictionary mapping strategy names to their descriptions
        """
        info = {}
        for name in cls._async_strategies.keys():
            # Use lazy import for info as well
            try:
                from .async_strategies import ActionStringAsyncStrategy

                strategy_classes = {
                    "action_string_direct": ActionStringAsyncStrategy,
                }
                strategy = strategy_classes[name]()
                info[
                    name
                ] = f"{strategy.get_strategy_name()}: {strategy.__class__.__doc__ or 'No description'}"
            except Exception as e:
                info[name] = f"{name}: Error creating strategy - {e}"

        return info


# Convenience functions for easy access
def get_experiment_strategy(
    experiment_type: str, use_sync: bool = False
) -> BaseExperimentStrategy:
    """
    Convenience function to get an experiment strategy.

    Args:
        experiment_type: The type of experiment strategy to create
        use_sync: If True, get sync strategy; if False, get async strategy (default)

    Returns:
        Instance of the requested strategy
    """
    return ExperimentFactory.create_strategy(experiment_type, use_sync=use_sync)
