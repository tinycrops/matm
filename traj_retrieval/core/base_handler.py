# traj_retrieval/core/base_handler.py
# Base classes for environment handlers

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any


@dataclass
class TrajectoryNode:
    """
    Generic representation of a single node in a trajectory.
    Contains all the state information needed for decision-making at one step.
    """

    # Core state
    observation: str
    goal: str
    admissible_actions: List[str]

    # Context
    inventory: str = ""
    recent_history: str = ""
    trajectory_context: str = ""  # Retrieved trajectory if RAG is used

    # Metadata
    step_number: int = 0
    internal_step_count: int = 0
    done: bool = False

    # Additional info (environment-specific)
    extra_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/debugging."""
        return {
            "step_number": self.step_number,
            "internal_step_count": self.internal_step_count,
            "observation": self.observation,
            "goal": self.goal,
            "inventory": self.inventory,
            "admissible_actions_count": len(self.admissible_actions),
            "recent_history_length": len(self.recent_history),
            "trajectory_context_length": len(self.trajectory_context),
            "done": self.done,
            "extra_info": self.extra_info,
        }


@dataclass
class StepResult:
    """
    Generic representation of the result of taking an action in an environment.
    """

    observation: str
    reward: float
    done: bool
    score: float
    info: Dict[str, Any]
    internal_steps_consumed: int = 1  # How many internal steps this action consumed


class EnvironmentHandler(ABC):
    """
    Abstract base class for environment handlers.
    Defines the interface that any environment must implement to work with the
    generic trajectory traversal system.

    The handler is responsible for:
    1. Parsing evaluation set data into episodes
    2. Determining max steps for each episode
    3. Managing environment-specific initialization
    4. Providing generic interface for episode execution
    """

    @abstractmethod
    def parse_evaluation_set(
        self, evaluation_set_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Parse environment-specific evaluation set data into generic episode descriptors.

        Args:
            evaluation_set_data: Raw evaluation set data (can be in any environment-specific format)

        Returns:
            List of episode descriptors, each containing:
            - episode_id: Unique identifier for the episode
            - task_name: Name of the task (environment-specific)
            - variation_idx: Variation index
            - max_steps: Maximum steps for this episode
            - Any other environment-specific metadata
        """
        pass

    @abstractmethod
    def initialize_from_episode(self, episode_descriptor: Dict[str, Any]) -> None:
        """
        Initialize the environment for a specific episode using an episode descriptor.

        Args:
            episode_descriptor: Episode descriptor from parse_evaluation_set()
        """
        pass

    @abstractmethod
    def initialize(
        self, task_name: str, max_steps: int, variation_idx: Optional[int] = None
    ) -> None:
        """
        Initialize the environment for a specific task.

        Args:
            task_name: Name of the task to run
            max_steps: Maximum steps allowed for the episode
            variation_idx: Optional specific variation to use (None for random)
        """
        pass

    @abstractmethod
    def reset(self) -> TrajectoryNode:
        """
        Reset the environment and return the initial state as a TrajectoryNode.

        Returns:
            Initial TrajectoryNode with starting observation and state
        """
        pass

    @abstractmethod
    def step(self, action: str, current_node: TrajectoryNode) -> StepResult:
        """
        Execute an action in the environment.

        Args:
            action: The action to execute
            current_node: Current trajectory node (for context)

        Returns:
            StepResult containing new observation, reward, done flag, etc.
        """
        pass

    @abstractmethod
    def get_current_node(self, step_number: int) -> TrajectoryNode:
        """
        Get the current state as a TrajectoryNode.

        Args:
            step_number: Current step number in the episode

        Returns:
            TrajectoryNode representing current state
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up and close the environment."""
        pass

    @abstractmethod
    def get_task_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the current task (for logging/debugging).

        Returns:
            Dictionary with task metadata (task_name, variation, max_variations, etc.)
        """
        pass

    @abstractmethod
    def is_successful_episode(self, final_score: float) -> bool:
        """
        Determine if an episode is successful based on environment-specific criteria.

        Args:
            final_score: The final score of the episode

        Returns:
            True if the episode is considered successful, False otherwise
        """
        pass

    @property
    @abstractmethod
    def environment_name(self) -> str:
        """Return the name of this environment type."""
        pass

    @abstractmethod
    def format_retrieval_result(
        self,
        raw_data: Dict[str, Any],
        max_steps: int = 20,
        retrieval_type: str = "trajectory",
    ) -> str:
        """
        Format retrieved data for inclusion in LLM prompt.

        This method allows each environment to customize how retrieved memories
        (trajectories, documents, etc.) are presented to the LLM.

        Args:
            raw_data: Raw retrieved data (format depends on retrieval_type)
            max_steps: Maximum steps/items to include in formatted output
            retrieval_type: Type of retrieval ("trajectory", "document", etc.)

        Returns:
            Formatted string ready for LLM prompt
        """
        pass

    @abstractmethod
    def get_environment_description(self) -> str:
        """
        Get a description of the environment for LLM system prompts.

        Returns:
            String describing the environment layout, rules, and mechanics
        """
        pass

    @abstractmethod
    def get_action_types_description(self) -> str:
        """
        Get a description of available action types in the environment.

        Returns:
            String describing the types of actions the agent can take
        """
        pass

    def get_example_episode(self) -> str:
        """
        Get a one-shot example episode for the LLM (optional).

        Returns:
            String containing an example episode, or empty string if not available
        """
        return ""  # Default: no example

    @abstractmethod
    def build_retrieval_query(
        self,
        goal: str,
        observation: str,
        inventory: str = "",
        recent_history: str = "",
        current_step: int = 0,
        current_reward: float = 0.0,
    ) -> str:
        """
        Build query string for retrieval matching the format used during indexing.

        This method must match the exact format used when creating database entries
        in the preprocessing step (trajectory_entry.py). The format differs between
        environments (e.g., AlfWorld has no inventory, WebArena includes a URL).

        Args:
            goal: Task goal/description
            observation: Current observation
            inventory: Current inventory (environment-specific)
            recent_history: Recent action history
            current_step: Current step number
            current_reward: Current reward/score (environment-specific)

        Returns:
            Query string formatted to match database entries
        """
        pass

    def get_agent_call_policy(self) -> str:
        """
        Get the agent call policy for this environment.

        Returns the policy for how the agent should make LLM calls:
        - "only_normal": Always use normal (unstructured) LLM calls
        - "only_structured": Always use structured LLM calls with response_format
        - "normal_then_structured": Try normal first, fallback to structured on failure
        - "adaptive": Use structured if < 500 actions, else normal then structured

        Returns:
            Policy string (default: "normal_then_structured")
        """
        return getattr(self, "_agent_call_policy", "normal_then_structured")

    def validate_action(self, action: str, admissible_actions: List[str]) -> bool:
        """
        Validate if an action is admissible for this environment.

        Default implementation: exact string matching.
        Environments can override this for custom validation logic (e.g., template matching).

        Args:
            action: The action to validate
            admissible_actions: List of admissible actions

        Returns:
            True if action is valid, False otherwise
        """
        return action in admissible_actions

    def format_history_entry(
        self,
        step: int,
        observation: str,
        action: str,
        reward: float,
        reasoning: str = "",
        inventory: str = "",
        url: str = "",
    ) -> str:
        """
        Format a single history entry for inclusion in LLM context.

        Each environment can customize the format based on what information is available.
        For example:
        - WebArena: includes URL
        - Others: basic format

        Args:
            step: Step number
            observation: Observation text
            action: Action taken
            reward: Reward received
            reasoning: Reasoning text (optional)
            inventory: Inventory contents (environment-specific, default empty)
            url: Current URL (environment-specific, default empty)

        Returns:
            Formatted history entry string
        """
        entry = f"STEP {step}: OBSERVATION: {observation}"

        # Add inventory if non-empty
        if inventory and inventory.strip():
            entry += f" | INVENTORY: {inventory}"

        # Add URL if non-empty (e.g., WebArena)
        if url and url.strip():
            entry += f" | URL: {url}"

        # Add action and reward
        entry += f" | ACTION: {action} | REWARD: {reward}"

        # Add reasoning if available
        if reasoning and reasoning.strip():
            entry += f" | REASONING: {reasoning}"

        return entry

    def get_current_url(self) -> str:
        """
        Get the current URL for this environment.

        Most environments don't have URLs (return empty string).
        WebArena and similar web environments should override this.

        Returns:
            Current URL string, or empty string if not applicable
        """
        return ""

    def compute_observation_similarity(self, obs1: str, obs2: str) -> float:
        """
        Compute similarity between two observations for pseudo_simulate_till_closest_tk strategy.

        Default implementation: simple string equality (0.0 or 1.0).
        Environments can override this for more sophisticated similarity metrics.

        Args:
            obs1: First observation string
            obs2: Second observation string

        Returns:
            Similarity score between 0.0 (completely different) and 1.0 (identical)
        """
        # Simple string equality by default
        return 1.0 if obs1 == obs2 else 0.0

    def should_trigger_retrieval(
        self,
        current_observation: str,
        target_observation: str,
        similarity_threshold: float = 0.95,  # Default: 95% similarity
        current_url: str = "",
        target_url: str = "",
    ) -> bool:
        """
        Determine if retrieval should be triggered based on state similarity.

        Used by pseudo_simulate_till_closest_tk strategy to decide when to start retrieval.

        Args:
            current_observation: Current observation from environment
            target_observation: Target observation from step k in simulation data
            similarity_threshold: Minimum similarity to trigger retrieval (default: 0.95 = 95%)
            current_url: Current URL (empty string for non-web environments)
            target_url: Target URL from step k (empty string means don't use URL for matching)

        Returns:
            True if retrieval should be triggered, False otherwise
        """
        similarity = self.compute_observation_similarity(
            current_observation, target_observation
        )
        return similarity >= similarity_threshold
