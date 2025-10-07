"""Pythonic API layer for GCF components.

This module provides convenience functions for creating GCF components
with sensible defaults.

Note: PyO3 classes cannot be subclassed, so the base classes are re-exported
directly from the Rust module. Use the factory functions for convenience.
"""

from gnomics._core import (
    BitArray,
    BlockMemory,
    BlockOutput,
    ContextLearner as _ContextLearner,
    DiscreteTransformer as _DiscreteTransformer,
    PatternClassifier as _PatternClassifier,
    PatternPooler as _PatternPooler,
    PersistenceTransformer as _PersistenceTransformer,
    ScalarTransformer as _ScalarTransformer,
)


# Type aliases for clarity
ScalarTransformer = _ScalarTransformer
DiscreteTransformer = _DiscreteTransformer
PersistenceTransformer = _PersistenceTransformer
PatternPooler = _PatternPooler
PatternClassifier = _PatternClassifier
ContextLearner = _ContextLearner


# Convenience factory functions
def create_scalar_encoder(
    min_value: float,
    max_value: float,
    num_segments: int = 10,
    active_per_segment: int = 5,
    num_history: int = 2,
    seed: int = 42,
) -> ScalarTransformer:
    """Create a ScalarTransformer with sensible defaults.

    Args:
        min_value: Minimum value in the range
        max_value: Maximum value in the range
        num_segments: Number of overlapping segments (default: 10)
        active_per_segment: Active bits per segment (default: 5)
        num_history: History depth (default: 2)
        seed: Random seed (default: 42)

    Returns:
        Configured ScalarTransformer
    """
    return ScalarTransformer(
        min_value,
        max_value,
        num_segments,
        active_per_segment,
        num_history,
        seed,
    )


def create_category_encoder(
    num_categories: int,
    num_segments: int = 10,
    num_history: int = 2,
    seed: int = 42,
) -> DiscreteTransformer:
    """Create a DiscreteTransformer with sensible defaults.

    Args:
        num_categories: Number of distinct categories
        num_segments: Number of segments (default: 10)
        num_history: History depth (default: 2)
        seed: Random seed (default: 42)

    Returns:
        Configured DiscreteTransformer
    """
    return DiscreteTransformer(
        num_categories,
        num_segments,
        num_history,
        seed,
    )


def create_pooler(
    num_statelets: int,
    active_statelets: int,
    permanence_threshold: int = 20,
    permanence_increment: int = 2,
    permanence_decrement: int = 1,
    pooling_pct: float = 0.8,
    connectivity_pct: float = 0.5,
    learning_pct: float = 0.3,
    num_history: int = 2,
    seed: int = 42,
) -> PatternPooler:
    """Create a PatternPooler with sensible defaults.

    Args:
        num_statelets: Total number of statelets
        active_statelets: Target number of active statelets
        permanence_threshold: Synaptic permanence threshold (default: 20)
        permanence_increment: Learning increment (default: 2)
        permanence_decrement: Punishment decrement (default: 1)
        pooling_pct: Percentage of statelets in pooling (default: 0.8)
        connectivity_pct: Initial connectivity percentage (default: 0.5)
        learning_pct: Percentage of receptors that learn (default: 0.3)
        num_history: History depth (default: 2)
        seed: Random seed (default: 42)

    Returns:
        Configured PatternPooler
    """
    return PatternPooler(
        num_statelets,
        active_statelets,
        permanence_threshold,
        permanence_increment,
        permanence_decrement,
        pooling_pct,
        connectivity_pct,
        learning_pct,
        num_history,
        seed,
    )


def create_classifier(
    num_labels: int,
    num_statelets: int,
    active_statelets: int,
    permanence_threshold: int = 20,
    permanence_increment: int = 2,
    permanence_decrement: int = 1,
    pooling_pct: float = 0.8,
    connectivity_pct: float = 0.5,
    learning_pct: float = 0.3,
    num_history: int = 2,
    seed: int = 42,
) -> PatternClassifier:
    """Create a PatternClassifier with sensible defaults.

    Args:
        num_labels: Number of classification labels
        num_statelets: Total number of statelets
        active_statelets: Target number of active statelets
        permanence_threshold: Synaptic permanence threshold (default: 20)
        permanence_increment: Learning increment (default: 2)
        permanence_decrement: Punishment decrement (default: 1)
        pooling_pct: Percentage of statelets in pooling (default: 0.8)
        connectivity_pct: Initial connectivity percentage (default: 0.5)
        learning_pct: Percentage of receptors that learn (default: 0.3)
        num_history: History depth (default: 2)
        seed: Random seed (default: 42)

    Returns:
        Configured PatternClassifier
    """
    return PatternClassifier(
        num_labels,
        num_statelets,
        active_statelets,
        permanence_threshold,
        permanence_increment,
        permanence_decrement,
        pooling_pct,
        connectivity_pct,
        learning_pct,
        num_history,
        seed,
    )


def create_temporal_learner(
    num_columns: int,
    statelets_per_column: int = 8,
    dendrites_per_statelet: int = 4,
    receptors_per_dendrite: int = 20,
    dendrite_threshold: int = 15,
    permanence_threshold: int = 20,
    permanence_increment: int = 2,
    permanence_decrement: int = 1,
    num_history: int = 3,
    seed: int = 42,
) -> ContextLearner:
    """Create a ContextLearner with sensible defaults.

    Args:
        num_columns: Number of columns (matches input size)
        statelets_per_column: Statelets per column (default: 8)
        dendrites_per_statelet: Dendrites per statelet (default: 4)
        receptors_per_dendrite: Receptors per dendrite (default: 20)
        dendrite_threshold: Dendrite activation threshold (default: 15)
        permanence_threshold: Synaptic permanence threshold (default: 20)
        permanence_increment: Learning increment (default: 2)
        permanence_decrement: Punishment decrement (default: 1)
        num_history: History depth (default: 3)
        seed: Random seed (default: 42)

    Returns:
        Configured ContextLearner
    """
    return ContextLearner(
        num_columns,
        statelets_per_column,
        dendrites_per_statelet,
        receptors_per_dendrite,
        dendrite_threshold,
        permanence_threshold,
        permanence_increment,
        permanence_decrement,
        num_history,
        seed,
    )


__all__ = [
    "BitArray",
    "BlockOutput",
    "BlockMemory",
    "ScalarTransformer",
    "DiscreteTransformer",
    "PersistenceTransformer",
    "PatternPooler",
    "PatternClassifier",
    "ContextLearner",
    "create_scalar_encoder",
    "create_category_encoder",
    "create_pooler",
    "create_classifier",
    "create_temporal_learner",
]
