"""Type stubs for the Rust extension module."""

from collections.abc import Iterator

__version__: str

class BitArray:
    """A high-performance binary array for sparse distributed representations."""

    def __init__(self, n: int) -> None:
        """Create a new BitArray with n bits, all initialized to 0."""
        ...

    @staticmethod
    def from_bits(bits: list[int]) -> BitArray:
        """Create a BitArray from a list of bit values (0s and 1s)."""
        ...

    @staticmethod
    def from_indices(size: int, indices: list[int]) -> BitArray:
        """Create a BitArray from a list of active bit indices."""
        ...

    @staticmethod
    def from_numpy(arr: object) -> BitArray:
        """Create a BitArray from a numpy array."""
        ...

    def resize(self, n: int) -> None:
        """Resize the BitArray to n bits."""
        ...

    def erase(self) -> None:
        """Clear all storage and reset size to 0."""
        ...

    def set_bit(self, b: int) -> None:
        """Set a single bit to 1."""
        ...

    def get_bit(self, b: int) -> int:
        """Get the value of a single bit."""
        ...

    def clear_bit(self, b: int) -> None:
        """Clear a single bit to 0."""
        ...

    def toggle_bit(self, b: int) -> None:
        """Toggle a single bit (0 -> 1, 1 -> 0)."""
        ...

    def assign_bit(self, b: int, val: int) -> None:
        """Assign a value to a single bit."""
        ...

    def set_range(self, beg: int, len: int) -> None:
        """Set a range of bits to 1."""
        ...

    def clear_range(self, beg: int, len: int) -> None:
        """Clear a range of bits to 0."""
        ...

    def toggle_range(self, beg: int, len: int) -> None:
        """Toggle a range of bits."""
        ...

    def set_all(self) -> None:
        """Set all bits to 1."""
        ...

    def clear_all(self) -> None:
        """Clear all bits to 0."""
        ...

    def toggle_all(self) -> None:
        """Toggle all bits."""
        ...

    def set_bits(self, vals: list[int]) -> None:
        """Set bits from a list of values."""
        ...

    def set_acts(self, idxs: list[int]) -> None:
        """Set bits at specified indices to 1."""
        ...

    def get_bits(self) -> list[int]:
        """Get all bit values as a list."""
        ...

    def get_acts(self) -> list[int]:
        """Get indices of all bits that are set to 1."""
        ...

    def num_set(self) -> int:
        """Count the number of bits set to 1."""
        ...

    def num_cleared(self) -> int:
        """Count the number of bits set to 0."""
        ...

    def num_similar(self, other: BitArray) -> int:
        """Count the number of bits with the same value in both BitArrays."""
        ...

    def find_next_set_bit(self, beg: int) -> int | None:
        """Find the next set bit starting from a given position."""
        ...

    def to_numpy(self) -> object:
        """Convert the BitArray to a numpy array."""
        ...

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> int: ...
    def __iter__(self) -> Iterator[int]: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __and__(self, other: BitArray) -> BitArray: ...
    def __or__(self, other: BitArray) -> BitArray: ...
    def __xor__(self, other: BitArray) -> BitArray: ...
    def __invert__(self) -> BitArray: ...

class BlockOutput:
    """Manages block outputs with history tracking and change detection.

    BlockOutput maintains a circular buffer of historical states and efficiently
    tracks when outputs change, enabling optimizations in downstream blocks.
    """

    def __init__(self) -> None:
        """Create a new BlockOutput.

        The output must be set up with specific dimensions before use.
        """
        ...

    def setup(self, num_t: int, num_b: int) -> None:
        """Set up the BlockOutput with specific dimensions.

        Args:
            num_t: Number of historical time steps to maintain (minimum 2)
            num_b: Number of bits in the output state
        """
        ...

    def step(self) -> None:
        """Advance to the next time step in the circular buffer."""
        ...

    def store(self) -> None:
        """Store the current state and detect if it has changed.

        This compares the current state with the previous state to set
        the internal change flag.
        """
        ...

    def has_changed(self) -> bool:
        """Check if the output has changed in the current time step.

        Returns:
            True if the current state differs from the previous state
        """
        ...

    def has_changed_at(self, time: int) -> bool:
        """Check if the output changed at a specific time offset.

        Args:
            time: Time offset (0 = current, 1 = previous, etc.)

        Returns:
            True if the state changed at that time
        """
        ...

    def state(self) -> BitArray:
        """Get the current output state.

        Returns:
            BitArray containing the current state
        """
        ...

    def set_state(self, bits: BitArray) -> None:
        """Set the current output state from a BitArray.

        Args:
            bits: BitArray to set as the current state
        """
        ...

    def get_bitarray(self, time: int) -> BitArray:
        """Get the output state at a specific time offset.

        Args:
            time: Time offset (0 = current, 1 = previous, etc.)

        Returns:
            BitArray containing the state at that time
        """
        ...

    def num_t(self) -> int:
        """Get the number of time steps in history.

        Returns:
            Number of time steps
        """
        ...

    def id(self) -> int:
        """Get the unique output ID.

        Returns:
            Output ID
        """
        ...

    def clear(self) -> None:
        """Clear all state to zeros."""
        ...

    def __repr__(self) -> str: ...

class BlockMemory:
    """Synaptic learning with permanence-based connections.

    BlockMemory implements biologically-inspired learning through "dendrites"
    with multiple "receptors" that have adjustable permanence values (0-99).
    Learning strengthens matching connections and weakens non-matching ones.
    """

    def __init__(
        self,
        num_d: int,
        num_rpd: int,
        perm_thr: int,
        perm_inc: int,
        perm_dec: int,
        pct_learn: float,
    ) -> None:
        """Create a new BlockMemory with learning parameters.

        Args:
            num_d: Number of dendrites
            num_rpd: Number of receptors per dendrite
            perm_thr: Permanence threshold for "connected" state (0-99)
            perm_inc: Permanence increment on learning (typically 1-5)
            perm_dec: Permanence decrement on unlearning (typically 1-2)
            pct_learn: Percentage of receptors to learn/move (0.0-1.0)
        """
        ...

    def init(self, num_i: int, seed: int) -> None:
        """Initialize the memory with a specific input size.

        Must be called before using overlap, learn, or other methods.

        Args:
            num_i: Number of input bits
            seed: Random seed for initialization
        """
        ...

    def overlap(self, d: int, input: BitArray) -> int:
        """Calculate overlap score for a specific dendrite.

        Overlap is the count of connected receptors (permanence >= threshold)
        that match active bits in the input.

        Args:
            d: Dendrite index
            input: Input BitArray pattern

        Returns:
            Overlap score
        """
        ...

    def learn(self, d: int, input: BitArray, seed: int) -> None:
        """Learn from an input pattern for a specific dendrite.

        Strengthens connections to active input bits and weakens connections
        to inactive bits. Requires an RNG for the learn_move operation.

        Args:
            d: Dendrite index
            input: Input BitArray pattern
            seed: Random seed for any receptor movement
        """
        ...

    def punish(self, d: int, input: BitArray, seed: int) -> None:
        """Punish (weaken) connections for a specific dendrite.

        Decrements permanence for receptors that match the input pattern,
        implementing negative learning.

        Args:
            d: Dendrite index
            input: Input BitArray pattern
            seed: Random seed for any receptor movement
        """
        ...

    def learn_move(self, d: int, input: BitArray, seed: int) -> None:
        """Move "dead" receptors (zero permanence) to new random positions.

        This helps maintain dendrite coverage across the input space by
        relocating receptors that have been completely unlearned.

        Args:
            d: Dendrite index
            input: Input BitArray pattern (used to determine connected receptors)
            seed: Random seed for receptor placement
        """
        ...

    def num_dendrites(self) -> int:
        """Get the number of dendrites.

        Returns:
            Number of dendrites
        """
        ...

    def addrs(self, d: int) -> list[int]:
        """Get receptor addresses for a specific dendrite.

        Args:
            d: Index of the dendrite

        Returns:
            List of input bit indices that receptors connect to
        """
        ...

    def perms(self, d: int) -> list[int]:
        """Get receptor permanence values for a specific dendrite.

        Args:
            d: Index of the dendrite

        Returns:
            List of permanence values (0-99)
        """
        ...

    def conns(self, d: int) -> BitArray | None:
        """Get connected receptors as a BitArray for a specific dendrite.

        Args:
            d: Index of the dendrite

        Returns:
            BitArray indicating which receptors are connected (permanence >= threshold),
            or None if not computed yet
        """
        ...

    def __repr__(self) -> str: ...

class ScalarTransformer:
    """Encodes continuous scalar values into sparse distributed representations.

    ScalarTransformer converts continuous values into binary patterns where
    semantically similar values have overlapping bit patterns. This enables
    downstream blocks to recognize similar values and generalize across
    continuous ranges.
    """

    def __init__(
        self, min_val: float, max_val: float, num_s: int, num_as: int, num_t: int, seed: int
    ) -> None:
        """Create a new ScalarTransformer.

        Args:
            min_val: Minimum input value (inclusive)
            max_val: Maximum input value (inclusive)
            num_s: Total number of statelets (output bits)
            num_as: Number of active statelets in the encoding
            num_t: History depth (minimum 2)
            seed: Random seed (currently unused, reserved for future use)
        """
        ...

    def set_value(self, value: float) -> None:
        """Set the scalar value to encode.

        Values are automatically clamped to [min_val, max_val] range.

        Args:
            value: Scalar value to encode
        """
        ...

    def get_value(self) -> float:
        """Get the current scalar value.

        Returns:
            Current value
        """
        ...

    def execute(self, learn_flag: bool) -> None:
        """Execute the computation pipeline.

        This performs: pull() -> compute() -> store() -> [learn()] -> step()

        Args:
            learn_flag: Whether to perform learning (no-op for transformers)
        """
        ...

    def output(self) -> BitArray:
        """Get the output BitArray containing the encoded representation.

        Returns:
            BitArray with sparse binary encoding of the scalar value
        """
        ...

    def output_at(self, time: int) -> BitArray:
        """Get the output state at a specific time offset.

        Args:
            time: Time offset (0 = current, 1 = previous, etc.)

        Returns:
            BitArray containing the state at that time
        """
        ...

    def has_changed(self) -> bool:
        """Check if the output has changed from the previous time step.

        Returns:
            True if the output changed
        """
        ...

    def clear(self) -> None:
        """Clear all internal state and reset the transformer."""
        ...

    def min_val(self) -> float:
        """Get the minimum value bound.

        Returns:
            Minimum value
        """
        ...

    def max_val(self) -> float:
        """Get the maximum value bound.

        Returns:
            Maximum value
        """
        ...

    def num_s(self) -> int:
        """Get the total number of statelets (output bits).

        Returns:
            Number of statelets
        """
        ...

    def num_as(self) -> int:
        """Get the number of active statelets.

        Returns:
            Number of active statelets
        """
        ...

    def memory_usage(self) -> int:
        """Get estimated memory usage in bytes.

        Returns:
            Estimated memory usage
        """
        ...

    def __repr__(self) -> str: ...

class DiscreteTransformer:
    """Encodes categorical/discrete values into unique sparse distributed representations.

    DiscreteTransformer converts discrete categories into binary patterns with
    zero overlap between different categories. This ensures each category has
    a unique representation while maintaining the benefits of sparse encoding.
    """

    def __init__(self, num_v: int, num_s: int, num_t: int, seed: int) -> None:
        """Create a new DiscreteTransformer.

        Args:
            num_v: Number of discrete categories (values 0 to num_v-1)
            num_s: Total number of statelets (output bits)
            num_t: History depth (minimum 2)
            seed: Random seed (currently unused, reserved for future use)
        """
        ...

    def set_value(self, value: int) -> None:
        """Set the category value to encode.

        Args:
            value: Category index (must be in range 0 to num_v-1)
        """
        ...

    def get_value(self) -> int:
        """Get the current category value.

        Returns:
            Current category index
        """
        ...

    def execute(self, learn_flag: bool) -> None:
        """Execute the computation pipeline.

        This performs: pull() -> compute() -> store() -> [learn()] -> step()

        Args:
            learn_flag: Whether to perform learning (no-op for transformers)
        """
        ...

    def output(self) -> BitArray:
        """Get the output BitArray containing the encoded representation.

        Returns:
            BitArray with sparse binary encoding of the category
        """
        ...

    def output_at(self, time: int) -> BitArray:
        """Get the output state at a specific time offset.

        Args:
            time: Time offset (0 = current, 1 = previous, etc.)

        Returns:
            BitArray containing the state at that time
        """
        ...

    def has_changed(self) -> bool:
        """Check if the output has changed from the previous time step.

        Returns:
            True if the output changed
        """
        ...

    def clear(self) -> None:
        """Clear all internal state and reset the transformer."""
        ...

    def num_v(self) -> int:
        """Get the number of categories.

        Returns:
            Number of categories
        """
        ...

    def num_s(self) -> int:
        """Get the total number of statelets (output bits).

        Returns:
            Number of statelets
        """
        ...

    def num_as(self) -> int:
        """Get the number of active statelets per category.

        Returns:
            Number of active statelets
        """
        ...

    def memory_usage(self) -> int:
        """Get estimated memory usage in bytes.

        Returns:
            Estimated memory usage
        """
        ...

    def __repr__(self) -> str: ...

class PersistenceTransformer:
    """Encodes temporal persistence/stability of scalar values.

    PersistenceTransformer tracks how long a value remains relatively unchanged,
    encoding the persistence duration as a sparse distributed representation.
    This enables downstream blocks to recognize temporal patterns and stability.
    """

    def __init__(
        self,
        min_val: float,
        max_val: float,
        num_s: int,
        num_as: int,
        max_step: int,
        num_t: int,
        seed: int,
    ) -> None:
        """Create a new PersistenceTransformer.

        Args:
            min_val: Minimum input value (inclusive)
            max_val: Maximum input value (inclusive)
            num_s: Total number of statelets (output bits)
            num_as: Number of active statelets in the encoding
            max_step: Maximum persistence steps to track
            num_t: History depth (minimum 2)
            seed: Random seed (currently unused, reserved for future use)
        """
        ...

    def set_value(self, value: float) -> None:
        """Set the scalar value to track for persistence.

        Values are automatically clamped to [min_val, max_val] range.
        If the value changes by more than 10%, the persistence counter resets.

        Args:
            value: Scalar value to track
        """
        ...

    def get_value(self) -> float:
        """Get the current scalar value.

        Returns:
            Current value
        """
        ...

    def get_counter(self) -> int:
        """Get the current persistence counter.

        The counter increments when the value remains stable (change <= 10%)
        and resets to 0 when the value changes significantly.

        Returns:
            Current persistence counter (capped at max_step)
        """
        ...

    def execute(self, learn_flag: bool) -> None:
        """Execute the computation pipeline.

        This performs: pull() -> compute() -> store() -> [learn()] -> step()

        Args:
            learn_flag: Whether to perform learning (no-op for transformers)
        """
        ...

    def output(self) -> BitArray:
        """Get the output BitArray containing the encoded persistence duration.

        Returns:
            BitArray with sparse binary encoding of persistence
        """
        ...

    def output_at(self, time: int) -> BitArray:
        """Get the output state at a specific time offset.

        Args:
            time: Time offset (0 = current, 1 = previous, etc.)

        Returns:
            BitArray containing the state at that time
        """
        ...

    def has_changed(self) -> bool:
        """Check if the output has changed from the previous time step.

        Returns:
            True if the output changed
        """
        ...

    def clear(self) -> None:
        """Clear all internal state and reset the transformer.

        This resets the persistence counter to 0.
        """
        ...

    def max_step(self) -> int:
        """Get the maximum persistence step count.

        Returns:
            Maximum step count
        """
        ...

    def min_val(self) -> float:
        """Get the minimum value bound.

        Returns:
            Minimum value
        """
        ...

    def max_val(self) -> float:
        """Get the maximum value bound.

        Returns:
            Maximum value
        """
        ...

    def num_s(self) -> int:
        """Get the total number of statelets (output bits).

        Returns:
            Number of statelets
        """
        ...

    def num_as(self) -> int:
        """Get the number of active statelets.

        Returns:
            Number of active statelets
        """
        ...

    def memory_usage(self) -> int:
        """Get estimated memory usage in bytes.

        Returns:
            Estimated memory usage
        """
        ...

    def __repr__(self) -> str: ...

class PatternPooler:
    """Unsupervised learning block for pattern recognition.

    PatternPooler creates sparse distributed representations through competitive
    winner-take-all learning. It learns to recognize patterns in input data
    without supervision, making it useful for feature extraction and clustering.
    """

    def __init__(
        self,
        num_s: int,
        num_as: int,
        perm_thr: int,
        perm_inc: int,
        perm_dec: int,
        pct_pool: float,
        pct_conn: float,
        pct_learn: float,
        num_t: int,
        seed: int,
    ) -> None:
        """Create a new PatternPooler for unsupervised learning.

        Args:
            num_s: Number of statelets (dendrites)
            num_as: Number of active statelets in output
            perm_thr: Permanence threshold (0-99, typically 20)
            perm_inc: Permanence increment (typically 2)
            perm_dec: Permanence decrement (typically 1)
            pct_pool: Pooling percentage (typically 0.8)
            pct_conn: Initial connectivity (typically 0.5)
            pct_learn: Learning percentage (typically 0.3)
            num_t: History depth (minimum 2)
            seed: Random seed for reproducibility
        """
        ...

    def init(self, num_i: int) -> None:
        """Initialize the pooler with input size.

        Must be called before using compute or learn methods.

        Args:
            num_i: Number of input bits
        """
        ...

    def compute(self, input: BitArray) -> None:
        """Compute output from input pattern.

        Finds top overlapping dendrites and activates them.

        Args:
            input: Input BitArray pattern
        """
        ...

    def learn(self, input: BitArray) -> None:
        """Learn from current input pattern.

        Strengthens connections of winning dendrites to active input bits.

        Args:
            input: Input BitArray pattern
        """
        ...

    def execute(self, input: BitArray, learn_flag: bool) -> None:
        """Execute full pipeline: compute and optionally learn.

        Args:
            input: Input BitArray pattern
            learn_flag: Whether to perform learning
        """
        ...

    def output(self) -> BitArray:
        """Get the output BitArray containing active dendrites.

        Returns:
            BitArray with exactly num_as bits set
        """
        ...

    def output_at(self, time: int) -> BitArray:
        """Get the output state at a specific time offset.

        Args:
            time: Time offset (0 = current, 1 = previous, etc.)

        Returns:
            BitArray containing the state at that time
        """
        ...

    def has_changed(self) -> bool:
        """Check if the output has changed from the previous time step.

        Returns:
            True if the output changed
        """
        ...

    def clear(self) -> None:
        """Clear all internal state and reset the pooler."""
        ...

    def num_s(self) -> int:
        """Get the number of statelets (dendrites).

        Returns:
            Number of statelets
        """
        ...

    def num_as(self) -> int:
        """Get the number of active statelets.

        Returns:
            Number of active statelets
        """
        ...

    def perm_thr(self) -> int:
        """Get the permanence threshold.

        Returns:
            Permanence threshold (0-99)
        """
        ...

    def memory_usage(self) -> int:
        """Get estimated memory usage in bytes.

        Returns:
            Estimated memory usage
        """
        ...

    def __repr__(self) -> str: ...

class PatternClassifier:
    """Supervised learning block for multi-class classification.

    PatternClassifier learns to classify input patterns into discrete categories
    through supervised learning. It produces probability distributions over
    class labels and can predict the most likely label for new inputs.
    """

    def __init__(
        self,
        num_l: int,
        num_s: int,
        num_as: int,
        perm_thr: int,
        perm_inc: int,
        perm_dec: int,
        pct_pool: float,
        pct_conn: float,
        pct_learn: float,
        num_t: int,
        seed: int,
    ) -> None:
        """Create a new PatternClassifier for supervised classification.

        Args:
            num_l: Number of labels/classes
            num_s: Total number of statelets
            num_as: Active statelets per label
            perm_thr: Permanence threshold (0-99, typically 20)
            perm_inc: Permanence increment (typically 2)
            perm_dec: Permanence decrement (typically 1)
            pct_pool: Pooling percentage (typically 0.8)
            pct_conn: Initial connectivity (typically 0.5)
            pct_learn: Learning percentage (typically 0.3)
            num_t: History depth (minimum 2)
            seed: Random seed for reproducibility
        """
        ...

    def init(self, num_i: int) -> None:
        """Initialize the classifier with input size.

        Must be called before using compute or learn methods.

        Args:
            num_i: Number of input bits
        """
        ...

    def set_label(self, label: int) -> None:
        """Set the ground truth label for learning.

        Must be called before learn() to specify which label to train.

        Args:
            label: Label index (0 to num_l-1)
        """
        ...

    def compute(self, input: BitArray) -> None:
        """Compute output and probabilities from input pattern.

        Args:
            input: Input BitArray pattern
        """
        ...

    def learn(self, input: BitArray) -> None:
        """Learn from current input pattern using the set label.

        Must call set_label() first to specify ground truth.

        Args:
            input: Input BitArray pattern
        """
        ...

    def execute(self, input: BitArray, learn_flag: bool) -> None:
        """Execute full pipeline: compute and optionally learn.

        If learn_flag is True, must call set_label() first.

        Args:
            input: Input BitArray pattern
            learn_flag: Whether to perform learning
        """
        ...

    def get_probabilities(self) -> list[float]:
        """Get probability distribution over all labels.

        Returns:
            List of probabilities for each label (sums to 1.0)
        """
        ...

    def get_predicted_label(self) -> int:
        """Get the predicted label (highest probability).

        Returns:
            Label index with highest probability
        """
        ...

    def get_labels(self) -> list[int]:
        """Get all label indices.

        Returns:
            List of label indices [0, 1, 2, ..., num_l-1]
        """
        ...

    def output(self) -> BitArray:
        """Get the output BitArray containing active dendrites.

        Returns:
            BitArray with num_l * num_as bits set
        """
        ...

    def output_at(self, time: int) -> BitArray:
        """Get the output state at a specific time offset.

        Args:
            time: Time offset (0 = current, 1 = previous, etc.)

        Returns:
            BitArray containing the state at that time
        """
        ...

    def has_changed(self) -> bool:
        """Check if the output has changed from the previous time step.

        Returns:
            True if the output changed
        """
        ...

    def clear(self) -> None:
        """Clear all internal state and reset the classifier."""
        ...

    def num_l(self) -> int:
        """Get the number of labels.

        Returns:
            Number of labels
        """
        ...

    def num_s(self) -> int:
        """Get the number of statelets.

        Returns:
            Total number of statelets
        """
        ...

    def num_as(self) -> int:
        """Get the number of active statelets per label.

        Returns:
            Active statelets per label
        """
        ...

    def num_spl(self) -> int:
        """Get the number of statelets per label.

        Returns:
            Statelets per label (num_s / num_l)
        """
        ...

    def memory_usage(self) -> int:
        """Get estimated memory usage in bytes.

        Returns:
            Estimated memory usage
        """
        ...

    def __repr__(self) -> str: ...

class ContextLearner:
    """Context learning block for context-dependent pattern recognition.

    ContextLearner learns associations between inputs and contexts, enabling
    it to predict inputs based on context and detect anomalies when inputs
    appear in unexpected contexts. It's useful for sequence learning,
    contextual disambiguation, and anomaly detection.
    """

    def __init__(
        self,
        num_c: int,
        num_spc: int,
        num_dps: int,
        num_rpd: int,
        d_thresh: int,
        perm_thr: int,
        perm_inc: int,
        perm_dec: int,
        num_t: int,
        seed: int,
    ) -> None:
        """Create a new ContextLearner for contextual learning.

        Args:
            num_c: Number of columns (matches input size)
            num_spc: Statelets per column
            num_dps: Dendrites per statelet
            num_rpd: Receptors per dendrite
            d_thresh: Dendrite activation threshold
            perm_thr: Permanence threshold (0-99, typically 20)
            perm_inc: Permanence increment (typically 2)
            perm_dec: Permanence decrement (typically 1)
            num_t: History depth (minimum 2)
            seed: Random seed for reproducibility
        """
        ...

    def init(self, num_input_bits: int, num_context_bits: int) -> None:
        """Initialize the learner with input and context sizes.

        Must be called before using compute or learn methods.

        Args:
            num_input_bits: Number of input bits (columns)
            num_context_bits: Number of context bits
        """
        ...

    def compute(self, input: BitArray, context: BitArray) -> None:
        """Compute predictions from input and context patterns.

        Args:
            input: Input BitArray pattern (active columns)
            context: Context BitArray pattern
        """
        ...

    def learn(self, input: BitArray, context: BitArray) -> None:
        """Learn from current input and context patterns.

        Args:
            input: Input BitArray pattern (active columns)
            context: Context BitArray pattern
        """
        ...

    def execute(self, input: BitArray, context: BitArray, learn_flag: bool) -> None:
        """Execute full pipeline: compute and optionally learn.

        Args:
            input: Input BitArray pattern (active columns)
            context: Context BitArray pattern
            learn_flag: Whether to perform learning
        """
        ...

    def get_anomaly_score(self) -> float:
        """Get the anomaly score for the current prediction.

        Returns:
            Anomaly score from 0.0 (fully predicted) to 1.0 (completely surprising)
        """
        ...

    def get_historical_count(self) -> int:
        """Get the number of statelets with at least one used dendrite.

        Returns:
            Count of statelets that have learned patterns
        """
        ...

    def output(self) -> BitArray:
        """Get the output BitArray containing predicted/active statelets.

        Returns:
            BitArray with active statelets
        """
        ...

    def output_at(self, time: int) -> BitArray:
        """Get the output state at a specific time offset.

        Args:
            time: Time offset (0 = current, 1 = previous, etc.)

        Returns:
            BitArray containing the state at that time
        """
        ...

    def has_changed(self) -> bool:
        """Check if the output has changed from the previous time step.

        Returns:
            True if the output changed
        """
        ...

    def clear(self) -> None:
        """Clear all internal state and reset the learner."""
        ...

    def num_c(self) -> int:
        """Get the number of columns.

        Returns:
            Number of columns
        """
        ...

    def num_spc(self) -> int:
        """Get the number of statelets per column.

        Returns:
            Statelets per column
        """
        ...

    def num_dps(self) -> int:
        """Get the number of dendrites per statelet.

        Returns:
            Dendrites per statelet
        """
        ...

    def d_thresh(self) -> int:
        """Get the dendrite activation threshold.

        Returns:
            Dendrite threshold
        """
        ...

    def memory_usage(self) -> int:
        """Get estimated memory usage in bytes.

        Returns:
            Estimated memory usage
        """
        ...

    def __repr__(self) -> str: ...

class SequenceLearner:
    """Sequence learning block for temporal pattern recognition with self-feedback.

    SequenceLearner learns temporal sequences and predicts next patterns.
    It is nearly identical to ContextLearner but uses its own previous output
    as context, enabling it to learn temporal transitions automatically.
    """

    def __init__(
        self,
        num_c: int,
        num_spc: int,
        num_dps: int,
        num_rpd: int,
        d_thresh: int,
        perm_thr: int,
        perm_inc: int,
        perm_dec: int,
        num_t: int,
        always_update: bool = False,
        seed: int = 0,
    ) -> None:
        """Create a new SequenceLearner for temporal sequence learning.

        Args:
            num_c: Number of columns (matches input size)
            num_spc: Statelets per column
            num_dps: Dendrites per statelet
            num_rpd: Receptors per dendrite
            d_thresh: Dendrite activation threshold
            perm_thr: Permanence threshold (0-99, typically 20)
            perm_inc: Permanence increment (typically 2)
            perm_dec: Permanence decrement (typically 1)
            num_t: History depth (minimum 2)
            always_update: Update even when inputs unchanged (default: False)
            seed: Random seed for reproducibility (default: 0)
        """
        ...

    def init(self, num_input_bits: int) -> None:
        """Initialize the learner with input size.

        Must be called before using compute or learn methods.
        Context is automatically connected to previous output (self-feedback).

        Args:
            num_input_bits: Number of input bits (columns)
        """
        ...

    def compute(self, input: BitArray) -> None:
        """Compute predictions from input pattern.

        Uses previous output as context to predict current input.

        Args:
            input: Input BitArray pattern (active columns)
        """
        ...

    def learn(self) -> None:
        """Learn the temporal transition from previous to current pattern."""
        ...

    def execute(self, input: BitArray, learn_flag: bool) -> None:
        """Execute full pipeline: compute and optionally learn.

        Args:
            input: Input BitArray pattern (active columns)
            learn_flag: Whether to perform learning
        """
        ...

    def clear(self) -> None:
        """Clear all state and history."""
        ...

    def get_anomaly_score(self) -> float:
        """Get current anomaly score (0.0-1.0).

        Indicates the percentage of input columns that were unexpected.
        - 0.0 = All columns were predicted by previous pattern
        - 1.0 = All columns were unexpected (sequence broken)

        Returns:
            Anomaly score from 0.0 to 1.0
        """
        ...

    def get_historical_count(self) -> int:
        """Get count of statelets that have learned at least one transition.

        Indicates how many different temporal patterns have been learned.

        Returns:
            Count of statelets with learned patterns
        """
        ...

    def output(self) -> BitArray:
        """Get the current output BitArray.

        Returns the active statelets predicted for the current time step.

        Returns:
            BitArray with active statelets
        """
        ...

    def output_at(self, time: int) -> BitArray:
        """Get output at a specific historical time step.

        Args:
            time: Time offset (0=current, 1=previous, etc.)

        Returns:
            BitArray at the specified time step
        """
        ...

    def has_changed(self) -> bool:
        """Check if output has changed since last time step.

        Returns:
            True if output differs from previous time step
        """
        ...

    def num_c(self) -> int:
        """Get the number of columns.

        Returns:
            Number of columns
        """
        ...

    def num_spc(self) -> int:
        """Get statelets per column.

        Returns:
            Statelets per column
        """
        ...

    def num_dps(self) -> int:
        """Get dendrites per statelet.

        Returns:
            Dendrites per statelet
        """
        ...

    def d_thresh(self) -> int:
        """Get the dendrite activation threshold.

        Returns:
            Dendrite threshold
        """
        ...

    def memory_usage(self) -> int:
        """Get estimated memory usage in bytes.

        Returns:
            Estimated memory usage
        """
        ...

    def __repr__(self) -> str: ...
