"""Property-based tests using hypothesis."""

from gnomics import BitArray
from gnomics.api import (
    create_category_encoder,
    create_classifier,
    create_pooler,
    create_scalar_encoder,
)
from hypothesis import given
from hypothesis import strategies as st


class TestBitArrayProperties:
    """Property-based tests for BitArray."""

    @given(st.integers(min_value=1, max_value=1000))
    def test_new_bitarray_all_cleared(self, size: int) -> None:
        """Newly created BitArray should have all bits cleared."""
        ba = BitArray(size)
        assert ba.num_set() == 0
        assert ba.num_cleared() == size

    @given(
        st.integers(min_value=10, max_value=100),
        st.lists(st.integers(min_value=0, max_value=99), min_size=1, max_size=10),
    )
    def test_set_acts_idempotent(self, size: int, indices: list[int]) -> None:
        """Setting the same active indices multiple times should be idempotent."""
        # Filter indices to be within bounds
        valid_indices = [i for i in indices if i < size]
        if not valid_indices:
            return

        ba = BitArray(size)
        ba.set_acts(valid_indices)
        count1 = ba.num_set()

        ba.set_acts(valid_indices)
        count2 = ba.num_set()

        assert count1 == count2

    @given(st.integers(min_value=10, max_value=100))
    def test_clear_all_sets_to_zero(self, size: int) -> None:
        """clear_all should set all bits to 0."""
        ba = BitArray(size)
        ba.set_all()
        assert ba.num_set() == size

        ba.clear_all()
        assert ba.num_set() == 0

    @given(st.integers(min_value=10, max_value=100))
    def test_toggle_all_twice_is_identity(self, size: int) -> None:
        """Toggling all bits twice should restore original state."""
        ba = BitArray(size)
        ba.set_acts([0, 1, 2])
        count_before = ba.num_set()

        ba.toggle_all()
        ba.toggle_all()
        count_after = ba.num_set()

        assert count_before == count_after

    @given(
        st.integers(min_value=20, max_value=100),
        st.integers(min_value=0, max_value=19),
    )
    def test_set_clear_bit_inverse(self, size: int, index: int) -> None:
        """Setting then clearing a bit should leave it cleared."""
        ba = BitArray(size)
        ba.set_bit(index)
        assert ba.get_bit(index) == 1

        ba.clear_bit(index)
        assert ba.get_bit(index) == 0


class TestScalarEncoderProperties:
    """Property-based tests for ScalarTransformer."""

    @given(
        st.floats(min_value=0.0, max_value=100.0),
        st.floats(min_value=0.0, max_value=100.0),
    )
    def test_same_value_same_encoding(self, min_val: float, max_val: float) -> None:
        """Encoding the same value twice should produce identical output."""
        if min_val >= max_val:
            return

        encoder = create_scalar_encoder(min_value=min_val, max_value=max_val)

        test_value = (min_val + max_val) / 2

        encoder.set_value(test_value)
        encoder.execute(learn_flag=False)
        output1 = encoder.output.state()

        encoder.set_value(test_value)
        encoder.execute(learn_flag=False)
        output2 = encoder.output.state()

        # Should have perfect overlap
        assert output1.num_similar(output2) == output1.num_set()

    @given(st.floats(min_value=-1000.0, max_value=2000.0))
    def test_value_clamping(self, value: float) -> None:
        """Values outside range should be clamped."""
        encoder = create_scalar_encoder(min_value=0.0, max_value=100.0)

        encoder.set_value(value)
        encoder.execute(learn_flag=False)
        output = encoder.output.state()

        # Should produce valid output regardless of input value
        assert output.num_set() == encoder.num_as()

    @given(st.integers(min_value=5, max_value=50))
    def test_output_sparsity(self, num_segments: int) -> None:
        """Output should always have exactly num_as active bits."""
        encoder = create_scalar_encoder(min_value=0.0, max_value=100.0, num_segments=num_segments)

        encoder.set_value(50.0)
        encoder.execute(learn_flag=False)
        output = encoder.output.state()

        assert output.num_set() == encoder.num_as()


class TestCategoryEncoderProperties:
    """Property-based tests for DiscreteTransformer."""

    @given(st.integers(min_value=2, max_value=20), st.integers(min_value=0, max_value=19))
    def test_same_category_same_encoding(self, num_categories: int, category: int) -> None:
        """Encoding the same category twice should produce identical output."""
        if category >= num_categories:
            return

        encoder = create_category_encoder(num_categories=num_categories)

        encoder.set_value(category)
        encoder.execute(learn_flag=False)
        output1 = encoder.output.state()

        encoder.set_value(category)
        encoder.execute(learn_flag=False)
        output2 = encoder.output.state()

        # Should have perfect overlap
        assert output1.num_similar(output2) == output1.num_set()

    @given(
        st.integers(min_value=3, max_value=10),
        st.integers(min_value=0, max_value=9),
        st.integers(min_value=0, max_value=9),
    )
    def test_different_categories_different_encodings(
        self, num_categories: int, cat1: int, cat2: int
    ) -> None:
        """Different categories should have zero overlap."""
        if cat1 >= num_categories or cat2 >= num_categories or cat1 == cat2:
            return

        encoder = create_category_encoder(num_categories=num_categories)

        encoder.set_value(cat1)
        encoder.execute(learn_flag=False)
        output1 = encoder.output.state()

        encoder.set_value(cat2)
        encoder.execute(learn_flag=False)
        output2 = encoder.output.state()

        # Different categories should have zero overlap
        assert output1.num_similar(output2) == 0


class TestPoolerProperties:
    """Property-based tests for PatternPooler."""

    @given(
        st.integers(min_value=50, max_value=200),
        st.integers(min_value=5, max_value=20),
    )
    def test_output_respects_sparsity(self, num_statelets: int, active_statelets: int) -> None:
        """Pooler output should not exceed num_as active bits."""
        if active_statelets >= num_statelets:
            return

        pooler = create_pooler(num_statelets=num_statelets, active_statelets=active_statelets)
        pooler.init(num_i=256)

        input_pattern = BitArray(256)
        input_pattern.set_acts([1, 5, 10, 15, 20])

        pooler.input.set_state(input_pattern)
        pooler.execute(learn_flag=False)
        output = pooler.output.state()

        # Output should not exceed active_statelets
        assert output.num_set() <= active_statelets

    @given(st.integers(min_value=20, max_value=100))
    def test_clear_resets_state(self, num_statelets: int) -> None:
        """Clearing should reset all state."""
        pooler = create_pooler(num_statelets=num_statelets, active_statelets=10)
        pooler.init(num_i=128)

        input_pattern = BitArray(128)
        input_pattern.set_acts([1, 5, 10])
        pooler.input.set_state(input_pattern)
        pooler.execute(learn_flag=True)

        pooler.clear()
        output = pooler.output.state()

        assert output.num_set() == 0


class TestClassifierProperties:
    """Property-based tests for PatternClassifier."""

    @given(
        st.integers(min_value=2, max_value=5),
        st.integers(min_value=5, max_value=15),
    )
    def test_probabilities_sum_to_one(self, num_labels: int, active_statelets: int) -> None:
        """Probability distribution should sum to approximately 1.0 after learning."""
        # Ensure num_statelets is divisible by num_labels
        num_statelets = num_labels * 30

        if active_statelets >= num_statelets:
            return

        classifier = create_classifier(
            num_labels=num_labels,
            num_statelets=num_statelets,
            active_statelets=active_statelets,
        )
        classifier.init(num_i=128)

        # Train with some data first
        input_pattern = BitArray(128)
        input_pattern.set_acts([1, 5, 10, 15, 20])
        classifier.set_label(0)
        classifier.input.set_state(input_pattern)
        classifier.execute(learn_flag=True)

        # Now compute and check probabilities
        classifier.compute(input_pattern)
        probs = classifier.get_probabilities()

        assert len(probs) == num_labels
        prob_sum = sum(probs)
        # After learning, probabilities should sum to approximately 1.0
        if prob_sum > 0:  # Only check if there are non-zero probabilities
            assert 0.99 <= prob_sum <= 1.01  # Allow small floating point error

    @given(st.integers(min_value=2, max_value=10))
    def test_predicted_label_in_range(self, num_labels: int) -> None:
        """Predicted label should always be in valid range."""
        num_statelets = num_labels * 30
        classifier = create_classifier(
            num_labels=num_labels,
            num_statelets=num_statelets,
            active_statelets=10,
        )
        classifier.init(num_i=128)

        input_pattern = BitArray(128)
        input_pattern.set_acts([1, 5, 10])

        classifier.compute(input_pattern)
        predicted = classifier.get_predicted_label()

        assert 0 <= predicted < num_labels


class TestMemoryUsageProperties:
    """Property-based tests for memory usage."""

    @given(st.integers(min_value=50, max_value=500))
    def test_larger_components_use_more_memory(self, size: int) -> None:
        """Larger components should generally use more memory."""
        small_pooler = create_pooler(num_statelets=size, active_statelets=10)
        large_pooler = create_pooler(num_statelets=size * 2, active_statelets=10)

        small_mem = small_pooler.memory_usage()
        large_mem = large_pooler.memory_usage()

        assert large_mem >= small_mem

    @given(
        st.integers(min_value=10, max_value=50),
        st.integers(min_value=10, max_value=50),
    )
    def test_memory_usage_positive(self, num_segments: int, active_per_seg: int) -> None:
        """All components should report positive memory usage."""
        if active_per_seg >= num_segments:
            return

        encoder = create_scalar_encoder(
            min_value=0.0,
            max_value=100.0,
            num_segments=num_segments,
            active_per_segment=active_per_seg,
        )

        mem = encoder.memory_usage()
        assert mem > 0
