"""Tests for DiscreteTransformer class."""

from gnomics import BitArray, DiscreteTransformer


class TestDiscreteTransformerConstruction:
    """Test DiscreteTransformer construction."""

    def test_new(self) -> None:
        """Test creating a new DiscreteTransformer."""
        transformer = DiscreteTransformer(num_v=10, num_s=1024, num_t=2, seed=42)
        assert transformer is not None
        assert transformer.num_v() == 10
        assert transformer.num_s() == 1024


class TestDiscreteTransformerValueManagement:
    """Test DiscreteTransformer value management."""

    def test_set_get_value(self) -> None:
        """Test setting and getting category values."""
        transformer = DiscreteTransformer(num_v=10, num_s=1024, num_t=2, seed=42)
        transformer.set_value(5)
        assert transformer.get_value() == 5

    def test_boundary_values(self) -> None:
        """Test setting boundary category values."""
        transformer = DiscreteTransformer(num_v=10, num_s=1024, num_t=2, seed=42)

        # Test minimum category
        transformer.set_value(0)
        assert transformer.get_value() == 0

        # Test maximum category
        transformer.set_value(9)
        assert transformer.get_value() == 9


class TestDiscreteTransformerEncoding:
    """Test DiscreteTransformer encoding."""

    def test_execute(self) -> None:
        """Test executing the computation pipeline."""
        transformer = DiscreteTransformer(num_v=10, num_s=1024, num_t=2, seed=42)
        transformer.set_value(5)
        transformer.execute(False)
        output = transformer.output.state()
        assert isinstance(output, BitArray)
        assert len(output) == 1024

    def test_output_sparsity(self) -> None:
        """Test that output has expected sparsity."""
        transformer = DiscreteTransformer(num_v=10, num_s=1024, num_t=2, seed=42)
        transformer.set_value(5)
        transformer.execute(False)
        output = transformer.output.state()
        # Should have exactly num_as active bits
        num_active = output.num_set()
        assert num_active == transformer.num_as()
        assert num_active > 0

    def test_unique_encodings(self) -> None:
        """Test that different categories have zero overlap."""
        transformer = DiscreteTransformer(num_v=10, num_s=1024, num_t=2, seed=42)

        # Encode category 0
        transformer.set_value(0)
        transformer.execute(False)
        output0 = transformer.output.state()

        # Encode category 5
        transformer.set_value(5)
        transformer.execute(False)
        output5 = transformer.output.state()

        # Different categories should have zero overlap
        overlap = output0.num_similar(output5)
        assert overlap == 0

    def test_consistent_encoding(self) -> None:
        """Test that same category produces same encoding."""
        transformer = DiscreteTransformer(num_v=10, num_s=1024, num_t=2, seed=42)

        # Encode category 3 twice
        transformer.set_value(3)
        transformer.execute(False)
        output1 = transformer.output.state()

        transformer.set_value(3)
        transformer.execute(False)
        output2 = transformer.output.state()

        # Same category should produce identical encoding
        assert output1.get_bits() == output2.get_bits()


class TestDiscreteTransformerHistory:
    """Test DiscreteTransformer history tracking."""

    def test_output_at(self) -> None:
        """Test accessing historical outputs."""
        transformer = DiscreteTransformer(num_v=10, num_s=1024, num_t=3, seed=42)
        transformer.set_value(5)
        transformer.execute(False)

        # Access current output
        current = transformer.output_at(0)
        assert isinstance(current, BitArray)

        # Access previous output
        previous = transformer.output_at(1)
        assert isinstance(previous, BitArray)

    def test_has_changed(self) -> None:
        """Test change detection."""
        transformer = DiscreteTransformer(num_v=10, num_s=1024, num_t=2, seed=42)

        # First execution
        transformer.set_value(3)
        transformer.execute(False)

        # Second execution with different category
        transformer.set_value(7)
        transformer.execute(False)

        # Output should have changed
        assert transformer.has_changed()

    def test_no_change_same_category(self) -> None:
        """Test that same category doesn't trigger change."""
        transformer = DiscreteTransformer(num_v=10, num_s=1024, num_t=2, seed=42)

        # First execution
        transformer.set_value(5)
        transformer.execute(False)

        # Second execution with same category
        transformer.set_value(5)
        transformer.execute(False)

        # Output should not have changed
        assert not transformer.has_changed()


class TestDiscreteTransformerOperations:
    """Test DiscreteTransformer operations."""

    def test_clear(self) -> None:
        """Test clearing transformer state."""
        transformer = DiscreteTransformer(num_v=10, num_s=1024, num_t=2, seed=42)
        transformer.set_value(5)
        transformer.execute(False)

        transformer.clear()

        # After clear, output should be empty
        output = transformer.output.state()
        assert output.num_set() == 0

    def test_memory_usage(self) -> None:
        """Test memory usage estimation."""
        transformer = DiscreteTransformer(num_v=10, num_s=1024, num_t=2, seed=42)
        usage = transformer.memory_usage()
        assert isinstance(usage, int)
        assert usage > 0


class TestDiscreteTransformerRepr:
    """Test DiscreteTransformer representation."""

    def test_repr(self) -> None:
        """Test __repr__."""
        transformer = DiscreteTransformer(num_v=10, num_s=1024, num_t=2, seed=42)
        repr_str = repr(transformer)
        assert "DiscreteTransformer" in repr_str
        assert "num_v=10" in repr_str
        assert "num_s=1024" in repr_str


class TestDiscreteTransformerIntegration:
    """Integration tests for DiscreteTransformer."""

    def test_all_categories_unique(self) -> None:
        """Test that all categories have unique non-overlapping encodings."""
        num_categories = 10
        transformer = DiscreteTransformer(num_v=num_categories, num_s=1024, num_t=2, seed=42)

        encodings = []
        for category in range(num_categories):
            transformer.set_value(category)
            transformer.execute(False)
            encodings.append(transformer.output.state())

        # Check all pairs for zero overlap
        for i in range(num_categories):
            for j in range(i + 1, num_categories):
                overlap = encodings[i].num_similar(encodings[j])
                assert overlap == 0, f"Categories {i} and {j} have overlap: {overlap}"

    def test_categorical_workflow(self) -> None:
        """Test a complete workflow encoding categorical values."""
        transformer = DiscreteTransformer(num_v=5, num_s=512, num_t=2, seed=42)

        categories = [0, 1, 2, 3, 4, 0, 2, 4]
        outputs = []

        for category in categories:
            transformer.set_value(category)
            transformer.execute(False)
            outputs.append(transformer.output.state())

        # All outputs should be sparse with same sparsity
        sparsity = outputs[0].num_set()
        assert all(output.num_set() == sparsity for output in outputs)

        # Same categories should produce identical encodings
        assert outputs[0].get_bits() == outputs[5].get_bits()  # Both category 0
        assert outputs[2].get_bits() == outputs[6].get_bits()  # Both category 2
        assert outputs[4].get_bits() == outputs[7].get_bits()  # Both category 4
