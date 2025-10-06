"""Tests for ScalarTransformer class."""

from gcf_core_python_client import BitArray, ScalarTransformer


class TestScalarTransformerConstruction:
    """Test ScalarTransformer construction."""

    def test_new(self) -> None:
        """Test creating a new ScalarTransformer."""
        transformer = ScalarTransformer(
            min_val=0.0, max_val=100.0, num_s=1024, num_as=40, num_t=2, seed=42
        )
        assert transformer is not None
        assert transformer.min_val() == 0.0
        assert transformer.max_val() == 100.0
        assert transformer.num_s() == 1024
        assert transformer.num_as() == 40


class TestScalarTransformerValueManagement:
    """Test ScalarTransformer value management."""

    def test_set_get_value(self) -> None:
        """Test setting and getting scalar values."""
        transformer = ScalarTransformer(
            min_val=0.0, max_val=100.0, num_s=1024, num_as=40, num_t=2, seed=42
        )
        transformer.set_value(50.0)
        assert transformer.get_value() == 50.0

    def test_value_clamping_max(self) -> None:
        """Test that values are clamped to max_val."""
        transformer = ScalarTransformer(
            min_val=0.0, max_val=100.0, num_s=1024, num_as=40, num_t=2, seed=42
        )
        transformer.set_value(150.0)
        # Value should be clamped to max_val
        assert transformer.get_value() <= 100.0

    def test_value_clamping_min(self) -> None:
        """Test that values are clamped to min_val."""
        transformer = ScalarTransformer(
            min_val=0.0, max_val=100.0, num_s=1024, num_as=40, num_t=2, seed=42
        )
        transformer.set_value(-50.0)
        # Value should be clamped to min_val
        assert transformer.get_value() >= 0.0


class TestScalarTransformerEncoding:
    """Test ScalarTransformer encoding."""

    def test_execute(self) -> None:
        """Test executing the computation pipeline."""
        transformer = ScalarTransformer(
            min_val=0.0, max_val=100.0, num_s=1024, num_as=40, num_t=2, seed=42
        )
        transformer.set_value(50.0)
        transformer.execute(False)
        output = transformer.output()
        assert isinstance(output, BitArray)
        assert len(output) == 1024

    def test_output_sparsity(self) -> None:
        """Test that output has expected number of active bits."""
        transformer = ScalarTransformer(
            min_val=0.0, max_val=100.0, num_s=1024, num_as=40, num_t=2, seed=42
        )
        transformer.set_value(50.0)
        transformer.execute(False)
        output = transformer.output()
        # Should have exactly num_as active bits
        assert output.num_set() == 40

    def test_semantic_similarity(self) -> None:
        """Test that similar values produce overlapping encodings."""
        transformer = ScalarTransformer(
            min_val=0.0, max_val=100.0, num_s=1024, num_as=40, num_t=2, seed=42
        )

        # Encode similar values
        transformer.set_value(50.0)
        transformer.execute(False)
        output1 = transformer.output()

        transformer.set_value(51.0)
        transformer.execute(False)
        output2 = transformer.output()

        # Similar values should have significant overlap
        overlap = output1.num_similar(output2)
        assert overlap > 30  # Most bits should overlap for close values

    def test_different_values_less_overlap(self) -> None:
        """Test that different values have less overlap."""
        transformer = ScalarTransformer(
            min_val=0.0, max_val=100.0, num_s=1024, num_as=40, num_t=2, seed=42
        )

        # Encode very different values
        transformer.set_value(10.0)
        transformer.execute(False)
        output1 = transformer.output()

        transformer.set_value(90.0)
        transformer.execute(False)
        output2 = transformer.output()

        # Different values should have minimal overlap
        overlap = output1.num_similar(output2)
        assert overlap < 20  # Less overlap for distant values


class TestScalarTransformerHistory:
    """Test ScalarTransformer history tracking."""

    def test_output_at(self) -> None:
        """Test accessing historical outputs."""
        transformer = ScalarTransformer(
            min_val=0.0, max_val=100.0, num_s=1024, num_as=40, num_t=3, seed=42
        )
        transformer.set_value(50.0)
        transformer.execute(False)

        # Access current output
        current = transformer.output_at(0)
        assert isinstance(current, BitArray)

        # Access previous output
        previous = transformer.output_at(1)
        assert isinstance(previous, BitArray)

    def test_has_changed(self) -> None:
        """Test change detection."""
        transformer = ScalarTransformer(
            min_val=0.0, max_val=100.0, num_s=1024, num_as=40, num_t=2, seed=42
        )

        # First execution
        transformer.set_value(50.0)
        transformer.execute(False)

        # Second execution with different value
        transformer.set_value(60.0)
        transformer.execute(False)

        # Output should have changed
        assert transformer.has_changed()


class TestScalarTransformerOperations:
    """Test ScalarTransformer operations."""

    def test_clear(self) -> None:
        """Test clearing transformer state."""
        transformer = ScalarTransformer(
            min_val=0.0, max_val=100.0, num_s=1024, num_as=40, num_t=2, seed=42
        )
        transformer.set_value(50.0)
        transformer.execute(False)

        transformer.clear()

        # After clear, output should be empty
        output = transformer.output()
        assert output.num_set() == 0

    def test_memory_usage(self) -> None:
        """Test memory usage estimation."""
        transformer = ScalarTransformer(
            min_val=0.0, max_val=100.0, num_s=1024, num_as=40, num_t=2, seed=42
        )
        usage = transformer.memory_usage()
        assert isinstance(usage, int)
        assert usage > 0


class TestScalarTransformerRepr:
    """Test ScalarTransformer representation."""

    def test_repr(self) -> None:
        """Test __repr__."""
        transformer = ScalarTransformer(
            min_val=0.0, max_val=100.0, num_s=1024, num_as=40, num_t=2, seed=42
        )
        repr_str = repr(transformer)
        assert "ScalarTransformer" in repr_str
        assert "min_val=0.0" in repr_str or "min_val=0" in repr_str
        assert "max_val=100.0" in repr_str or "max_val=100" in repr_str
        assert "num_s=1024" in repr_str
        assert "num_as=40" in repr_str


class TestScalarTransformerIntegration:
    """Integration tests for ScalarTransformer."""

    def test_continuous_encoding_workflow(self) -> None:
        """Test a complete workflow encoding continuous values."""
        transformer = ScalarTransformer(
            min_val=0.0, max_val=100.0, num_s=1024, num_as=40, num_t=2, seed=42
        )

        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        outputs = []

        for value in values:
            transformer.set_value(value)
            transformer.execute(False)
            outputs.append(transformer.output())

        # All outputs should be sparse
        assert all(output.num_set() == 40 for output in outputs)

        # All encodings should be valid
        assert all(len(output) == 1024 for output in outputs)
