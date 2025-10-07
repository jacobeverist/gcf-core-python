"""Tests for PersistenceTransformer class."""

from gnomics import BitArray, PersistenceTransformer


class TestPersistenceTransformerConstruction:
    """Test PersistenceTransformer construction."""

    def test_new(self) -> None:
        """Test creating a new PersistenceTransformer."""
        transformer = PersistenceTransformer(
            min_val=0.0,
            max_val=100.0,
            num_s=1024,
            num_as=40,
            max_step=100,
            num_t=2,
            seed=42,
        )
        assert transformer is not None
        assert transformer.min_val() == 0.0
        assert transformer.max_val() == 100.0
        assert transformer.num_s() == 1024
        assert transformer.num_as() == 40
        assert transformer.max_step() == 100


class TestPersistenceTransformerValueManagement:
    """Test PersistenceTransformer value management."""

    def test_set_get_value(self) -> None:
        """Test setting and getting scalar values."""
        transformer = PersistenceTransformer(
            min_val=0.0,
            max_val=100.0,
            num_s=1024,
            num_as=40,
            max_step=100,
            num_t=2,
            seed=42,
        )
        transformer.set_value(50.0)
        assert transformer.get_value() == 50.0

    def test_get_counter(self) -> None:
        """Test accessing the persistence counter."""
        transformer = PersistenceTransformer(
            min_val=0.0,
            max_val=100.0,
            num_s=1024,
            num_as=40,
            max_step=100,
            num_t=2,
            seed=42,
        )
        # Initial counter should be 0
        counter = transformer.get_counter()
        assert isinstance(counter, int)
        assert counter >= 0


class TestPersistenceTransformerEncoding:
    """Test PersistenceTransformer encoding."""

    def test_execute(self) -> None:
        """Test executing the computation pipeline."""
        transformer = PersistenceTransformer(
            min_val=0.0,
            max_val=100.0,
            num_s=1024,
            num_as=40,
            max_step=100,
            num_t=2,
            seed=42,
        )
        transformer.set_value(50.0)
        transformer.execute(False)
        output = transformer.output()
        assert isinstance(output, BitArray)
        assert len(output) == 1024

    def test_output_sparsity(self) -> None:
        """Test that output has expected number of active bits."""
        transformer = PersistenceTransformer(
            min_val=0.0,
            max_val=100.0,
            num_s=1024,
            num_as=40,
            max_step=100,
            num_t=2,
            seed=42,
        )
        transformer.set_value(50.0)
        transformer.execute(False)
        output = transformer.output()
        # Should have exactly num_as active bits
        assert output.num_set() == 40

    def test_persistence_counter_stable(self) -> None:
        """Test that counter increments when value is stable."""
        transformer = PersistenceTransformer(
            min_val=0.0,
            max_val=100.0,
            num_s=1024,
            num_as=40,
            max_step=100,
            num_t=2,
            seed=42,
        )

        # Set initial value
        transformer.set_value(50.0)
        transformer.execute(False)
        counter1 = transformer.get_counter()

        # Keep value stable (within 10% threshold)
        transformer.set_value(50.0)
        transformer.execute(False)
        counter2 = transformer.get_counter()

        # Counter should increment for stable values
        assert counter2 >= counter1

    def test_persistence_counter_reset(self) -> None:
        """Test that counter resets on significant value change."""
        transformer = PersistenceTransformer(
            min_val=0.0,
            max_val=100.0,
            num_s=1024,
            num_as=40,
            max_step=100,
            num_t=2,
            seed=42,
        )

        # Build up some persistence
        transformer.set_value(50.0)
        transformer.execute(False)
        transformer.set_value(50.0)
        transformer.execute(False)
        transformer.set_value(50.0)
        transformer.execute(False)
        counter_stable = transformer.get_counter()
        assert counter_stable > 0

        # Make significant change (>10% of range)
        transformer.set_value(80.0)  # 30% change
        transformer.execute(False)
        counter_after_change = transformer.get_counter()

        # Counter should reset to 0
        assert counter_after_change == 0


class TestPersistenceTransformerHistory:
    """Test PersistenceTransformer history tracking."""

    def test_output_at(self) -> None:
        """Test accessing historical outputs."""
        transformer = PersistenceTransformer(
            min_val=0.0,
            max_val=100.0,
            num_s=1024,
            num_as=40,
            max_step=100,
            num_t=3,
            seed=42,
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
        transformer = PersistenceTransformer(
            min_val=0.0,
            max_val=100.0,
            num_s=1024,
            num_as=40,
            max_step=100,
            num_t=2,
            seed=42,
        )

        # First execution
        transformer.set_value(50.0)
        transformer.execute(False)

        # Second execution with different persistence
        transformer.set_value(50.0)
        transformer.execute(False)

        # Output may or may not have changed depending on encoding
        # Just verify the method works
        changed = transformer.has_changed()
        assert isinstance(changed, bool)


class TestPersistenceTransformerOperations:
    """Test PersistenceTransformer operations."""

    def test_clear(self) -> None:
        """Test clearing transformer state."""
        transformer = PersistenceTransformer(
            min_val=0.0,
            max_val=100.0,
            num_s=1024,
            num_as=40,
            max_step=100,
            num_t=2,
            seed=42,
        )
        transformer.set_value(50.0)
        transformer.execute(False)

        transformer.clear()

        # After clear, output should be empty
        output = transformer.output()
        assert output.num_set() == 0

        # Counter should also be reset
        assert transformer.get_counter() == 0

    def test_memory_usage(self) -> None:
        """Test memory usage estimation."""
        transformer = PersistenceTransformer(
            min_val=0.0,
            max_val=100.0,
            num_s=1024,
            num_as=40,
            max_step=100,
            num_t=2,
            seed=42,
        )
        usage = transformer.memory_usage()
        assert isinstance(usage, int)
        assert usage > 0


class TestPersistenceTransformerRepr:
    """Test PersistenceTransformer representation."""

    def test_repr(self) -> None:
        """Test __repr__."""
        transformer = PersistenceTransformer(
            min_val=0.0,
            max_val=100.0,
            num_s=1024,
            num_as=40,
            max_step=100,
            num_t=2,
            seed=42,
        )
        repr_str = repr(transformer)
        assert "PersistenceTransformer" in repr_str
        assert "min_val=0.0" in repr_str or "min_val=0" in repr_str
        assert "max_val=100.0" in repr_str or "max_val=100" in repr_str
        assert "num_s=1024" in repr_str
        assert "num_as=40" in repr_str
        assert "max_step=100" in repr_str


class TestPersistenceTransformerIntegration:
    """Integration tests for PersistenceTransformer."""

    def test_temporal_stability_workflow(self) -> None:
        """Test a complete workflow tracking temporal persistence."""
        transformer = PersistenceTransformer(
            min_val=0.0,
            max_val=100.0,
            num_s=1024,
            num_as=40,
            max_step=100,
            num_t=2,
            seed=42,
        )

        # Scenario: stable period, then change, then stable again
        values = [50.0, 50.0, 50.0, 50.0, 80.0, 80.0, 80.0]
        counters = []

        for value in values:
            transformer.set_value(value)
            transformer.execute(False)
            counters.append(transformer.get_counter())

        # During first stable period, counter should increase
        assert counters[0] < counters[3]

        # After big change, counter should reset
        assert counters[4] == 0

        # During second stable period, counter should increase again
        assert counters[4] < counters[6]

    def test_counter_capping(self) -> None:
        """Test that counter caps at max_step."""
        transformer = PersistenceTransformer(
            min_val=0.0,
            max_val=100.0,
            num_s=1024,
            num_as=40,
            max_step=10,  # Low max for testing
            num_t=2,
            seed=42,
        )

        # Run many stable steps
        for _ in range(20):
            transformer.set_value(50.0)
            transformer.execute(False)

        # Counter should be capped at max_step
        counter = transformer.get_counter()
        assert counter <= 10
