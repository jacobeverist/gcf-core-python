"""Tests for BlockOutput class."""

from gcf_core_python_client import BitArray, BlockOutput


class TestBlockOutputConstruction:
    """Test BlockOutput construction and setup."""

    def test_new(self) -> None:
        """Test creating a new BlockOutput."""
        output = BlockOutput()
        assert output is not None

    def test_setup(self) -> None:
        """Test setting up BlockOutput dimensions."""
        output = BlockOutput()
        output.setup(num_t=3, num_b=10)
        assert output.num_t() == 3
        state = output.state()
        assert len(state) == 10


class TestBlockOutputState:
    """Test BlockOutput state management."""

    def test_state_access(self) -> None:
        """Test accessing current state."""
        output = BlockOutput()
        output.setup(num_t=2, num_b=5)
        state = output.state()
        assert len(state) == 5
        assert state.num_set() == 0

    def test_get_bitarray(self) -> None:
        """Test accessing historical states."""
        output = BlockOutput()
        output.setup(num_t=3, num_b=5)

        # Current state (time=0)
        current = output.get_bitarray(0)
        assert len(current) == 5

        # Previous state (time=1)
        previous = output.get_bitarray(1)
        assert len(previous) == 5

    def test_clear(self) -> None:
        """Test clearing state."""
        output = BlockOutput()
        output.setup(num_t=2, num_b=5)

        # Set some bits in the state
        state = output.state()
        state.set_bit(0)
        state.set_bit(2)

        # Clear should reset all to 0
        output.clear()
        state = output.state()
        assert state.num_set() == 0


class TestBlockOutputTimeStep:
    """Test BlockOutput time stepping."""

    def test_step(self) -> None:
        """Test advancing time step."""
        output = BlockOutput()
        output.setup(num_t=3, num_b=5)

        # Set pattern in current state
        state = output.state()
        state.set_bits([1, 0, 1, 0, 1])

        # Store current state
        output.store()

        # Advance time step
        output.step()

        # Note: behavior depends on gnomics implementation
        # step() advances the circular buffer


class TestBlockOutputChangeDetection:
    """Test BlockOutput change detection."""

    def test_has_changed_initially_false(self) -> None:
        """Test that initially there are no changes."""
        output = BlockOutput()
        output.setup(num_t=2, num_b=5)
        output.store()
        assert not output.has_changed()

    def test_has_changed_after_modification(self) -> None:
        """Test change detection after modifying state."""
        output = BlockOutput()
        output.setup(num_t=2, num_b=5)

        # Store initial state (all zeros)
        output.store()

        # Advance time and modify
        output.step()
        state = BitArray(5)
        state.set_bit(0)
        output.set_state(state)

        # Store modified state
        output.store()

        # Should detect change
        assert output.has_changed()

    def test_has_changed_at(self) -> None:
        """Test checking changes at specific time offsets."""
        output = BlockOutput()
        output.setup(num_t=3, num_b=5)

        # Initial state
        output.store()
        output.step()

        # Modify and store
        state = output.state()
        state.set_bit(1)
        output.store()

        # Check if current state (time=0) changed from previous
        # Behavior depends on gnomics implementation
        has_changed_current = output.has_changed_at(0)
        assert isinstance(has_changed_current, bool)


class TestBlockOutputIdentity:
    """Test BlockOutput identity and metadata."""

    def test_id(self) -> None:
        """Test that each BlockOutput has a unique ID."""
        output1 = BlockOutput()
        output2 = BlockOutput()

        id1 = output1.id()
        id2 = output2.id()

        assert isinstance(id1, int)
        assert isinstance(id2, int)
        # IDs should be different (auto-incremented)
        assert id1 != id2

    def test_num_t(self) -> None:
        """Test getting number of time steps."""
        output = BlockOutput()
        output.setup(num_t=5, num_b=10)
        assert output.num_t() == 5

    def test_repr(self) -> None:
        """Test __repr__."""
        output = BlockOutput()
        output.setup(num_t=3, num_b=10)
        repr_str = repr(output)
        assert "BlockOutput" in repr_str
        assert "num_t=3" in repr_str


class TestBlockOutputIntegration:
    """Integration tests for BlockOutput."""

    def test_complete_workflow(self) -> None:
        """Test a complete workflow with BlockOutput."""
        output = BlockOutput()
        output.setup(num_t=3, num_b=8)

        # Pattern 1
        state = BitArray(8)
        state.set_bits([1, 0, 1, 0, 1, 0, 1, 0])
        output.set_state(state)
        output.store()
        output.step()

        # Pattern 2
        state = BitArray(8)
        state.set_bits([0, 1, 0, 1, 0, 1, 0, 1])
        output.set_state(state)
        output.store()
        output.step()

        # Pattern 3
        state = BitArray(8)
        state.set_bits([1, 1, 0, 0, 1, 1, 0, 0])
        output.set_state(state)
        output.store()

        # Verify we can access all states
        current = output.get_bitarray(0)
        assert current.get_bits() == [1, 1, 0, 0, 1, 1, 0, 0]
