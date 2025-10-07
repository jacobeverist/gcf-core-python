"""Tests for BlockInput wrapper."""

import pytest
from gnomics.core import BitArray, BlockInput, BlockOutput


class TestBlockInputConstruction:
    """Test BlockInput construction."""

    def test_new(self):
        """Test creating a new BlockInput."""
        input_block = BlockInput()
        assert input_block.num_children() == 0
        assert input_block.num_bits() == 0

    def test_repr(self):
        """Test BlockInput repr."""
        input_block = BlockInput()
        repr_str = repr(input_block)
        assert "BlockInput" in repr_str
        assert "num_children=0" in repr_str


class TestBlockInputChildManagement:
    """Test adding and managing child outputs."""

    def test_add_single_child(self):
        """Test adding a single child output."""
        input_block = BlockInput()
        output = BlockOutput()
        output.setup(2, 32)

        input_block.add_child(output, 0)

        assert input_block.num_children() == 1
        assert input_block.num_bits() == 32

    def test_add_multiple_children(self):
        """Test adding multiple child outputs."""
        input_block = BlockInput()

        output1 = BlockOutput()
        output1.setup(2, 32)

        output2 = BlockOutput()
        output2.setup(2, 64)

        output3 = BlockOutput()
        output3.setup(2, 128)

        input_block.add_child(output1, 0)
        input_block.add_child(output2, 0)
        input_block.add_child(output3, 0)

        assert input_block.num_children() == 3
        assert input_block.num_bits() == 32 + 64 + 128

    def test_add_child_with_time_offset(self):
        """Test adding children with different time offsets."""
        input_block = BlockInput()

        output = BlockOutput()
        output.setup(5, 32)  # 5 time steps

        # Can add with different time offsets
        input_block.add_child(output, 0)  # Current
        input_block.add_child(output, 1)  # Previous
        input_block.add_child(output, 2)  # Two steps back

        assert input_block.num_children() == 3
        assert input_block.num_bits() == 32 * 3


class TestBlockInputDataPulling:
    """Test pulling data from child outputs."""

    def test_pull_single_child(self):
        """Test pulling data from a single child."""
        input_block = BlockInput()
        output = BlockOutput()
        output.setup(2, 32)

        # Set some bits in the output
        state = output.state()
        state.set_bit(5)
        state.set_bit(10)
        output.set_state(state)
        output.store()

        input_block.add_child(output, 0)
        input_block.pull()

        # Check that bits were copied
        input_state = input_block.state()
        assert input_state.get_bit(5) == 1
        assert input_state.get_bit(10) == 1

    def test_pull_concatenates_children(self):
        """Test that pull concatenates multiple children."""
        input_block = BlockInput()

        # First child: 32 bits with bit 5 set
        output1 = BlockOutput()
        output1.setup(2, 32)
        state1 = output1.state()
        state1.set_bit(5)
        output1.set_state(state1)
        output1.store()

        # Second child: 32 bits with bit 10 set
        output2 = BlockOutput()
        output2.setup(2, 32)
        state2 = output2.state()
        state2.set_bit(10)
        output2.set_state(state2)
        output2.store()

        input_block.add_child(output1, 0)
        input_block.add_child(output2, 0)
        input_block.pull()

        # Check concatenation
        input_state = input_block.state()
        assert input_state.get_bit(5) == 1  # From output1
        assert input_state.get_bit(32 + 10) == 1  # From output2 (offset by 32)
        assert input_state.get_bit(0) == 0  # Unset bit

    def test_pull_from_historical_output(self):
        """Test pulling from a historical time offset."""
        input_block = BlockInput()
        output = BlockOutput()
        output.setup(3, 32)  # 3 time steps

        # Set bit 5 in current state
        state = output.state()
        state.set_bit(5)
        output.set_state(state)
        output.store()

        # Step forward
        output.step()

        # Clear and set bit 10 in new current state
        state = output.state()
        state.clear_all()
        state.set_bit(10)
        output.set_state(state)
        output.store()

        # Add child with time offset 0 (current, bit 10) and 1 (previous, bit 5)
        input_block.add_child(output, 0)  # Current state
        input_block.add_child(output, 1)  # Previous state

        input_block.pull()

        input_state = input_block.state()
        # First 32 bits from current state (bit 10 set)
        assert input_state.get_bit(10) == 1
        assert input_state.get_bit(5) == 0

        # Next 32 bits from previous state (bit 5 set)
        assert input_state.get_bit(32 + 5) == 1
        assert input_state.get_bit(32 + 10) == 0


class TestBlockInputLazyCopying:
    """Test lazy copying optimization."""

    def test_children_changed_initially_false(self):
        """Test that children_changed is false when nothing changed."""
        input_block = BlockInput()
        output = BlockOutput()
        output.setup(2, 32)
        output.store()

        # Step without modification
        output.step()
        output.store()

        input_block.add_child(output, 0)

        # No change should be detected
        assert not input_block.children_changed()

    def test_children_changed_after_modification(self):
        """Test that children_changed detects modifications."""
        input_block = BlockInput()
        output = BlockOutput()
        output.setup(2, 32)
        output.store()

        input_block.add_child(output, 0)

        # Modify the output
        state = output.state()
        state.set_bit(5)
        output.set_state(state)
        output.store()

        # Change should be detected
        assert input_block.children_changed()

    def test_children_changed_short_circuits(self):
        """Test that children_changed short-circuits on first change."""
        input_block = BlockInput()

        # First output - unchanged
        output1 = BlockOutput()
        output1.setup(2, 32)
        output1.store()
        output1.step()
        output1.store()

        # Second output - changed
        output2 = BlockOutput()
        output2.setup(2, 32)
        output2.store()
        state = output2.state()
        state.set_bit(5)
        output2.set_state(state)
        output2.store()

        input_block.add_child(output1, 0)
        input_block.add_child(output2, 0)

        # Should detect change (from output2)
        assert input_block.children_changed()

    def test_pull_skips_unchanged_children(self):
        """Test that pull skips copying from unchanged children."""
        input_block = BlockInput()

        # First output - set bit 5, then step without change
        output1 = BlockOutput()
        output1.setup(2, 32)
        state = output1.state()
        state.set_bit(5)
        output1.set_state(state)
        output1.store()

        input_block.add_child(output1, 0)

        # First pull - should copy
        input_block.pull()
        input_state = input_block.state()
        assert input_state.get_bit(5) == 1

        # Clear input state manually
        input_block.clear()
        input_state = input_block.state()
        assert input_state.get_bit(5) == 0

        # Step output without modification
        output1.step()
        output1.store()

        # Second pull - should skip copy (child unchanged)
        input_block.pull()
        input_state = input_block.state()
        # State should still be clear (no copy happened)
        assert input_state.get_bit(5) == 0

    def test_pull_copies_only_changed_children(self):
        """Test that pull only copies from changed children."""
        input_block = BlockInput()

        # First output - unchanged after initial store
        output1 = BlockOutput()
        output1.setup(2, 32)
        state1 = output1.state()
        state1.set_bit(5)
        output1.set_state(state1)
        output1.store()

        # Second output - will be changed
        output2 = BlockOutput()
        output2.setup(2, 32)
        state2 = output2.state()
        state2.set_bit(10)
        output2.set_state(state2)
        output2.store()

        input_block.add_child(output1, 0)
        input_block.add_child(output2, 0)

        # First pull
        input_block.pull()
        input_state = input_block.state()
        assert input_state.get_bit(5) == 1
        assert input_state.get_bit(32 + 10) == 1

        # Step both outputs
        output1.step()
        output1.store()  # No change

        output2.step()
        state2 = output2.state()
        state2.set_bit(15)  # Add new bit
        output2.set_state(state2)
        output2.store()  # Changed

        # Clear input to verify selective copy
        input_block.clear()

        # Second pull - should only copy from output2
        input_block.pull()
        input_state = input_block.state()

        # First 32 bits should still be clear (output1 unchanged, not copied)
        assert input_state.get_bit(5) == 0

        # Second 32 bits should have new data (output2 changed, copied)
        assert input_state.get_bit(32 + 15) == 1


class TestBlockInputConnections:
    """Test connecting block outputs to block inputs."""

    def test_shared_output_multiple_inputs(self):
        """Test that a single output can be shared by multiple inputs."""
        output = BlockOutput()
        output.setup(2, 32)
        state = output.state()
        state.set_bit(7)
        output.set_state(state)
        output.store()

        # Create two inputs connected to the same output
        input1 = BlockInput()
        input1.add_child(output, 0)

        input2 = BlockInput()
        input2.add_child(output, 0)

        # Both inputs should pull the same data
        input1.pull()
        input2.pull()

        assert input1.state().get_bit(7) == 1
        assert input2.state().get_bit(7) == 1

    def test_chained_connections(self):
        """Test chaining outputs from one block to inputs of another."""
        # This simulates a simple pipeline: output1 -> (process) -> output2 -> input

        # First output
        output1 = BlockOutput()
        output1.setup(2, 32)
        state = output1.state()
        state.set_bit(3)
        output1.set_state(state)
        output1.store()

        # Create input that uses output1
        intermediate_input = BlockInput()
        intermediate_input.add_child(output1, 0)
        intermediate_input.pull()

        # Second output (simulating processing)
        output2 = BlockOutput()
        output2.setup(2, 32)
        # Copy from intermediate input and add bit
        state2 = intermediate_input.state()
        state2.set_bit(8)
        output2.set_state(state2)
        output2.store()

        # Final input
        final_input = BlockInput()
        final_input.add_child(output2, 0)
        final_input.pull()

        # Should have both bits
        final_state = final_input.state()
        assert final_state.get_bit(3) == 1  # From output1
        assert final_state.get_bit(8) == 1  # Added in output2

    def test_multiple_outputs_different_sizes(self):
        """Test connecting outputs of different sizes."""
        input_block = BlockInput()

        # Use sizes that are multiples of 32 (word size) to avoid rounding
        sizes = [32, 64, 128, 256, 512]
        outputs = []

        for i, size in enumerate(sizes):
            output = BlockOutput()
            output.setup(2, size)
            state = output.state()
            state.set_bit(i)  # Set different bit in each
            output.set_state(state)
            output.store()
            outputs.append(output)
            input_block.add_child(output, 0)

        input_block.pull()

        # Verify total size
        expected_total = sum(sizes)
        assert input_block.num_bits() == expected_total

        # Verify each output's contribution
        offset = 0
        for i, size in enumerate(sizes):
            state = input_block.state()
            assert state.get_bit(offset + i) == 1
            offset += size

    def test_temporal_connections(self):
        """Test connecting to historical states of outputs."""
        output = BlockOutput()
        output.setup(4, 32)  # 4 time steps

        # Create a sequence: bit 0, bit 1, bit 2, bit 3
        for i in range(4):
            state = output.state()
            state.set_bit(i)
            output.set_state(state)
            output.store()
            if i < 3:
                output.step()

        # Connect to current and three previous states
        input_block = BlockInput()
        input_block.add_child(output, 0)  # Current (bit 3)
        input_block.add_child(output, 1)  # Previous (bit 2)
        input_block.add_child(output, 2)  # Two back (bit 1)
        input_block.add_child(output, 3)  # Three back (bit 0)

        input_block.pull()

        # Verify temporal sequence is preserved
        state = input_block.state()
        assert state.get_bit(32 * 0 + 3) == 1  # Current
        assert state.get_bit(32 * 1 + 2) == 1  # Previous
        assert state.get_bit(32 * 2 + 1) == 1  # Two back
        assert state.get_bit(32 * 3 + 0) == 1  # Three back


class TestBlockInputOperations:
    """Test BlockInput operations."""

    def test_clear(self):
        """Test clearing the input state."""
        input_block = BlockInput()
        output = BlockOutput()
        output.setup(2, 32)

        state = output.state()
        state.set_bit(5)
        state.set_bit(10)
        output.set_state(state)
        output.store()

        input_block.add_child(output, 0)
        input_block.pull()

        # State should have bits set
        assert input_block.state().num_set() > 0

        # Clear
        input_block.clear()

        # State should be all zeros
        assert input_block.state().num_set() == 0

    def test_id(self):
        """Test that each input has a unique ID."""
        input1 = BlockInput()
        input2 = BlockInput()
        input3 = BlockInput()

        # IDs should be unique
        ids = {input1.id(), input2.id(), input3.id()}
        assert len(ids) == 3

    def test_memory_usage(self):
        """Test memory usage reporting."""
        input_block = BlockInput()

        # Empty input should have some base memory
        base_usage = input_block.memory_usage()
        assert base_usage > 0

        # Add a child
        output = BlockOutput()
        output.setup(2, 1024)  # Large output
        input_block.add_child(output, 0)

        # Memory usage should increase
        new_usage = input_block.memory_usage()
        assert new_usage > base_usage


class TestBlockInputEdgeCases:
    """Test edge cases and error conditions."""

    def test_pull_without_children(self):
        """Test pulling with no children."""
        input_block = BlockInput()
        # Should not crash
        input_block.pull()
        assert input_block.num_bits() == 0

    def test_children_changed_without_children(self):
        """Test children_changed with no children."""
        input_block = BlockInput()
        # Should return False
        assert not input_block.children_changed()

    def test_state_empty_input(self):
        """Test getting state from empty input."""
        input_block = BlockInput()
        state = input_block.state()
        assert len(state) == 0
        assert state.num_set() == 0

    def test_repeated_pulls(self):
        """Test repeated pulling from same children."""
        input_block = BlockInput()
        output = BlockOutput()
        output.setup(2, 32)

        state = output.state()
        state.set_bit(5)
        output.set_state(state)
        output.store()

        input_block.add_child(output, 0)

        # Pull multiple times
        for _ in range(10):
            input_block.pull()

        # Should still work correctly
        assert input_block.state().get_bit(5) == 1


class TestBlockInputIntegration:
    """Integration tests combining multiple features."""

    def test_complex_network(self):
        """Test a complex network of outputs and inputs."""
        # Create a small network:
        # output1 -> input1 -> (process) -> output2 \
        #                                             -> input3
        # output3 -> input2 -> (process) -> output4 /

        # Layer 1 outputs
        output1 = BlockOutput()
        output1.setup(2, 32)
        state1 = output1.state()
        state1.set_bit(1)
        output1.set_state(state1)
        output1.store()

        output3 = BlockOutput()
        output3.setup(2, 32)
        state3 = output3.state()
        state3.set_bit(3)
        output3.set_state(state3)
        output3.store()

        # Layer 1 inputs
        input1 = BlockInput()
        input1.add_child(output1, 0)
        input1.pull()

        input2 = BlockInput()
        input2.add_child(output3, 0)
        input2.pull()

        # Layer 2 outputs (simulate processing)
        output2 = BlockOutput()
        output2.setup(2, 32)
        output2.set_state(input1.state())
        output2.store()

        output4 = BlockOutput()
        output4.setup(2, 32)
        output4.set_state(input2.state())
        output4.store()

        # Layer 2 input (combines both paths)
        input3 = BlockInput()
        input3.add_child(output2, 0)
        input3.add_child(output4, 0)
        input3.pull()

        # Verify final state has both bits
        final_state = input3.state()
        assert final_state.get_bit(1) == 1  # From output1 path
        assert final_state.get_bit(32 + 3) == 1  # From output3 path

    def test_lazy_copying_performance_scenario(self):
        """Test scenario where lazy copying provides benefit."""
        # Simulate a block with many stable inputs and one changing input

        input_block = BlockInput()
        outputs = []

        # Create 10 outputs, only last one will change
        for i in range(10):
            output = BlockOutput()
            output.setup(2, 32)
            state = output.state()
            state.set_bit(i)
            output.set_state(state)
            output.store()
            outputs.append(output)
            input_block.add_child(output, 0)

        # Initial pull
        input_block.pull()

        # Step all outputs
        for output in outputs:
            output.step()

        # Only modify the last output
        state = outputs[-1].state()
        state.set_bit(20)
        outputs[-1].set_state(state)
        outputs[-1].store()

        # Store others without change
        for output in outputs[:-1]:
            output.store()

        # Should detect that only one child changed
        assert input_block.children_changed()

        # Clear and pull again
        input_block.clear()
        input_block.pull()

        # Only the changed child should be copied
        # We can't directly verify this, but we can verify correctness
        final_state = input_block.state()
        # First 9 outputs should not be copied (cleared)
        for i in range(9):
            assert final_state.get_bit(32 * i + i) == 0
        # Last output should be copied
        assert final_state.get_bit(32 * 9 + 20) == 1
