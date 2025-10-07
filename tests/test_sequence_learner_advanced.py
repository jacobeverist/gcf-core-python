"""Advanced tests for SequenceLearner based on gcf-core-rust test suite."""

from gnomics import BitArray, SequenceLearner


class TestSequenceLearnerOutputSparsity:
    """Test SequenceLearner output sparsity."""

    def test_output_is_sparse(self) -> None:
        """Test that output is sparse (not all statelets active)."""
        learner = SequenceLearner(
            num_c=10,
            num_spc=4,
            num_dps=8,
            num_rpd=32,
            d_thresh=20,
            perm_thr=20,
            perm_inc=2,
            perm_dec=1,
            num_t=2,
            seed=42,
        )
        learner.init(num_input_bits=10)

        input_pattern = BitArray(10)
        input_pattern.set_acts([0, 1, 2, 3, 4])

        learner.execute(input_pattern, learn_flag=True)
        output = learner.output.state()

        num_active = output.num_set()
        total_statelets = 10 * 4  # num_c * num_spc

        assert num_active > 0, "Output should have some active statelets"
        assert num_active < total_statelets, "Output should be sparse"


class TestSequenceLearnerComplexSequences:
    """Test SequenceLearner with complex sequence patterns."""

    def test_complex_sequence_learning(self) -> None:
        """Test learning a longer, more complex sequence."""
        learner = SequenceLearner(
            num_c=20,
            num_spc=4,
            num_dps=8,
            num_rpd=32,
            d_thresh=20,
            perm_thr=20,
            perm_inc=2,
            perm_dec=1,
            num_t=2,
            seed=42,
        )
        learner.init(num_input_bits=20)

        # Create longer sequence of distinct patterns
        sequence = []
        for i in range(11):
            pattern = BitArray(20)
            pattern.set_acts([i, i + 1])
            sequence.append(pattern)

        # Train on sequence multiple times
        for _ in range(20):
            for pattern in sequence:
                learner.execute(pattern, learn_flag=True)

        # Verify that historical count increases
        count = learner.get_historical_count()
        assert count > 0, "Should have learned some patterns"

    def test_cyclic_pattern_learning(self) -> None:
        """Test learning cyclic patterns."""
        learner = SequenceLearner(
            num_c=10,
            num_spc=4,
            num_dps=8,
            num_rpd=32,
            d_thresh=20,
            perm_thr=20,
            perm_inc=2,
            perm_dec=1,
            num_t=2,
            seed=42,
        )
        learner.init(num_input_bits=10)

        # Create cyclic pattern: 0 -> 1 -> 2 -> 3 -> 0 -> 1 -> ...
        cycle = []
        for i in range(4):
            pattern = BitArray(10)
            pattern.set_acts([i * 2, i * 2 + 1])
            cycle.append(pattern)

        initial_count = learner.get_historical_count()

        # Train multiple full cycles
        for _ in range(20):
            for pattern in cycle:
                learner.execute(pattern, learn_flag=True)

        final_count = learner.get_historical_count()
        assert final_count > initial_count, "Should have learned the cyclic pattern"
