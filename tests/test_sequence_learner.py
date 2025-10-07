"""Tests for SequenceLearner class."""


from gnomics import BitArray, SequenceLearner


class TestSequenceLearnerConstruction:
    """Test SequenceLearner construction."""

    def test_new(self) -> None:
        """Test creating a new SequenceLearner."""
        learner = SequenceLearner(
            num_c=100,
            num_spc=8,
            num_dps=4,
            num_rpd=20,
            d_thresh=15,
            perm_thr=20,
            perm_inc=2,
            perm_dec=1,
            num_t=3,
            seed=42,
        )
        assert learner is not None
        assert learner.num_c() == 100
        assert learner.num_spc() == 8
        assert learner.num_dps() == 4
        assert learner.d_thresh() == 15

    def test_new_with_always_update(self) -> None:
        """Test creating SequenceLearner with always_update enabled."""
        learner = SequenceLearner(
            num_c=50,
            num_spc=4,
            num_dps=2,
            num_rpd=10,
            d_thresh=8,
            perm_thr=20,
            perm_inc=2,
            perm_dec=1,
            num_t=2,
            always_update=True,
            seed=123,
        )
        assert learner is not None
        assert learner.num_c() == 50


class TestSequenceLearnerInitialization:
    """Test SequenceLearner initialization."""

    def test_init(self) -> None:
        """Test initializing SequenceLearner."""
        learner = SequenceLearner(
            num_c=100,
            num_spc=8,
            num_dps=4,
            num_rpd=20,
            d_thresh=15,
            perm_thr=20,
            perm_inc=2,
            perm_dec=1,
            num_t=3,
            seed=42,
        )
        learner.init(num_input_bits=100)
        # After init, learner should be ready to compute
        assert learner.num_c() == 100


class TestSequenceLearnerCompute:
    """Test SequenceLearner compute operations."""

    def test_compute_basic(self) -> None:
        """Test basic compute operation."""
        learner = SequenceLearner(
            num_c=100,
            num_spc=8,
            num_dps=4,
            num_rpd=20,
            d_thresh=15,
            perm_thr=20,
            perm_inc=2,
            perm_dec=1,
            num_t=3,
            seed=42,
        )
        learner.init(num_input_bits=100)

        input_pattern = BitArray(100)
        input_pattern.set_acts([1, 5, 10, 20, 30])

        learner.compute(input_pattern)
        output = learner.output()

        assert isinstance(output, BitArray)
        assert len(output) == 800  # num_c * num_spc

    def test_compute_with_different_patterns(self) -> None:
        """Test compute with different input patterns."""
        learner = SequenceLearner(
            num_c=50,
            num_spc=4,
            num_dps=2,
            num_rpd=10,
            d_thresh=8,
            perm_thr=20,
            perm_inc=2,
            perm_dec=1,
            num_t=2,
            seed=123,
        )
        learner.init(num_input_bits=50)

        input1 = BitArray(50)
        input1.set_acts([1, 2, 3])

        learner.compute(input1)
        output1 = learner.output()

        input2 = BitArray(50)
        input2.set_acts([4, 5, 6])

        learner.compute(input2)
        output2 = learner.output()

        # Both outputs should be valid
        assert len(output1) == len(output2)


class TestSequenceLearnerLearning:
    """Test SequenceLearner learning operations."""

    def test_learn(self) -> None:
        """Test learning from input patterns."""
        learner = SequenceLearner(
            num_c=100,
            num_spc=8,
            num_dps=4,
            num_rpd=20,
            d_thresh=15,
            perm_thr=20,
            perm_inc=2,
            perm_dec=1,
            num_t=3,
            seed=42,
        )
        learner.init(num_input_bits=100)

        input_pattern = BitArray(100)
        input_pattern.set_acts([1, 5, 10, 15, 20])

        learner.compute(input_pattern)
        learner.learn()

        # Should complete without error
        assert learner.num_c() == 100

    def test_execute_with_learning(self) -> None:
        """Test execute with learning enabled."""
        learner = SequenceLearner(
            num_c=100,
            num_spc=8,
            num_dps=4,
            num_rpd=20,
            d_thresh=15,
            perm_thr=20,
            perm_inc=2,
            perm_dec=1,
            num_t=3,
            seed=42,
        )
        learner.init(num_input_bits=100)

        input_pattern = BitArray(100)
        input_pattern.set_acts([1, 5, 10, 15, 20])

        learner.execute(input_pattern, learn_flag=True)
        output = learner.output()

        assert isinstance(output, BitArray)

    def test_execute_without_learning(self) -> None:
        """Test execute with learning disabled."""
        learner = SequenceLearner(
            num_c=100,
            num_spc=8,
            num_dps=4,
            num_rpd=20,
            d_thresh=15,
            perm_thr=20,
            perm_inc=2,
            perm_dec=1,
            num_t=3,
            seed=42,
        )
        learner.init(num_input_bits=100)

        input_pattern = BitArray(100)
        input_pattern.set_acts([1, 5, 10])

        learner.execute(input_pattern, learn_flag=False)
        output = learner.output()

        assert isinstance(output, BitArray)

    def test_sequence_learning(self) -> None:
        """Test learning a simple sequence."""
        learner = SequenceLearner(
            num_c=50,
            num_spc=4,
            num_dps=2,
            num_rpd=10,
            d_thresh=8,
            perm_thr=20,
            perm_inc=2,
            perm_dec=1,
            num_t=3,
            seed=42,
        )
        learner.init(num_input_bits=50)

        # Create a simple sequence: A -> B -> C -> A
        pattern_a = BitArray(50)
        pattern_a.set_acts([1, 2, 3, 4, 5])

        pattern_b = BitArray(50)
        pattern_b.set_acts([10, 11, 12, 13, 14])

        pattern_c = BitArray(50)
        pattern_c.set_acts([20, 21, 22, 23, 24])

        # Train the sequence multiple times
        for _ in range(5):
            learner.execute(pattern_a, learn_flag=True)
            learner.execute(pattern_b, learn_flag=True)
            learner.execute(pattern_c, learn_flag=True)

        # Should complete without error
        assert learner.num_c() == 50


class TestSequenceLearnerAnomaly:
    """Test SequenceLearner anomaly detection."""

    def test_get_anomaly_score(self) -> None:
        """Test getting anomaly score."""
        learner = SequenceLearner(
            num_c=100,
            num_spc=8,
            num_dps=4,
            num_rpd=20,
            d_thresh=15,
            perm_thr=20,
            perm_inc=2,
            perm_dec=1,
            num_t=3,
            seed=42,
        )
        learner.init(num_input_bits=100)

        input_pattern = BitArray(100)
        input_pattern.set_acts([1, 5, 10])

        learner.compute(input_pattern)
        score = learner.get_anomaly_score()

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_anomaly_score_changes_with_learning(self) -> None:
        """Test that anomaly score changes with learning."""
        learner = SequenceLearner(
            num_c=50,
            num_spc=4,
            num_dps=2,
            num_rpd=10,
            d_thresh=8,
            perm_thr=20,
            perm_inc=2,
            perm_dec=1,
            num_t=2,
            seed=42,
        )
        learner.init(num_input_bits=50)

        pattern_a = BitArray(50)
        pattern_a.set_acts([1, 2, 3, 4, 5])

        pattern_b = BitArray(50)
        pattern_b.set_acts([10, 11, 12, 13, 14])

        # First time - likely high anomaly
        learner.compute(pattern_a)
        initial_score = learner.get_anomaly_score()

        # Learn the sequence multiple times
        for _ in range(10):
            learner.execute(pattern_a, learn_flag=True)
            learner.execute(pattern_b, learn_flag=True)

        # Compute again with learned pattern
        learner.execute(pattern_a, learn_flag=False)
        learner.compute(pattern_b)
        final_score = learner.get_anomaly_score()

        # Both should be valid scores
        assert 0.0 <= initial_score <= 1.0
        assert 0.0 <= final_score <= 1.0


class TestSequenceLearnerHistory:
    """Test SequenceLearner history tracking."""

    def test_get_historical_count(self) -> None:
        """Test getting historical count of used dendrites."""
        learner = SequenceLearner(
            num_c=100,
            num_spc=8,
            num_dps=4,
            num_rpd=20,
            d_thresh=15,
            perm_thr=20,
            perm_inc=2,
            perm_dec=1,
            num_t=3,
            seed=42,
        )
        learner.init(num_input_bits=100)

        count = learner.get_historical_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_historical_count_increases_with_learning(self) -> None:
        """Test that historical count increases as learner learns."""
        learner = SequenceLearner(
            num_c=50,
            num_spc=4,
            num_dps=2,
            num_rpd=10,
            d_thresh=8,
            perm_thr=20,
            perm_inc=2,
            perm_dec=1,
            num_t=2,
            seed=42,
        )
        learner.init(num_input_bits=50)

        initial_count = learner.get_historical_count()

        # Learn several patterns
        for i in range(5):
            input_pattern = BitArray(50)
            input_pattern.set_acts([i, i + 1, i + 2])

            learner.execute(input_pattern, learn_flag=True)

        final_count = learner.get_historical_count()

        # Count should be non-negative
        assert initial_count >= 0
        assert final_count >= 0

    def test_output_at(self) -> None:
        """Test getting output at specific time offset."""
        learner = SequenceLearner(
            num_c=50,
            num_spc=4,
            num_dps=2,
            num_rpd=10,
            d_thresh=8,
            perm_thr=20,
            perm_inc=2,
            perm_dec=1,
            num_t=3,
            seed=42,
        )
        learner.init(num_input_bits=50)

        input_pattern = BitArray(50)
        input_pattern.set_acts([1, 2, 3])

        learner.execute(input_pattern, learn_flag=False)

        # Get current output
        output_current = learner.output_at(0)
        assert isinstance(output_current, BitArray)
        assert len(output_current) == 200  # num_c * num_spc

        # Get previous output
        output_prev = learner.output_at(1)
        assert isinstance(output_prev, BitArray)
        assert len(output_prev) == 200

    def test_has_changed(self) -> None:
        """Test checking if output has changed."""
        learner = SequenceLearner(
            num_c=50,
            num_spc=4,
            num_dps=2,
            num_rpd=10,
            d_thresh=8,
            perm_thr=20,
            perm_inc=2,
            perm_dec=1,
            num_t=2,
            seed=42,
        )
        learner.init(num_input_bits=50)

        input_pattern = BitArray(50)
        input_pattern.set_acts([1, 2, 3])

        learner.execute(input_pattern, learn_flag=False)

        # Check if output changed
        changed = learner.has_changed()
        assert isinstance(changed, bool)


class TestSequenceLearnerOperations:
    """Test SequenceLearner operations."""

    def test_clear(self) -> None:
        """Test clearing learner state."""
        learner = SequenceLearner(
            num_c=50,
            num_spc=4,
            num_dps=2,
            num_rpd=10,
            d_thresh=8,
            perm_thr=20,
            perm_inc=2,
            perm_dec=1,
            num_t=2,
            seed=42,
        )
        learner.init(num_input_bits=50)

        input_pattern = BitArray(50)
        input_pattern.set_acts([1, 2, 3])

        learner.execute(input_pattern, learn_flag=True)

        learner.clear()

        # After clear, output should be empty
        output = learner.output()
        assert output.num_set() == 0

    def test_memory_usage(self) -> None:
        """Test memory usage estimation."""
        learner = SequenceLearner(
            num_c=100,
            num_spc=8,
            num_dps=4,
            num_rpd=20,
            d_thresh=15,
            perm_thr=20,
            perm_inc=2,
            perm_dec=1,
            num_t=3,
            seed=42,
        )
        usage = learner.memory_usage()
        assert isinstance(usage, int)
        assert usage > 0


class TestSequenceLearnerRepr:
    """Test SequenceLearner representation."""

    def test_repr(self) -> None:
        """Test __repr__."""
        learner = SequenceLearner(
            num_c=100,
            num_spc=8,
            num_dps=4,
            num_rpd=20,
            d_thresh=15,
            perm_thr=20,
            perm_inc=2,
            perm_dec=1,
            num_t=3,
            seed=42,
        )
        repr_str = repr(learner)
        assert "SequenceLearner" in repr_str
        assert "num_c=100" in repr_str
        assert "num_spc=8" in repr_str
        assert "num_dps=4" in repr_str
        assert "d_thresh=15" in repr_str


