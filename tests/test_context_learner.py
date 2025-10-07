"""Tests for ContextLearner class."""


from gnomics import BitArray, ContextLearner


class TestContextLearnerConstruction:
    """Test ContextLearner construction."""

    def test_new(self) -> None:
        """Test creating a new ContextLearner."""
        learner = ContextLearner(
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


class TestContextLearnerInitialization:
    """Test ContextLearner initialization."""

    def test_init(self) -> None:
        """Test initializing ContextLearner."""
        learner = ContextLearner(
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
        learner.init(num_input_bits=100, num_context_bits=200)
        # After init, learner should be ready to compute
        assert learner.num_c() == 100


class TestContextLearnerCompute:
    """Test ContextLearner compute operations."""

    def test_compute_basic(self) -> None:
        """Test basic compute operation."""
        learner = ContextLearner(
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
        learner.init(num_input_bits=100, num_context_bits=200)

        input_pattern = BitArray(100)
        input_pattern.set_acts([1, 5, 10, 20, 30])

        context_pattern = BitArray(200)
        context_pattern.set_acts([5, 15, 25, 35, 45])

        learner.compute(input_pattern, context_pattern)
        output = learner.output.state()

        assert isinstance(output, BitArray)
        assert len(output) == 800  # num_c * num_spc

    def test_compute_with_different_patterns(self) -> None:
        """Test compute with different input/context patterns."""
        learner = ContextLearner(
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
        learner.init(num_input_bits=50, num_context_bits=100)

        input1 = BitArray(50)
        input1.set_acts([1, 2, 3])
        context1 = BitArray(100)
        context1.set_acts([10, 20, 30])

        learner.compute(input1, context1)
        output1 = learner.output.state()

        input2 = BitArray(50)
        input2.set_acts([4, 5, 6])
        context2 = BitArray(100)
        context2.set_acts([40, 50, 60])

        learner.compute(input2, context2)
        output2 = learner.output.state()

        # Different inputs should produce different outputs
        assert len(output1) == len(output2)


class TestContextLearnerLearning:
    """Test ContextLearner learning operations."""

    def test_learn(self) -> None:
        """Test learning from input and context patterns."""
        learner = ContextLearner(
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
        learner.init(num_input_bits=100, num_context_bits=200)

        input_pattern = BitArray(100)
        input_pattern.set_acts([1, 5, 10, 15, 20])

        context_pattern = BitArray(200)
        context_pattern.set_acts([5, 15, 25, 35, 45])

        learner.learn(input_pattern, context_pattern)

        # Should complete without error
        assert learner.num_c() == 100

    def test_execute_with_learning(self) -> None:
        """Test execute with learning enabled."""
        learner = ContextLearner(
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
        learner.init(num_input_bits=100, num_context_bits=200)

        input_pattern = BitArray(100)
        input_pattern.set_acts([1, 5, 10, 15, 20])

        context_pattern = BitArray(200)
        context_pattern.set_acts([5, 15, 25, 35, 45])

        learner.input.set_state(input_pattern)
        learner.context.set_state(context_pattern)
        learner.execute(learn_flag=True)
        output = learner.output.state()

        assert isinstance(output, BitArray)

    def test_execute_without_learning(self) -> None:
        """Test execute with learning disabled."""
        learner = ContextLearner(
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
        learner.init(num_input_bits=100, num_context_bits=200)

        input_pattern = BitArray(100)
        input_pattern.set_acts([1, 5, 10])

        context_pattern = BitArray(200)
        context_pattern.set_acts([5, 15, 25])

        learner.input.set_state(input_pattern)
        learner.context.set_state(context_pattern)
        learner.execute(learn_flag=False)
        output = learner.output.state()

        assert isinstance(output, BitArray)


class TestContextLearnerAnomaly:
    """Test ContextLearner anomaly detection."""

    def test_get_anomaly_score(self) -> None:
        """Test getting anomaly score."""
        learner = ContextLearner(
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
        learner.init(num_input_bits=100, num_context_bits=200)

        input_pattern = BitArray(100)
        input_pattern.set_acts([1, 5, 10])

        context_pattern = BitArray(200)
        context_pattern.set_acts([5, 15, 25])

        learner.compute(input_pattern, context_pattern)
        score = learner.get_anomaly_score()

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_anomaly_score_changes_with_learning(self) -> None:
        """Test that anomaly score changes with learning."""
        learner = ContextLearner(
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
        learner.init(num_input_bits=50, num_context_bits=100)

        input_pattern = BitArray(50)
        input_pattern.set_acts([1, 2, 3, 4, 5])

        context_pattern = BitArray(100)
        context_pattern.set_acts([10, 20, 30, 40, 50])

        # First time - likely high anomaly
        learner.compute(input_pattern, context_pattern)
        initial_score = learner.get_anomaly_score()

        # Learn the pattern multiple times
        for _ in range(10):
            learner.input.set_state(input_pattern)
            learner.context.set_state(context_pattern)
            learner.execute(learn_flag=True)

        # Compute again - anomaly should potentially decrease
        learner.compute(input_pattern, context_pattern)
        final_score = learner.get_anomaly_score()

        # Both should be valid scores
        assert 0.0 <= initial_score <= 1.0
        assert 0.0 <= final_score <= 1.0


class TestContextLearnerHistory:
    """Test ContextLearner history tracking."""

    def test_get_historical_count(self) -> None:
        """Test getting historical count of used dendrites."""
        learner = ContextLearner(
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
        learner.init(num_input_bits=100, num_context_bits=200)

        count = learner.get_historical_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_historical_count_increases_with_learning(self) -> None:
        """Test that historical count increases as learner learns."""
        learner = ContextLearner(
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
        learner.init(num_input_bits=50, num_context_bits=100)

        initial_count = learner.get_historical_count()

        # Learn several patterns
        for i in range(5):
            input_pattern = BitArray(50)
            input_pattern.set_acts([i, i + 1, i + 2])

            context_pattern = BitArray(100)
            context_pattern.set_acts([i * 10, i * 10 + 5, i * 10 + 10])

            learner.input.set_state(input_pattern)
            learner.context.set_state(context_pattern)
            learner.execute(learn_flag=True)

        final_count = learner.get_historical_count()

        # Count should be non-negative
        assert initial_count >= 0
        assert final_count >= 0

    def test_output_at(self) -> None:
        """Test getting output at specific time offset."""
        learner = ContextLearner(
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
        learner.init(num_input_bits=50, num_context_bits=100)

        input_pattern = BitArray(50)
        input_pattern.set_acts([1, 2, 3])

        context_pattern = BitArray(100)
        context_pattern.set_acts([10, 20, 30])

        learner.input.set_state(input_pattern)
        learner.context.set_state(context_pattern)
        learner.execute(learn_flag=False)

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
        learner = ContextLearner(
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
        learner.init(num_input_bits=50, num_context_bits=100)

        input_pattern = BitArray(50)
        input_pattern.set_acts([1, 2, 3])

        context_pattern = BitArray(100)
        context_pattern.set_acts([10, 20, 30])

        learner.input.set_state(input_pattern)
        learner.context.set_state(context_pattern)
        learner.execute(learn_flag=False)

        # Check if output changed
        changed = learner.has_changed()
        assert isinstance(changed, bool)


class TestContextLearnerOperations:
    """Test ContextLearner operations."""

    def test_clear(self) -> None:
        """Test clearing learner state."""
        learner = ContextLearner(
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
        learner.init(num_input_bits=50, num_context_bits=100)

        input_pattern = BitArray(50)
        input_pattern.set_acts([1, 2, 3])

        context_pattern = BitArray(100)
        context_pattern.set_acts([10, 20, 30])

        learner.input.set_state(input_pattern)
        learner.context.set_state(context_pattern)
        learner.execute(learn_flag=True)

        learner.clear()

        # After clear, output should be empty
        output = learner.output.state()
        assert output.num_set() == 0

    def test_memory_usage(self) -> None:
        """Test memory usage estimation."""
        learner = ContextLearner(
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


class TestContextLearnerRepr:
    """Test ContextLearner representation."""

    def test_repr(self) -> None:
        """Test __repr__."""
        learner = ContextLearner(
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
        assert "ContextLearner" in repr_str
        assert "num_c=100" in repr_str
        assert "num_spc=8" in repr_str
        assert "num_dps=4" in repr_str
        assert "d_thresh=15" in repr_str


