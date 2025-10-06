"""Tests for PatternPooler class."""

from gcf_core_python_client import BitArray, PatternPooler


class TestPatternPoolerConstruction:
    """Test PatternPooler construction."""

    def test_new(self) -> None:
        """Test creating a new PatternPooler."""
        pooler = PatternPooler(
            num_s=100,
            num_as=10,
            perm_thr=20,
            perm_inc=2,
            perm_dec=1,
            pct_pool=0.8,
            pct_conn=0.5,
            pct_learn=0.3,
            num_t=2,
            seed=42,
        )
        assert pooler is not None
        assert pooler.num_s() == 100
        assert pooler.num_as() == 10
        assert pooler.perm_thr() == 20


class TestPatternPoolerInitialization:
    """Test PatternPooler initialization."""

    def test_init(self) -> None:
        """Test initializing PatternPooler."""
        pooler = PatternPooler(
            num_s=50,
            num_as=5,
            perm_thr=20,
            perm_inc=2,
            perm_dec=1,
            pct_pool=0.8,
            pct_conn=0.5,
            pct_learn=0.3,
            num_t=2,
            seed=42,
        )
        pooler.init(num_i=1024)
        # After init, pooler should be ready to compute
        assert pooler.num_s() == 50
        assert pooler.num_as() == 5


class TestPatternPoolerCompute:
    """Test PatternPooler compute operations."""

    def test_compute_basic(self) -> None:
        """Test basic compute operation."""
        pooler = PatternPooler(
            num_s=100,
            num_as=10,
            perm_thr=20,
            perm_inc=2,
            perm_dec=1,
            pct_pool=0.8,
            pct_conn=0.5,
            pct_learn=0.3,
            num_t=2,
            seed=42,
        )
        pooler.init(num_i=512)

        input_pattern = BitArray(512)
        input_pattern.set_acts([1, 5, 10, 20, 30, 40])

        pooler.compute(input_pattern)
        output = pooler.output()

        assert isinstance(output, BitArray)
        assert len(output) == 100


class TestPatternPoolerLearning:
    """Test PatternPooler learning operations."""

    def test_learn(self) -> None:
        """Test learning from an input pattern."""
        pooler = PatternPooler(
            num_s=50,
            num_as=5,
            perm_thr=20,
            perm_inc=2,
            perm_dec=1,
            pct_pool=0.8,
            pct_conn=0.5,
            pct_learn=0.3,
            num_t=2,
            seed=42,
        )
        pooler.init(num_i=256)

        input_pattern = BitArray(256)
        input_pattern.set_acts([1, 5, 10, 15, 20])

        # Learn from pattern
        pooler.learn(input_pattern)

        # Should complete without error
        assert pooler.num_s() == 50

    def test_execute_with_learning(self) -> None:
        """Test execute with learning enabled."""
        pooler = PatternPooler(
            num_s=50,
            num_as=5,
            perm_thr=20,
            perm_inc=2,
            perm_dec=1,
            pct_pool=0.8,
            pct_conn=0.5,
            pct_learn=0.3,
            num_t=2,
            seed=42,
        )
        pooler.init(num_i=256)

        input_pattern = BitArray(256)
        input_pattern.set_acts([1, 5, 10, 15, 20])

        pooler.execute(input_pattern, learn_flag=True)
        output = pooler.output()

        assert isinstance(output, BitArray)


class TestPatternPoolerHistory:
    """Test PatternPooler history tracking."""

    def test_output_at(self) -> None:
        """Test accessing historical outputs."""
        pooler = PatternPooler(
            num_s=50,
            num_as=5,
            perm_thr=20,
            perm_inc=2,
            perm_dec=1,
            pct_pool=0.8,
            pct_conn=0.5,
            pct_learn=0.3,
            num_t=3,
            seed=42,
        )
        pooler.init(num_i=256)

        input_pattern = BitArray(256)
        input_pattern.set_acts([1, 5, 10])

        pooler.execute(input_pattern, learn_flag=False)

        # Access current output
        current = pooler.output_at(0)
        assert isinstance(current, BitArray)

        # Access previous output
        previous = pooler.output_at(1)
        assert isinstance(previous, BitArray)


class TestPatternPoolerOperations:
    """Test PatternPooler operations."""

    def test_clear(self) -> None:
        """Test clearing pooler state."""
        pooler = PatternPooler(
            num_s=50,
            num_as=5,
            perm_thr=20,
            perm_inc=2,
            perm_dec=1,
            pct_pool=0.8,
            pct_conn=0.5,
            pct_learn=0.3,
            num_t=2,
            seed=42,
        )
        pooler.init(num_i=256)

        input_pattern = BitArray(256)
        input_pattern.set_acts([1, 5, 10])
        pooler.execute(input_pattern, learn_flag=False)

        pooler.clear()

        # After clear, output should be empty
        output = pooler.output()
        assert output.num_set() == 0

    def test_memory_usage(self) -> None:
        """Test memory usage estimation."""
        pooler = PatternPooler(
            num_s=50,
            num_as=5,
            perm_thr=20,
            perm_inc=2,
            perm_dec=1,
            pct_pool=0.8,
            pct_conn=0.5,
            pct_learn=0.3,
            num_t=2,
            seed=42,
        )
        usage = pooler.memory_usage()
        assert isinstance(usage, int)
        assert usage > 0


class TestPatternPoolerRepr:
    """Test PatternPooler representation."""

    def test_repr(self) -> None:
        """Test __repr__."""
        pooler = PatternPooler(
            num_s=100,
            num_as=10,
            perm_thr=20,
            perm_inc=2,
            perm_dec=1,
            pct_pool=0.8,
            pct_conn=0.5,
            pct_learn=0.3,
            num_t=2,
            seed=42,
        )
        repr_str = repr(pooler)
        assert "PatternPooler" in repr_str
        assert "num_s=100" in repr_str
        assert "num_as=10" in repr_str
        assert "perm_thr=20" in repr_str
