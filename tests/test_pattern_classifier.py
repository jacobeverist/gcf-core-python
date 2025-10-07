"""Tests for PatternClassifier class."""

from gnomics import BitArray, PatternClassifier


class TestPatternClassifierConstruction:
    """Test PatternClassifier construction."""

    def test_new(self) -> None:
        """Test creating a new PatternClassifier."""
        classifier = PatternClassifier(
            num_l=3,
            num_s=90,
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
        assert classifier is not None
        assert classifier.num_l() == 3
        assert classifier.num_s() == 90
        assert classifier.num_as() == 10
        assert classifier.num_spl() == 30  # num_s / num_l


class TestPatternClassifierInitialization:
    """Test PatternClassifier initialization."""

    def test_init(self) -> None:
        """Test initializing PatternClassifier."""
        classifier = PatternClassifier(
            num_l=4,
            num_s=100,
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
        classifier.init(num_i=512)
        # After init, classifier should be ready to compute
        assert classifier.num_l() == 4
        assert classifier.num_s() == 100


class TestPatternClassifierCompute:
    """Test PatternClassifier compute operations."""

    def test_compute_basic(self) -> None:
        """Test basic compute operation."""
        classifier = PatternClassifier(
            num_l=3,
            num_s=90,
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
        classifier.init(num_i=256)

        input_pattern = BitArray(256)
        input_pattern.set_acts([1, 5, 10, 20, 30])

        classifier.compute(input_pattern)
        output = classifier.output()

        assert isinstance(output, BitArray)
        assert len(output) == 90


class TestPatternClassifierPrediction:
    """Test PatternClassifier prediction operations."""

    def test_get_probabilities(self) -> None:
        """Test getting probability distribution."""
        classifier = PatternClassifier(
            num_l=3,
            num_s=90,
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
        classifier.init(num_i=256)

        input_pattern = BitArray(256)
        input_pattern.set_acts([1, 5, 10])

        classifier.compute(input_pattern)
        probs = classifier.get_probabilities()

        assert isinstance(probs, list)
        assert len(probs) == 3
        assert all(isinstance(p, float) for p in probs)

    def test_get_predicted_label(self) -> None:
        """Test getting predicted label."""
        classifier = PatternClassifier(
            num_l=4,
            num_s=100,
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
        classifier.init(num_i=256)

        input_pattern = BitArray(256)
        input_pattern.set_acts([1, 5, 10])

        classifier.compute(input_pattern)
        pred = classifier.get_predicted_label()

        assert isinstance(pred, int)
        assert 0 <= pred < 4

    def test_get_labels(self) -> None:
        """Test getting all label indices."""
        classifier = PatternClassifier(
            num_l=3,
            num_s=90,
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
        labels = classifier.get_labels()

        assert isinstance(labels, list)
        assert labels == [0, 1, 2]


class TestPatternClassifierLearning:
    """Test PatternClassifier learning operations."""

    def test_set_label(self) -> None:
        """Test setting ground truth label."""
        classifier = PatternClassifier(
            num_l=3,
            num_s=90,
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
        classifier.init(num_i=256)

        # Should complete without error
        classifier.set_label(0)
        classifier.set_label(1)
        classifier.set_label(2)

    def test_learn(self) -> None:
        """Test learning from an input pattern."""
        classifier = PatternClassifier(
            num_l=3,
            num_s=90,
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
        classifier.init(num_i=256)

        input_pattern = BitArray(256)
        input_pattern.set_acts([1, 5, 10, 15, 20])

        classifier.set_label(0)
        classifier.learn(input_pattern)

        # Should complete without error
        assert classifier.num_l() == 3

    def test_execute_with_learning(self) -> None:
        """Test execute with learning enabled."""
        classifier = PatternClassifier(
            num_l=3,
            num_s=90,
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
        classifier.init(num_i=256)

        input_pattern = BitArray(256)
        input_pattern.set_acts([1, 5, 10, 15, 20])

        classifier.set_label(1)
        classifier.execute(input_pattern, learn_flag=True)
        output = classifier.output()

        assert isinstance(output, BitArray)


class TestPatternClassifierOperations:
    """Test PatternClassifier operations."""

    def test_clear(self) -> None:
        """Test clearing classifier state."""
        classifier = PatternClassifier(
            num_l=3,
            num_s=90,
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
        classifier.init(num_i=256)

        input_pattern = BitArray(256)
        input_pattern.set_acts([1, 5, 10])
        classifier.execute(input_pattern, learn_flag=False)

        classifier.clear()

        # After clear, output should be empty
        output = classifier.output()
        assert output.num_set() == 0

    def test_memory_usage(self) -> None:
        """Test memory usage estimation."""
        classifier = PatternClassifier(
            num_l=3,
            num_s=90,
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
        usage = classifier.memory_usage()
        assert isinstance(usage, int)
        assert usage > 0


class TestPatternClassifierRepr:
    """Test PatternClassifier representation."""

    def test_repr(self) -> None:
        """Test __repr__."""
        classifier = PatternClassifier(
            num_l=3,
            num_s=90,
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
        repr_str = repr(classifier)
        assert "PatternClassifier" in repr_str
        assert "num_l=3" in repr_str
        assert "num_s=90" in repr_str
        assert "num_as=10" in repr_str
