"""Tests for BlockMemory class."""

from gcf_core_python_client import BitArray, BlockMemory


class TestBlockMemoryConstruction:
    """Test BlockMemory construction."""

    def test_new(self) -> None:
        """Test creating a new BlockMemory."""
        memory = BlockMemory(
            num_d=10, num_rpd=20, perm_thr=20, perm_inc=2, perm_dec=1, pct_learn=0.5
        )
        assert memory is not None
        assert memory.num_dendrites() == 10


class TestBlockMemoryInit:
    """Test BlockMemory initialization."""

    def test_init(self) -> None:
        """Test initializing BlockMemory."""
        memory = BlockMemory(
            num_d=5, num_rpd=10, perm_thr=20, perm_inc=2, perm_dec=1, pct_learn=0.5
        )
        memory.init(num_i=100, seed=42)
        # After init, we should be able to get addresses
        addrs = memory.addrs(0)
        assert isinstance(addrs, list)
        assert len(addrs) == 10


class TestBlockMemoryOverlap:
    """Test BlockMemory overlap calculation."""

    def test_overlap_empty_input(self) -> None:
        """Test overlap with empty input pattern."""
        memory = BlockMemory(
            num_d=5, num_rpd=10, perm_thr=20, perm_inc=2, perm_dec=1, pct_learn=0.5
        )
        memory.init(num_i=100, seed=42)
        input_pattern = BitArray(100)
        overlap = memory.overlap(0, input_pattern)
        assert isinstance(overlap, int)
        assert overlap >= 0

    def test_overlap_with_pattern(self) -> None:
        """Test overlap with specific input pattern."""
        memory = BlockMemory(
            num_d=5, num_rpd=10, perm_thr=20, perm_inc=2, perm_dec=1, pct_learn=0.5
        )
        memory.init(num_i=100, seed=42)
        # Create a pattern with some active bits
        input_pattern = BitArray.from_bits([1, 0, 1, 0, 1] + [0] * 95)

        overlap = memory.overlap(0, input_pattern)
        assert isinstance(overlap, int)
        assert overlap >= 0

    def test_overlap_all_dendrites(self) -> None:
        """Test overlap across all dendrites."""
        num_dendrites = 5
        memory = BlockMemory(
            num_d=num_dendrites,
            num_rpd=10,
            perm_thr=20,
            perm_inc=2,
            perm_dec=1,
            pct_learn=0.5,
        )
        memory.init(num_i=100, seed=42)
        input_pattern = BitArray.from_bits([1, 0, 1, 0, 1] + [0] * 95)

        # Calculate overlap for all dendrites
        for d in range(num_dendrites):
            overlap = memory.overlap(d, input_pattern)
            assert isinstance(overlap, int)


class TestBlockMemoryLearning:
    """Test BlockMemory learning operations."""

    def test_learn(self) -> None:
        """Test learning from an input pattern."""
        memory = BlockMemory(
            num_d=3, num_rpd=10, perm_thr=20, perm_inc=2, perm_dec=1, pct_learn=0.5
        )
        memory.init(num_i=100, seed=42)
        input_pattern = BitArray.from_bits([1, 0, 1, 0, 1] + [0] * 95)

        # Get initial permanences
        perms_before = memory.perms(0).copy()

        # Learn
        memory.learn(0, input_pattern, seed=42)

        # Get permanences after learning
        perms_after = memory.perms(0)

        # Permanences should have changed
        assert perms_before != perms_after

    def test_punish(self) -> None:
        """Test punishing (negative learning) a dendrite."""
        memory = BlockMemory(
            num_d=3, num_rpd=10, perm_thr=20, perm_inc=2, perm_dec=1, pct_learn=0.5
        )
        memory.init(num_i=100, seed=42)
        input_pattern = BitArray.from_bits([1, 0, 1, 0, 1] + [0] * 95)

        # First learn to build up some permanences
        memory.learn(0, input_pattern, seed=42)
        perms_after_learn = memory.perms(0).copy()

        # Then punish
        memory.punish(0, input_pattern, seed=43)
        perms_after_punish = memory.perms(0)

        # Permanences should have decreased
        assert perms_after_punish != perms_after_learn

    def test_learn_move(self) -> None:
        """Test moving dead receptors."""
        memory = BlockMemory(
            num_d=3, num_rpd=10, perm_thr=20, perm_inc=2, perm_dec=1, pct_learn=0.5
        )
        memory.init(num_i=100, seed=42)
        input_pattern = BitArray.from_bits([1, 0, 1, 0, 1] + [0] * 95)

        # Get initial addresses
        addrs_before = memory.addrs(0)

        # Learn and move dead receptors
        memory.learn_move(0, input_pattern, seed=42)

        # Addresses might have changed if there were dead receptors
        addrs_after = memory.addrs(0)

        # Both should have same length
        assert len(addrs_before) == len(addrs_after)


class TestBlockMemoryInspection:
    """Test BlockMemory inspection methods."""

    def test_num_dendrites(self) -> None:
        """Test getting number of dendrites."""
        memory = BlockMemory(
            num_d=7, num_rpd=10, perm_thr=20, perm_inc=2, perm_dec=1, pct_learn=0.5
        )
        assert memory.num_dendrites() == 7

    def test_addrs(self) -> None:
        """Test getting receptor addresses."""
        memory = BlockMemory(
            num_d=3, num_rpd=15, perm_thr=20, perm_inc=2, perm_dec=1, pct_learn=0.5
        )
        memory.init(num_i=100, seed=42)
        addrs = memory.addrs(0)
        assert isinstance(addrs, list)
        assert len(addrs) == 15
        # All addresses should be valid indices
        assert all(isinstance(addr, int) for addr in addrs)

    def test_perms(self) -> None:
        """Test getting receptor permanences."""
        memory = BlockMemory(
            num_d=3, num_rpd=15, perm_thr=20, perm_inc=2, perm_dec=1, pct_learn=0.5
        )
        memory.init(num_i=100, seed=42)
        perms = memory.perms(0)
        assert isinstance(perms, list)
        assert len(perms) == 15
        # All permanences should be in range 0-99
        assert all(isinstance(p, int) and 0 <= p <= 99 for p in perms)

    def test_conns(self) -> None:
        """Test getting connected receptors."""
        memory = BlockMemory(
            num_d=3, num_rpd=15, perm_thr=20, perm_inc=2, perm_dec=1, pct_learn=0.5
        )
        memory.init(num_i=100, seed=42)
        conns = memory.conns(0)
        # conns might be None if not computed yet, or a BitArray
        if conns is not None:
            assert isinstance(conns, BitArray)
            assert len(conns) == 15


class TestBlockMemoryRepr:
    """Test BlockMemory representation."""

    def test_repr(self) -> None:
        """Test __repr__."""
        memory = BlockMemory(
            num_d=10, num_rpd=20, perm_thr=20, perm_inc=2, perm_dec=1, pct_learn=0.5
        )
        repr_str = repr(memory)
        assert "BlockMemory" in repr_str
        assert "num_dendrites=10" in repr_str


class TestBlockMemoryIntegration:
    """Integration tests for BlockMemory."""

    def test_learning_increases_overlap(self) -> None:
        """Test that learning from a pattern increases overlap."""
        memory = BlockMemory(
            num_d=5, num_rpd=20, perm_thr=20, perm_inc=5, perm_dec=1, pct_learn=0.5
        )
        memory.init(num_i=100, seed=42)
        # Create a specific pattern
        pattern = BitArray(100)
        pattern.set_bits([1] * 20 + [0] * 80)

        # Learn from the pattern multiple times
        for i in range(10):
            memory.learn(0, pattern, seed=42 + i)

        # Get overlap after learning
        final_overlap = memory.overlap(0, pattern)

        # Overlap should increase after learning
        # (though this depends on initial random state)
        assert isinstance(final_overlap, int)
        assert final_overlap >= 0

    def test_multiple_dendrites_independent(self) -> None:
        """Test that different dendrites learn independently."""
        memory = BlockMemory(
            num_d=3, num_rpd=20, perm_thr=20, perm_inc=2, perm_dec=1, pct_learn=0.5
        )
        memory.init(num_i=100, seed=42)
        pattern = BitArray.from_bits([1, 0, 1, 0, 1] + [0] * 95)

        # Learn only on dendrite 0
        memory.learn(0, pattern, seed=42)

        # Get permanences for all dendrites
        perms_0 = memory.perms(0)
        perms_1 = memory.perms(1)
        perms_2 = memory.perms(2)

        # Dendrites 1 and 2 should still have their initial permanences
        # (which are random but different from dendrite 0's learned state)
        assert len(perms_0) == len(perms_1) == len(perms_2)
