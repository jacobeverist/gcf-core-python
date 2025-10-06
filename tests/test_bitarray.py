"""Tests for BitArray class."""

import numpy as np
import pytest
from gcf_core_python_client import BitArray


class TestBitArrayConstruction:
    """Test BitArray construction methods."""

    def test_new(self) -> None:
        """Test creating a new BitArray."""
        ba = BitArray(10)
        assert len(ba) == 10
        assert ba.num_set() == 0
        assert ba.num_cleared() == 10

    def test_from_bits(self) -> None:
        """Test creating BitArray from list of bits."""
        ba = BitArray.from_bits([1, 0, 1, 0, 1])
        assert len(ba) == 5
        assert ba.num_set() == 3
        assert ba.get_bits() == [1, 0, 1, 0, 1]

    def test_from_bits_invalid(self) -> None:
        """Test that from_bits validates bit values."""
        with pytest.raises(ValueError, match="Bit values must be 0 or 1"):
            BitArray.from_bits([1, 0, 2])

    def test_from_indices(self) -> None:
        """Test creating BitArray from active indices."""
        ba = BitArray.from_indices(10, [0, 2, 4, 6])
        assert len(ba) == 10
        assert ba.num_set() == 4
        assert ba.get_acts() == [0, 2, 4, 6]

    def test_from_numpy(self) -> None:
        """Test creating BitArray from numpy array."""
        arr = np.array([1, 0, 1, 0, 1], dtype=np.uint8)
        ba = BitArray.from_numpy(arr)
        assert len(ba) == 5
        assert ba.num_set() == 3
        assert ba.get_bits() == [1, 0, 1, 0, 1]


class TestBitArraySingleBitOps:
    """Test single bit operations."""

    def test_set_bit(self) -> None:
        """Test setting a single bit."""
        ba = BitArray(10)
        ba.set_bit(5)
        assert ba.get_bit(5) == 1
        assert ba.num_set() == 1

    def test_get_bit(self) -> None:
        """Test getting a single bit."""
        ba = BitArray.from_bits([1, 0, 1])
        assert ba.get_bit(0) == 1
        assert ba.get_bit(1) == 0
        assert ba.get_bit(2) == 1

    def test_clear_bit(self) -> None:
        """Test clearing a single bit."""
        ba = BitArray.from_bits([1, 1, 1])
        ba.clear_bit(1)
        assert ba.get_bit(1) == 0
        assert ba.num_set() == 2

    def test_toggle_bit(self) -> None:
        """Test toggling a single bit."""
        ba = BitArray.from_bits([1, 0, 1])
        ba.toggle_bit(0)
        ba.toggle_bit(1)
        assert ba.get_bit(0) == 0
        assert ba.get_bit(1) == 1

    def test_assign_bit(self) -> None:
        """Test assigning a value to a single bit."""
        ba = BitArray(5)
        ba.assign_bit(2, 1)
        assert ba.get_bit(2) == 1
        ba.assign_bit(2, 0)
        assert ba.get_bit(2) == 0

    def test_assign_bit_invalid(self) -> None:
        """Test that assign_bit validates values."""
        ba = BitArray(5)
        with pytest.raises(ValueError, match="Bit value must be 0 or 1"):
            ba.assign_bit(0, 2)


class TestBitArrayRangeOps:
    """Test range operations."""

    def test_set_range(self) -> None:
        """Test setting a range of bits."""
        ba = BitArray(10)
        ba.set_range(2, 5)
        assert ba.get_bits() == [0, 0, 1, 1, 1, 1, 1, 0, 0, 0]

    def test_clear_range(self) -> None:
        """Test clearing a range of bits."""
        ba = BitArray.from_bits([1, 1, 1, 1, 1])
        ba.clear_range(1, 3)
        assert ba.get_bits() == [1, 0, 0, 0, 1]

    def test_toggle_range(self) -> None:
        """Test toggling a range of bits."""
        ba = BitArray.from_bits([1, 0, 1, 0, 1])
        ba.toggle_range(1, 3)
        assert ba.get_bits() == [1, 1, 0, 1, 1]


class TestBitArrayBulkOps:
    """Test bulk operations."""

    def test_set_all(self) -> None:
        """Test setting all bits."""
        ba = BitArray(10)
        ba.set_all()
        assert ba.num_set() == 10

    def test_clear_all(self) -> None:
        """Test clearing all bits."""
        ba = BitArray.from_bits([1, 1, 1, 1, 1])
        ba.clear_all()
        assert ba.num_set() == 0

    def test_toggle_all(self) -> None:
        """Test toggling all bits."""
        ba = BitArray.from_bits([1, 0, 1, 0, 1])
        ba.toggle_all()
        assert ba.get_bits() == [0, 1, 0, 1, 0]


class TestBitArrayVectorOps:
    """Test vector operations."""

    def test_set_bits(self) -> None:
        """Test setting bits from a list."""
        ba = BitArray(5)
        ba.set_bits([1, 0, 1, 0, 1])
        assert ba.get_bits() == [1, 0, 1, 0, 1]

    def test_set_acts(self) -> None:
        """Test setting bits from indices."""
        ba = BitArray(10)
        ba.set_acts([1, 3, 5, 7])
        assert ba.get_acts() == [1, 3, 5, 7]

    def test_get_bits(self) -> None:
        """Test getting all bits."""
        ba = BitArray.from_bits([1, 0, 1, 0, 1])
        assert ba.get_bits() == [1, 0, 1, 0, 1]

    def test_get_acts(self) -> None:
        """Test getting active indices."""
        ba = BitArray.from_bits([1, 0, 1, 0, 1])
        assert ba.get_acts() == [0, 2, 4]


class TestBitArrayCounting:
    """Test counting operations."""

    def test_num_set(self) -> None:
        """Test counting set bits."""
        ba = BitArray.from_bits([1, 0, 1, 0, 1])
        assert ba.num_set() == 3

    def test_num_cleared(self) -> None:
        """Test counting cleared bits."""
        ba = BitArray.from_bits([1, 0, 1, 0, 1])
        assert ba.num_cleared() == 2

    def test_num_similar(self) -> None:
        """Test counting similar bits (counts shared 1s, not all similar positions)."""
        ba1 = BitArray.from_bits([1, 0, 1, 0, 1])
        ba2 = BitArray.from_bits([1, 1, 1, 0, 0])
        # num_similar counts bits that are set in both arrays (shared 1s)
        # ba1 has 1s at positions 0, 2, 4
        # ba2 has 1s at positions 0, 1, 2
        # Shared 1s are at positions 0, 2
        assert ba1.num_similar(ba2) == 2


class TestBitArraySearch:
    """Test search operations."""

    def test_find_next_set_bit(self) -> None:
        """Test finding next set bit."""
        ba = BitArray.from_bits([0, 0, 1, 0, 1, 0])
        assert ba.find_next_set_bit(0) == 2
        assert ba.find_next_set_bit(3) == 4
        # find_next_set_bit wraps around to the beginning
        assert ba.find_next_set_bit(5) == 2


class TestBitArrayPythonProtocols:
    """Test Python special methods."""

    def test_len(self) -> None:
        """Test __len__."""
        ba = BitArray(42)
        assert len(ba) == 42

    def test_getitem(self) -> None:
        """Test __getitem__."""
        ba = BitArray.from_bits([1, 0, 1, 0, 1])
        assert ba[0] == 1
        assert ba[1] == 0
        assert ba[2] == 1
        assert ba[-1] == 1
        assert ba[-2] == 0

    def test_iter(self) -> None:
        """Test __iter__."""
        ba = BitArray.from_bits([1, 0, 1, 0, 1])
        assert list(ba) == [1, 0, 1, 0, 1]

    def test_repr(self) -> None:
        """Test __repr__."""
        ba = BitArray.from_bits([1, 0, 1])
        repr_str = repr(ba)
        assert "BitArray" in repr_str
        assert "num_set=2" in repr_str
        assert "len=3" in repr_str

    def test_str(self) -> None:
        """Test __str__."""
        ba = BitArray.from_bits([1, 0, 1, 0, 1])
        assert str(ba) == "10101"


class TestBitArrayBitwiseOps:
    """Test bitwise operations."""

    def test_and(self) -> None:
        """Test bitwise AND."""
        ba1 = BitArray.from_bits([1, 1, 0, 0])
        ba2 = BitArray.from_bits([1, 0, 1, 0])
        result = ba1 & ba2
        assert result.get_bits() == [1, 0, 0, 0]

    def test_or(self) -> None:
        """Test bitwise OR."""
        ba1 = BitArray.from_bits([1, 1, 0, 0])
        ba2 = BitArray.from_bits([1, 0, 1, 0])
        result = ba1 | ba2
        assert result.get_bits() == [1, 1, 1, 0]

    def test_xor(self) -> None:
        """Test bitwise XOR."""
        ba1 = BitArray.from_bits([1, 1, 0, 0])
        ba2 = BitArray.from_bits([1, 0, 1, 0])
        result = ba1 ^ ba2
        assert result.get_bits() == [0, 1, 1, 0]

    def test_invert(self) -> None:
        """Test bitwise NOT."""
        ba = BitArray.from_bits([1, 0, 1, 0, 1])
        result = ~ba
        assert result.get_bits() == [0, 1, 0, 1, 0]


class TestBitArrayNumpyIntegration:
    """Test numpy array integration."""

    def test_to_numpy(self) -> None:
        """Test converting to numpy array."""
        ba = BitArray.from_bits([1, 0, 1, 0, 1])
        arr = ba.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.tolist() == [1, 0, 1, 0, 1]

    def test_from_numpy_roundtrip(self) -> None:
        """Test round-trip conversion with numpy."""
        original = [1, 0, 1, 0, 1]
        ba1 = BitArray.from_bits(original)
        arr = ba1.to_numpy()
        ba2 = BitArray.from_numpy(arr)
        assert ba2.get_bits() == original


class TestBitArrayMutation:
    """Test that BitArray operations properly mutate or return new instances."""

    def test_resize(self) -> None:
        """Test resizing BitArray (clears all bits)."""
        ba = BitArray(10)
        ba.set_bit(5)
        ba.resize(20)
        assert len(ba) == 20
        # resize clears all bits
        assert ba.num_set() == 0

    def test_erase(self) -> None:
        """Test erasing BitArray."""
        ba = BitArray.from_bits([1, 0, 1, 0, 1])
        ba.erase()
        assert len(ba) == 0
