//! Python bindings for gnomics::BitArray

use gnomics::BitArray as RustBitArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

/// A high-performance binary array for sparse distributed representations.
///
/// BitArray efficiently stores and manipulates binary patterns (vectors of 1s and 0s)
/// using compact 32-bit word storage, providing 32× compression compared to byte arrays.
#[pyclass(name = "BitArray", module = "gnomics")]
#[derive(Clone)]
pub struct PyBitArray {
    inner: RustBitArray,
}

#[pymethods]
impl PyBitArray {
    /// Create a new BitArray with n bits, all initialized to 0.
    ///
    /// Args:
    ///     n: Number of bits in the array
    ///
    /// Returns:
    ///     A new BitArray instance
    #[new]
    pub fn new(n: usize) -> Self {
        PyBitArray {
            inner: RustBitArray::new(n),
        }
    }

    /// Create a BitArray from a list of bit values (0s and 1s).
    ///
    /// Args:
    ///     bits: List of integers (0 or 1)
    ///
    /// Returns:
    ///     A new BitArray instance
    #[staticmethod]
    pub fn from_bits(bits: &Bound<'_, PyList>) -> PyResult<Self> {
        let vec: Vec<u8> = bits
            .iter()
            .map(|item| {
                let val = item.extract::<u8>()?;
                if val > 1 {
                    return Err(PyValueError::new_err(format!(
                        "Bit values must be 0 or 1, got {}",
                        val
                    )));
                }
                Ok(val)
            })
            .collect::<PyResult<Vec<u8>>>()?;

        let mut bitarray = RustBitArray::new(vec.len());
        bitarray.set_bits(&vec);
        Ok(PyBitArray { inner: bitarray })
    }

    /// Create a BitArray from a list of active bit indices.
    ///
    /// Args:
    ///     size: Total number of bits in the array
    ///     indices: List of indices of bits to set to 1
    ///
    /// Returns:
    ///     A new BitArray instance
    #[staticmethod]
    pub fn from_indices(size: usize, indices: &Bound<'_, PyList>) -> PyResult<Self> {
        let idxs: Vec<usize> = indices.extract()?;
        let mut bitarray = RustBitArray::new(size);
        bitarray.set_acts(&idxs);
        Ok(PyBitArray { inner: bitarray })
    }

    /// Resize the BitArray to n bits.
    ///
    /// Args:
    ///     n: New number of bits
    pub fn resize(&mut self, n: usize) {
        self.inner.resize(n);
    }

    /// Clear all storage and reset size to 0.
    pub fn erase(&mut self) {
        self.inner.erase();
    }

    /// Set a single bit to 1.
    ///
    /// Args:
    ///     b: Index of the bit to set
    pub fn set_bit(&mut self, b: usize) {
        self.inner.set_bit(b);
    }

    /// Get the value of a single bit.
    ///
    /// Args:
    ///     b: Index of the bit to get
    ///
    /// Returns:
    ///     0 or 1
    pub fn get_bit(&self, b: usize) -> u8 {
        self.inner.get_bit(b)
    }

    /// Clear a single bit to 0.
    ///
    /// Args:
    ///     b: Index of the bit to clear
    pub fn clear_bit(&mut self, b: usize) {
        self.inner.clear_bit(b);
    }

    /// Toggle a single bit (0 -> 1, 1 -> 0).
    ///
    /// Args:
    ///     b: Index of the bit to toggle
    pub fn toggle_bit(&mut self, b: usize) {
        self.inner.toggle_bit(b);
    }

    /// Assign a value to a single bit.
    ///
    /// Args:
    ///     b: Index of the bit
    ///     val: Value to assign (0 or 1)
    pub fn assign_bit(&mut self, b: usize, val: u8) -> PyResult<()> {
        if val > 1 {
            return Err(PyValueError::new_err(format!(
                "Bit value must be 0 or 1, got {}",
                val
            )));
        }
        self.inner.assign_bit(b, val);
        Ok(())
    }

    /// Set a range of bits to 1.
    ///
    /// Args:
    ///     beg: Starting index
    ///     len: Number of bits to set
    pub fn set_range(&mut self, beg: usize, len: usize) {
        self.inner.set_range(beg, len);
    }

    /// Clear a range of bits to 0.
    ///
    /// Args:
    ///     beg: Starting index
    ///     len: Number of bits to clear
    pub fn clear_range(&mut self, beg: usize, len: usize) {
        self.inner.clear_range(beg, len);
    }

    /// Toggle a range of bits.
    ///
    /// Args:
    ///     beg: Starting index
    ///     len: Number of bits to toggle
    pub fn toggle_range(&mut self, beg: usize, len: usize) {
        self.inner.toggle_range(beg, len);
    }

    /// Set all bits to 1.
    pub fn set_all(&mut self) {
        self.inner.set_all();
    }

    /// Clear all bits to 0.
    pub fn clear_all(&mut self) {
        self.inner.clear_all();
    }

    /// Toggle all bits.
    pub fn toggle_all(&mut self) {
        self.inner.toggle_all();
    }

    /// Set bits from a list of values.
    ///
    /// Args:
    ///     vals: List of bit values (0s and 1s)
    pub fn set_bits(&mut self, vals: &Bound<'_, PyList>) -> PyResult<()> {
        let vec: Vec<u8> = vals
            .iter()
            .map(|item| {
                let val = item.extract::<u8>()?;
                if val > 1 {
                    return Err(PyValueError::new_err(format!(
                        "Bit values must be 0 or 1, got {}",
                        val
                    )));
                }
                Ok(val)
            })
            .collect::<PyResult<Vec<u8>>>()?;

        self.inner.set_bits(&vec);
        Ok(())
    }

    /// Set bits at specified indices to 1.
    ///
    /// Args:
    ///     idxs: List of indices of bits to set
    pub fn set_acts(&mut self, idxs: &Bound<'_, PyList>) -> PyResult<()> {
        let indices: Vec<usize> = idxs.extract()?;
        self.inner.set_acts(&indices);
        Ok(())
    }

    /// Get all bit values as a list.
    ///
    /// Returns:
    ///     List of bit values (0s and 1s)
    pub fn get_bits(&self) -> Vec<u8> {
        self.inner.get_bits()
    }

    /// Get indices of all bits that are set to 1.
    ///
    /// Returns:
    ///     List of indices
    pub fn get_acts(&self) -> Vec<usize> {
        self.inner.get_acts()
    }

    /// Count the number of bits set to 1.
    ///
    /// Returns:
    ///     Number of set bits
    pub fn num_set(&self) -> usize {
        self.inner.num_set()
    }

    /// Count the number of bits set to 0.
    ///
    /// Returns:
    ///     Number of cleared bits
    pub fn num_cleared(&self) -> usize {
        self.inner.num_cleared()
    }

    /// Count the number of bits with the same value in both BitArrays.
    ///
    /// Args:
    ///     other: Another BitArray to compare with
    ///
    /// Returns:
    ///     Number of matching bits
    pub fn num_similar(&self, other: &PyBitArray) -> usize {
        self.inner.num_similar(&other.inner)
    }

    /// Find the next set bit starting from a given position.
    ///
    /// Args:
    ///     beg: Starting index
    ///
    /// Returns:
    ///     Index of the next set bit, or None if no set bit is found
    pub fn find_next_set_bit(&self, beg: usize) -> Option<usize> {
        self.inner.find_next_set_bit(beg)
    }

    // Python special methods

    fn __len__(&self) -> usize {
        self.inner.num_bits()
    }

    fn __getitem__(&self, index: isize) -> u8 {
        let len = self.inner.num_bits() as isize;
        let idx = if index < 0 {
            (len + index) as usize
        } else {
            index as usize
        };
        self.inner.get_bit(idx)
    }

    fn __repr__(&self) -> String {
        let bits = self.inner.get_bits();
        let preview: Vec<String> = bits.iter().take(20).map(|b| b.to_string()).collect();
        let suffix = if bits.len() > 20 { ", ..." } else { "" };
        format!(
            "BitArray([{}{}], num_set={}, len={})",
            preview.join(", "),
            suffix,
            self.inner.num_set(),
            self.inner.num_bits()
        )
    }

    fn __str__(&self) -> String {
        let bits = self.inner.get_bits();
        let bit_str: String = bits.iter().map(|b| b.to_string()).collect();
        bit_str
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<PyBitArrayIterator> {
        Ok(PyBitArrayIterator {
            bits: slf.inner.get_bits(),
            index: 0,
        })
    }

    // Bitwise operations

    fn __and__(&self, other: &PyBitArray) -> PyBitArray {
        let result = &self.inner & &other.inner;
        PyBitArray { inner: result }
    }

    fn __or__(&self, other: &PyBitArray) -> PyBitArray {
        let result = &self.inner | &other.inner;
        PyBitArray { inner: result }
    }

    fn __xor__(&self, other: &PyBitArray) -> PyBitArray {
        let result = &self.inner ^ &other.inner;
        PyBitArray { inner: result }
    }

    fn __invert__(&self) -> PyBitArray {
        let result = !&self.inner;
        PyBitArray { inner: result }
    }

    /// Convert the BitArray to a numpy array.
    ///
    /// Returns:
    ///     A numpy array of dtype uint8
    pub fn to_numpy(&self, py: Python<'_>) -> PyResult<PyObject> {
        let numpy = py.import_bound("numpy")?;
        let bits = self.inner.get_bits();
        let array = numpy.call_method1("array", (bits,))?;
        Ok(array.into())
    }

    /// Create a BitArray from a numpy array.
    ///
    /// Args:
    ///     arr: A numpy array of dtype uint8 or int
    ///
    /// Returns:
    ///     A new BitArray instance
    #[staticmethod]
    pub fn from_numpy(arr: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Convert to list and use from_bits
        let list = arr.call_method0("tolist")?;
        let py_list = list.downcast::<PyList>()?;
        Self::from_bits(py_list)
    }
}

/// Iterator for BitArray
#[pyclass]
struct PyBitArrayIterator {
    bits: Vec<u8>,
    index: usize,
}

#[pymethods]
impl PyBitArrayIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<u8> {
        if slf.index < slf.bits.len() {
            let result = slf.bits[slf.index];
            slf.index += 1;
            Some(result)
        } else {
            None
        }
    }
}

// Helper methods for internal use by other modules
impl PyBitArray {
    pub(crate) fn from_rust(bitarray: RustBitArray) -> Self {
        PyBitArray { inner: bitarray }
    }

    pub(crate) fn as_rust(&self) -> &RustBitArray {
        &self.inner
    }
}
