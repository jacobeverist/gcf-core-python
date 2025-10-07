//! Python bindings for gnomics::BlockOutput

use crate::bitarray::PyBitArray;
use gnomics::BlockOutput as RustBlockOutput;
use pyo3::prelude::*;

/// Manages block outputs with history tracking and change detection.
///
/// BlockOutput maintains a circular buffer of historical states and efficiently
/// tracks when outputs change, enabling optimizations in downstream blocks.
#[pyclass(name = "BlockOutput", module = "gnomics.core")]
pub struct PyBlockOutput {
    inner: RustBlockOutput,
}

#[pymethods]
impl PyBlockOutput {
    /// Create a new BlockOutput.
    ///
    /// The output must be set up with specific dimensions before use.
    #[new]
    pub fn new() -> Self {
        PyBlockOutput {
            inner: RustBlockOutput::new(),
        }
    }

    /// Set up the BlockOutput with specific dimensions.
    ///
    /// Args:
    ///     num_t: Number of historical time steps to maintain (minimum 2)
    ///     num_b: Number of bits in the output state
    pub fn setup(&mut self, num_t: usize, num_b: usize) {
        self.inner.setup(num_t, num_b);
    }

    /// Advance to the next time step in the circular buffer.
    pub fn step(&mut self) {
        self.inner.step();
    }

    /// Store the current state and detect if it has changed.
    ///
    /// This compares the current state with the previous state to set
    /// the internal change flag.
    pub fn store(&mut self) {
        self.inner.store();
    }

    /// Check if the output has changed in the current time step.
    ///
    /// Returns:
    ///     True if the current state differs from the previous state
    pub fn has_changed(&self) -> bool {
        self.inner.has_changed()
    }

    /// Check if the output changed at a specific time offset.
    ///
    /// Args:
    ///     time: Time offset (0 = current, 1 = previous, etc.)
    ///
    /// Returns:
    ///     True if the state changed at that time
    pub fn has_changed_at(&self, time: usize) -> bool {
        self.inner.has_changed_at(time)
    }

    /// Get the current output state.
    ///
    /// Returns:
    ///     BitArray containing the current state
    pub fn state(&self) -> PyBitArray {
        PyBitArray::from_rust(self.inner.state.clone())
    }

    /// Set the current output state from a BitArray.
    ///
    /// Args:
    ///     bits: BitArray to set as the current state
    pub fn set_state(&mut self, bits: &PyBitArray) {
        self.inner.state = bits.as_rust().clone();
    }

    /// Get the output state at a specific time offset.
    ///
    /// Args:
    ///     time: Time offset (0 = current, 1 = previous, etc.)
    ///
    /// Returns:
    ///     BitArray containing the state at that time
    pub fn get_bitarray(&self, time: usize) -> PyBitArray {
        PyBitArray::from_rust(self.inner.get_bitarray(time).clone())
    }

    /// Get the number of time steps in history.
    ///
    /// Returns:
    ///     Number of time steps
    pub fn num_t(&self) -> usize {
        self.inner.num_t()
    }

    /// Get the unique output ID.
    ///
    /// Returns:
    ///     Output ID
    pub fn id(&self) -> u32 {
        self.inner.id()
    }

    /// Clear all state to zeros.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    fn __repr__(&self) -> String {
        format!(
            "BlockOutput(num_bits={}, num_t={}, changed={})",
            self.inner.state.num_bits(),
            self.inner.num_t(),
            self.inner.has_changed()
        )
    }
}

