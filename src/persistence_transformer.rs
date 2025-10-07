//! Python bindings for gnomics::PersistenceTransformer

use crate::bitarray::PyBitArray;
use gnomics::{Block, PersistenceTransformer as RustPersistenceTransformer};
use pyo3::prelude::*;

/// Encodes temporal persistence/stability of scalar values.
///
/// PersistenceTransformer tracks how long a value remains relatively unchanged,
/// encoding the persistence duration as a sparse distributed representation.
/// This enables downstream blocks to recognize temporal patterns and stability.
#[pyclass(name = "PersistenceTransformer", module = "gnomics")]
pub struct PyPersistenceTransformer {
    inner: RustPersistenceTransformer,
}

#[pymethods]
impl PyPersistenceTransformer {
    /// Create a new PersistenceTransformer.
    ///
    /// Args:
    ///     min_val: Minimum input value (inclusive)
    ///     max_val: Maximum input value (inclusive)
    ///     num_s: Total number of statelets (output bits)
    ///     num_as: Number of active statelets in the encoding
    ///     max_step: Maximum persistence steps to track
    ///     num_t: History depth (minimum 2)
    ///     seed: Random seed (currently unused, reserved for future use)
    #[new]
    #[pyo3(signature = (min_val, max_val, num_s, num_as, max_step, num_t, seed))]
    pub fn new(
        min_val: f64,
        max_val: f64,
        num_s: usize,
        num_as: usize,
        max_step: usize,
        num_t: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let mut transformer =
            RustPersistenceTransformer::new(min_val, max_val, num_s, num_as, max_step, num_t, seed);
        transformer.init().map_err(crate::error::gnomics_error_to_pyerr)?;
        Ok(PyPersistenceTransformer { inner: transformer })
    }

    /// Set the scalar value to track for persistence.
    ///
    /// Values are automatically clamped to [min_val, max_val] range.
    /// If the value changes by more than 10%, the persistence counter resets.
    ///
    /// Args:
    ///     value: Scalar value to track
    pub fn set_value(&mut self, value: f64) {
        self.inner.set_value(value);
    }

    /// Get the current scalar value.
    ///
    /// Returns:
    ///     Current value
    pub fn get_value(&self) -> f64 {
        self.inner.get_value()
    }

    /// Get the current persistence counter.
    ///
    /// The counter increments when the value remains stable (change <= 10%)
    /// and resets to 0 when the value changes significantly.
    ///
    /// Returns:
    ///     Current persistence counter (capped at max_step)
    pub fn get_counter(&self) -> usize {
        self.inner.get_counter()
    }

    /// Execute the computation pipeline.
    ///
    /// This performs: pull() -> compute() -> store() -> [learn()] -> step()
    ///
    /// Args:
    ///     learn_flag: Whether to perform learning (no-op for transformers)
    pub fn execute(&mut self, learn_flag: bool) -> PyResult<()> {
        self.inner.execute(learn_flag).map_err(crate::error::gnomics_error_to_pyerr)
    }

    /// Get the output BitArray containing the encoded persistence duration.
    ///
    /// Returns:
    ///     BitArray with sparse binary encoding of persistence
    pub fn output(&self) -> PyBitArray {
        PyBitArray::from_rust(self.inner.output.state.clone())
    }

    /// Get the output state at a specific time offset.
    ///
    /// Args:
    ///     time: Time offset (0 = current, 1 = previous, etc.)
    ///
    /// Returns:
    ///     BitArray containing the state at that time
    pub fn output_at(&self, time: usize) -> PyBitArray {
        PyBitArray::from_rust(self.inner.output.get_bitarray(time).clone())
    }

    /// Check if the output has changed from the previous time step.
    ///
    /// Returns:
    ///     True if the output changed
    pub fn has_changed(&self) -> bool {
        self.inner.output.has_changed()
    }

    /// Clear all internal state and reset the transformer.
    ///
    /// This resets the persistence counter to 0.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Get the maximum persistence step count.
    ///
    /// Returns:
    ///     Maximum step count
    pub fn max_step(&self) -> usize {
        self.inner.max_step()
    }

    /// Get the minimum value bound.
    ///
    /// Returns:
    ///     Minimum value
    pub fn min_val(&self) -> f64 {
        self.inner.min_val()
    }

    /// Get the maximum value bound.
    ///
    /// Returns:
    ///     Maximum value
    pub fn max_val(&self) -> f64 {
        self.inner.max_val()
    }

    /// Get the total number of statelets (output bits).
    ///
    /// Returns:
    ///     Number of statelets
    pub fn num_s(&self) -> usize {
        self.inner.num_s()
    }

    /// Get the number of active statelets.
    ///
    /// Returns:
    ///     Number of active statelets
    pub fn num_as(&self) -> usize {
        self.inner.num_as()
    }

    /// Get estimated memory usage in bytes.
    ///
    /// Returns:
    ///     Estimated memory usage
    pub fn memory_usage(&self) -> usize {
        self.inner.memory_usage()
    }

    fn __repr__(&self) -> String {
        format!(
            "PersistenceTransformer(min_val={}, max_val={}, num_s={}, num_as={}, max_step={}, value={}, counter={})",
            self.inner.min_val(),
            self.inner.max_val(),
            self.inner.num_s(),
            self.inner.num_as(),
            self.inner.max_step(),
            self.inner.get_value(),
            self.inner.get_counter()
        )
    }
}
