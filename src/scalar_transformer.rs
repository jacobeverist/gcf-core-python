//! Python bindings for gnomics::ScalarTransformer

use crate::bitarray::PyBitArray;
use gnomics::{Block, ScalarTransformer as RustScalarTransformer};
use pyo3::prelude::*;

/// Encodes continuous scalar values into sparse distributed representations.
///
/// ScalarTransformer converts continuous values into binary patterns where
/// semantically similar values have overlapping bit patterns. This enables
/// downstream blocks to recognize similar values and generalize across
/// continuous ranges.
#[pyclass(name = "ScalarTransformer", module = "gnomics")]
pub struct PyScalarTransformer {
    inner: RustScalarTransformer,
}

#[pymethods]
impl PyScalarTransformer {
    /// Create a new ScalarTransformer.
    ///
    /// Args:
    ///     min_val: Minimum input value (inclusive)
    ///     max_val: Maximum input value (inclusive)
    ///     num_s: Total number of statelets (output bits)
    ///     num_as: Number of active statelets in the encoding
    ///     num_t: History depth (minimum 2)
    ///     seed: Random seed (currently unused, reserved for future use)
    #[new]
    #[pyo3(signature = (min_val, max_val, num_s, num_as, num_t, seed))]
    pub fn new(
        min_val: f64,
        max_val: f64,
        num_s: usize,
        num_as: usize,
        num_t: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let mut transformer = RustScalarTransformer::new(min_val, max_val, num_s, num_as, num_t, seed);
        transformer.init().map_err(crate::error::gnomics_error_to_pyerr)?;
        Ok(PyScalarTransformer { inner: transformer })
    }

    /// Set the scalar value to encode.
    ///
    /// Values are automatically clamped to [min_val, max_val] range.
    ///
    /// Args:
    ///     value: Scalar value to encode
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

    /// Execute the computation pipeline.
    ///
    /// This performs: pull() -> compute() -> store() -> [learn()] -> step()
    ///
    /// Args:
    ///     learn_flag: Whether to perform learning (no-op for transformers)
    pub fn execute(&mut self, learn_flag: bool) -> PyResult<()> {
        self.inner.execute(learn_flag).map_err(crate::error::gnomics_error_to_pyerr)
    }

    /// Get the output BitArray containing the encoded representation.
    ///
    /// Returns:
    ///     BitArray with sparse binary encoding of the scalar value
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
    pub fn clear(&mut self) {
        self.inner.clear();
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
            "ScalarTransformer(min_val={}, max_val={}, num_s={}, num_as={}, value={})",
            self.inner.min_val(),
            self.inner.max_val(),
            self.inner.num_s(),
            self.inner.num_as(),
            self.inner.get_value()
        )
    }
}
