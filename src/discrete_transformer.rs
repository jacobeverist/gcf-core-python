//! Python bindings for gnomics::DiscreteTransformer

use crate::bitarray::PyBitArray;
use gnomics::{Block, DiscreteTransformer as RustDiscreteTransformer};
use pyo3::prelude::*;

/// Encodes categorical/discrete values into unique sparse distributed representations.
///
/// DiscreteTransformer converts discrete categories into binary patterns with
/// zero overlap between different categories. This ensures each category has
/// a unique representation while maintaining the benefits of sparse encoding.
#[pyclass(name = "DiscreteTransformer", module = "gnomics")]
pub struct PyDiscreteTransformer {
    inner: RustDiscreteTransformer,
}

#[pymethods]
impl PyDiscreteTransformer {
    /// Create a new DiscreteTransformer.
    ///
    /// Args:
    ///     num_v: Number of discrete categories (values 0 to num_v-1)
    ///     num_s: Total number of statelets (output bits)
    ///     num_t: History depth (minimum 2)
    ///     seed: Random seed (currently unused, reserved for future use)
    #[new]
    #[pyo3(signature = (num_v, num_s, num_t, seed))]
    pub fn new(num_v: usize, num_s: usize, num_t: usize, seed: u64) -> PyResult<Self> {
        let mut transformer = RustDiscreteTransformer::new(num_v, num_s, num_t, seed);
        transformer.init().map_err(crate::error::gnomics_error_to_pyerr)?;
        Ok(PyDiscreteTransformer { inner: transformer })
    }

    /// Set the category value to encode.
    ///
    /// Args:
    ///     value: Category index (must be in range 0 to num_v-1)
    pub fn set_value(&mut self, value: usize) {
        self.inner.set_value(value);
    }

    /// Get the current category value.
    ///
    /// Returns:
    ///     Current category index
    pub fn get_value(&self) -> usize {
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
    ///     BitArray with sparse binary encoding of the category
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

    /// Get the number of categories.
    ///
    /// Returns:
    ///     Number of categories
    pub fn num_v(&self) -> usize {
        self.inner.num_v()
    }

    /// Get the total number of statelets (output bits).
    ///
    /// Returns:
    ///     Number of statelets
    pub fn num_s(&self) -> usize {
        self.inner.num_s()
    }

    /// Get the number of active statelets per category.
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
            "DiscreteTransformer(num_v={}, num_s={}, num_as={}, value={})",
            self.inner.num_v(),
            self.inner.num_s(),
            self.inner.num_as(),
            self.inner.get_value()
        )
    }
}
