//! Python bindings for gnomics::PatternPooler

use crate::bitarray::PyBitArray;
use gnomics::{Block, PatternPooler as RustPatternPooler};
use pyo3::prelude::*;

/// Unsupervised learning block for pattern recognition.
///
/// PatternPooler creates sparse distributed representations through competitive
/// winner-take-all learning. It learns to recognize patterns in input data
/// without supervision, making it useful for feature extraction and clustering.
#[pyclass(name = "PatternPooler", module = "gnomics", unsendable)]
pub struct PyPatternPooler {
    inner: RustPatternPooler,
}

#[pymethods]
impl PyPatternPooler {
    /// Create a new PatternPooler for unsupervised learning.
    ///
    /// Args:
    ///     num_s: Number of statelets (dendrites)
    ///     num_as: Number of active statelets in output
    ///     perm_thr: Permanence threshold (0-99, typically 20)
    ///     perm_inc: Permanence increment (typically 2)
    ///     perm_dec: Permanence decrement (typically 1)
    ///     pct_pool: Pooling percentage (typically 0.8)
    ///     pct_conn: Initial connectivity (typically 0.5)
    ///     pct_learn: Learning percentage (typically 0.3)
    ///     num_t: History depth (minimum 2)
    ///     seed: Random seed for reproducibility
    #[new]
    #[pyo3(signature = (num_s, num_as, perm_thr, perm_inc, perm_dec, pct_pool, pct_conn, pct_learn, num_t, seed))]
    pub fn new(
        num_s: usize,
        num_as: usize,
        perm_thr: u8,
        perm_inc: u8,
        perm_dec: u8,
        pct_pool: f64,
        pct_conn: f64,
        pct_learn: f64,
        num_t: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let pooler = RustPatternPooler::new(
            num_s,
            num_as,
            perm_thr,
            perm_inc,
            perm_dec,
            pct_pool,
            pct_conn,
            pct_learn,
            false, // always_update
            num_t,
            seed,
        );
        Ok(PyPatternPooler { inner: pooler })
    }

    /// Initialize the pooler with input size.
    ///
    /// Must be called before using compute or learn methods.
    ///
    /// Args:
    ///     num_i: Number of input bits
    pub fn init(&mut self, num_i: usize) -> PyResult<()> {
        // Set input size in the block
        self.inner.input.state.resize(num_i);
        self.inner.init().map_err(crate::error::gnomics_error_to_pyerr)
    }

    /// Compute output from input pattern.
    ///
    /// Finds top overlapping dendrites and activates them.
    ///
    /// Args:
    ///     input: Input BitArray pattern
    pub fn compute(&mut self, input: &PyBitArray) {
        // Copy input to internal state
        self.inner.input.state = input.as_rust().clone();

        // Compute overlaps and activate winners
        self.inner.compute();
    }

    /// Learn from current input pattern.
    ///
    /// Strengthens connections of winning dendrites to active input bits.
    ///
    /// Args:
    ///     input: Input BitArray pattern
    pub fn learn(&mut self, input: &PyBitArray) {
        // Copy input to internal state
        self.inner.input.state = input.as_rust().clone();

        // Learn from current pattern
        self.inner.learn();
    }

    /// Execute full pipeline: compute and optionally learn.
    ///
    /// Args:
    ///     input: Input BitArray pattern
    ///     learn_flag: Whether to perform learning
    pub fn execute(&mut self, input: &PyBitArray, learn_flag: bool) {
        self.compute(input);
        if learn_flag {
            self.learn(input);
        }
        self.inner.store();
        self.inner.step();
    }

    /// Get the output BitArray containing active dendrites.
    ///
    /// Returns:
    ///     BitArray with exactly num_as bits set
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

    /// Clear all internal state and reset the pooler.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Get the number of statelets (dendrites).
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

    /// Get the permanence threshold.
    ///
    /// Returns:
    ///     Permanence threshold (0-99)
    pub fn perm_thr(&self) -> u8 {
        self.inner.perm_thr()
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
            "PatternPooler(num_s={}, num_as={}, perm_thr={})",
            self.inner.num_s(),
            self.inner.num_as(),
            self.inner.perm_thr()
        )
    }
}
