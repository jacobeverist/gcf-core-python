//! Python bindings for gnomics::ContextLearner

use crate::bitarray::PyBitArray;
use gnomics::{Block, ContextLearner as RustContextLearner};
use pyo3::prelude::*;

/// Temporal learning block for context-dependent pattern recognition.
///
/// ContextLearner learns associations between inputs and contexts, enabling
/// it to predict inputs based on context and detect anomalies when inputs
/// appear in unexpected contexts. It's useful for sequence learning,
/// contextual disambiguation, and anomaly detection.
#[pyclass(name = "ContextLearner", module = "gcf_core_python_client", unsendable)]
pub struct PyContextLearner {
    inner: RustContextLearner,
}

#[pymethods]
impl PyContextLearner {
    /// Create a new ContextLearner for temporal/contextual learning.
    ///
    /// Args:
    ///     num_c: Number of columns (matches input size)
    ///     num_spc: Statelets per column
    ///     num_dps: Dendrites per statelet
    ///     num_rpd: Receptors per dendrite
    ///     d_thresh: Dendrite activation threshold
    ///     perm_thr: Permanence threshold (0-99, typically 20)
    ///     perm_inc: Permanence increment (typically 2)
    ///     perm_dec: Permanence decrement (typically 1)
    ///     num_t: History depth (minimum 2)
    ///     seed: Random seed for reproducibility
    #[new]
    #[pyo3(signature = (num_c, num_spc, num_dps, num_rpd, d_thresh, perm_thr, perm_inc, perm_dec, num_t, seed))]
    pub fn new(
        num_c: usize,
        num_spc: usize,
        num_dps: usize,
        num_rpd: usize,
        d_thresh: u32,
        perm_thr: u8,
        perm_inc: u8,
        perm_dec: u8,
        num_t: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let learner = RustContextLearner::new(
            num_c,
            num_spc,
            num_dps,
            num_rpd,
            d_thresh,
            perm_thr,
            perm_inc,
            perm_dec,
            num_t,
            false, // always_update
            seed,
        );
        Ok(PyContextLearner { inner: learner })
    }

    /// Initialize the learner with input and context sizes.
    ///
    /// Must be called before using compute or learn methods.
    ///
    /// Args:
    ///     num_input_bits: Number of input bits (columns)
    ///     num_context_bits: Number of context bits
    pub fn init(&mut self, num_input_bits: usize, num_context_bits: usize) -> PyResult<()> {
        // Set input sizes in the block
        self.inner.input.state.resize(num_input_bits);
        self.inner.context.state.resize(num_context_bits);
        self.inner.init().map_err(crate::error::gnomics_error_to_pyerr)
    }

    /// Compute predictions from input and context patterns.
    ///
    /// Args:
    ///     input: Input BitArray pattern (active columns)
    ///     context: Context BitArray pattern
    pub fn compute(&mut self, input: &PyBitArray, context: &PyBitArray) {
        // Copy inputs to internal state
        self.inner.input.state = input.as_rust().clone();
        self.inner.context.state = context.as_rust().clone();

        // Compute predictions
        self.inner.compute();
    }

    /// Learn from current input and context patterns.
    ///
    /// Args:
    ///     input: Input BitArray pattern (active columns)
    ///     context: Context BitArray pattern
    pub fn learn(&mut self, input: &PyBitArray, context: &PyBitArray) {
        // Copy inputs to internal state
        self.inner.input.state = input.as_rust().clone();
        self.inner.context.state = context.as_rust().clone();

        // Learn from current pattern
        self.inner.learn();
    }

    /// Execute full pipeline: compute and optionally learn.
    ///
    /// Args:
    ///     input: Input BitArray pattern (active columns)
    ///     context: Context BitArray pattern
    ///     learn_flag: Whether to perform learning
    pub fn execute(&mut self, input: &PyBitArray, context: &PyBitArray, learn_flag: bool) {
        self.compute(input, context);
        if learn_flag {
            self.learn(input, context);
        }
        self.inner.store();
        self.inner.step();
    }

    /// Get the anomaly score for the current prediction.
    ///
    /// Returns:
    ///     Anomaly score from 0.0 (fully predicted) to 1.0 (completely surprising)
    pub fn get_anomaly_score(&self) -> f64 {
        self.inner.get_anomaly_score()
    }

    /// Get the number of statelets with at least one used dendrite.
    ///
    /// Returns:
    ///     Count of statelets that have learned patterns
    pub fn get_historical_count(&self) -> usize {
        self.inner.get_historical_count()
    }

    /// Get the output BitArray containing predicted/active statelets.
    ///
    /// Returns:
    ///     BitArray with active statelets
    pub fn output(&self) -> PyBitArray {
        PyBitArray::from_rust(self.inner.output.borrow().state.clone())
    }

    /// Get the output state at a specific time offset.
    ///
    /// Args:
    ///     time: Time offset (0 = current, 1 = previous, etc.)
    ///
    /// Returns:
    ///     BitArray containing the state at that time
    pub fn output_at(&self, time: usize) -> PyBitArray {
        PyBitArray::from_rust(self.inner.output.borrow().get_bitarray(time).clone())
    }

    /// Check if the output has changed from the previous time step.
    ///
    /// Returns:
    ///     True if the output changed
    pub fn has_changed(&self) -> bool {
        self.inner.output.borrow().has_changed()
    }

    /// Clear all internal state and reset the learner.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Get the number of columns.
    ///
    /// Returns:
    ///     Number of columns
    pub fn num_c(&self) -> usize {
        self.inner.num_c()
    }

    /// Get the number of statelets per column.
    ///
    /// Returns:
    ///     Statelets per column
    pub fn num_spc(&self) -> usize {
        self.inner.num_spc()
    }

    /// Get the number of dendrites per statelet.
    ///
    /// Returns:
    ///     Dendrites per statelet
    pub fn num_dps(&self) -> usize {
        self.inner.num_dps()
    }

    /// Get the dendrite activation threshold.
    ///
    /// Returns:
    ///     Dendrite threshold
    pub fn d_thresh(&self) -> u32 {
        self.inner.d_thresh()
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
            "ContextLearner(num_c={}, num_spc={}, num_dps={}, d_thresh={})",
            self.inner.num_c(),
            self.inner.num_spc(),
            self.inner.num_dps(),
            self.inner.d_thresh()
        )
    }
}
