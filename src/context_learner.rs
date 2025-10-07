//! Python bindings for gnomics::ContextLearner

use crate::bitarray::PyBitArray;
use crate::block_input::PyBlockInput;
use crate::block_output::PyBlockOutput;
use gnomics::{Block, ContextLearner as RustContextLearner};
use pyo3::prelude::*;
use std::rc::Rc;

/// Context learning block for context-dependent pattern recognition.
///
/// ContextLearner learns associations between inputs and contexts, enabling
/// it to predict inputs based on context and detect anomalies when inputs
/// appear in unexpected contexts. It's useful for sequence learning,
/// contextual disambiguation, and anomaly detection.
#[pyclass(name = "ContextLearner", module = "gnomics.core", unsendable)]
pub struct PyContextLearner {
    inner: RustContextLearner,
    // Python wrappers for input/context to track child references
    py_input: Option<Py<PyBlockInput>>,
    py_context: Option<Py<PyBlockInput>>,
}

#[pymethods]
impl PyContextLearner {
    /// Create a new ContextLearner for contextual learning.
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
        Ok(PyContextLearner {
            inner: learner,
            py_input: None,
            py_context: None,
        })
    }

    /// Get the BlockInput for connecting encoder outputs.
    ///
    /// Returns:
    ///     BlockInput that can receive connections from other blocks' outputs
    #[getter]
    pub fn input(&mut self, py: Python) -> Py<PyBlockInput> {
        // Return cached input wrapper if it exists
        if let Some(ref input) = self.py_input {
            return input.clone_ref(py);
        }

        // Create new PyBlockInput wrapper by swapping with a temporary BlockInput
        let mut temp_input = gnomics::BlockInput::new();
        std::mem::swap(&mut temp_input, &mut self.inner.input);

        let py_input_obj = PyBlockInput {
            inner: temp_input,
            py_children: Vec::new(),
        };

        let py_input = Py::new(py, py_input_obj).unwrap();
        self.py_input = Some(py_input.clone_ref(py));
        py_input
    }

    /// Get the BlockInput for connecting context outputs.
    ///
    /// Returns:
    ///     BlockInput that can receive connections from other blocks' outputs
    #[getter]
    pub fn context(&mut self, py: Python) -> Py<PyBlockInput> {
        // Return cached context wrapper if it exists
        if let Some(ref context) = self.py_context {
            return context.clone_ref(py);
        }

        // Create new PyBlockInput wrapper by swapping with a temporary BlockInput
        let mut temp_context = gnomics::BlockInput::new();
        std::mem::swap(&mut temp_context, &mut self.inner.context);

        let py_context_obj = PyBlockInput {
            inner: temp_context,
            py_children: Vec::new(),
        };

        let py_context = Py::new(py, py_context_obj).unwrap();
        self.py_context = Some(py_context.clone_ref(py));
        py_context
    }

    /// Initialize the learner based on connected inputs.
    ///
    /// Must be called after connecting inputs and before using compute or learn methods.
    ///
    /// Args:
    ///     num_input_bits: Optional number of input bits. If not provided, uses connected input size.
    ///     num_context_bits: Optional number of context bits. If not provided, uses connected context size.
    #[pyo3(signature = (num_input_bits=None, num_context_bits=None))]
    pub fn init(&mut self, py: Python, num_input_bits: Option<usize>, num_context_bits: Option<usize>) -> PyResult<()> {
        // Sync from py_input if it exists by swapping back
        if let Some(ref py_input) = self.py_input {
            let mut py_input_mut = py_input.borrow_mut(py);
            std::mem::swap(&mut self.inner.input, &mut py_input_mut.inner);
        }

        // Sync from py_context if it exists by swapping back
        if let Some(ref py_context) = self.py_context {
            let mut py_context_mut = py_context.borrow_mut(py);
            std::mem::swap(&mut self.inner.context, &mut py_context_mut.inner);
        }

        // Set input sizes if provided
        if let Some(size) = num_input_bits {
            self.inner.input.state.resize(size);
        }
        if let Some(size) = num_context_bits {
            self.inner.context.state.resize(size);
        }

        let result = self.inner.init().map_err(crate::error::gnomics_error_to_pyerr);

        // Swap forward again to keep wrappers in sync
        if let Some(ref py_input) = self.py_input {
            let mut py_input_mut = py_input.borrow_mut(py);
            std::mem::swap(&mut self.inner.input, &mut py_input_mut.inner);
        }

        if let Some(ref py_context) = self.py_context {
            let mut py_context_mut = py_context.borrow_mut(py);
            std::mem::swap(&mut self.inner.context, &mut py_context_mut.inner);
        }

        result
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

    /// Get the output BlockOutput object for connecting to other blocks.
    ///
    /// Returns:
    ///     BlockOutput that can be connected to other blocks' inputs
    #[getter]
    pub fn output(&self) -> PyBlockOutput {
        PyBlockOutput {
            inner: Rc::clone(&self.inner.output),
        }
    }

    /// Get the current output state as a BitArray.
    ///
    /// Returns:
    ///     BitArray with predicted/active statelets
    pub fn get_output_state(&self) -> PyBitArray {
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
