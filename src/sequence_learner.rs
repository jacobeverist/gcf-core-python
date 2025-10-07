//! Python bindings for gnomics::SequenceLearner

use crate::bitarray::PyBitArray;
use crate::block_input::PyBlockInput;
use crate::block_output::PyBlockOutput;
use gnomics::{Block, SequenceLearner as RustSequenceLearner};
use pyo3::prelude::*;
use std::rc::Rc;

/// Sequence learning block for temporal pattern recognition with self-feedback.
///
/// SequenceLearner learns temporal sequences and predicts next patterns.
/// It is nearly identical to ContextLearner but uses its own previous output
/// as context, enabling it to learn temporal transitions automatically.
#[pyclass(name = "SequenceLearner", module = "gnomics.core", unsendable)]
pub struct PySequenceLearner {
    inner: RustSequenceLearner,
    // Python wrapper for input to track child references
    py_input: Option<Py<PyBlockInput>>,
}

#[pymethods]
impl PySequenceLearner {
    /// Create a new SequenceLearner for temporal sequence learning.
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
    ///     always_update: Update even when inputs unchanged
    ///     seed: Random seed for reproducibility
    #[new]
    #[pyo3(signature = (num_c, num_spc, num_dps, num_rpd, d_thresh, perm_thr, perm_inc, perm_dec, num_t, always_update=false, seed=0))]
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
        always_update: bool,
        seed: u64,
    ) -> PyResult<Self> {
        let learner = RustSequenceLearner::new(
            num_c,
            num_spc,
            num_dps,
            num_rpd,
            d_thresh,
            perm_thr,
            perm_inc,
            perm_dec,
            num_t,
            always_update,
            seed,
        );
        Ok(PySequenceLearner {
            inner: learner,
            py_input: None,
        })
    }

    /// Initialize the learner based on connected inputs.
    ///
    /// Must be called after connecting inputs and before using compute or learn methods.
    /// Context is automatically connected to previous output (self-feedback).
    ///
    /// Args:
    ///     num_input_bits: Optional number of input bits. If not provided, uses connected input size.
    #[pyo3(signature = (num_input_bits=None))]
    pub fn init(&mut self, py: Python, num_input_bits: Option<usize>) -> PyResult<()> {
        // Sync from py_input if it exists by swapping back
        if let Some(ref py_input) = self.py_input {
            let mut py_input_mut = py_input.borrow_mut(py);
            std::mem::swap(&mut self.inner.input, &mut py_input_mut.inner);
        }

        // Set input size if provided
        if let Some(size) = num_input_bits {
            self.inner.input.state.resize(size);
        }

        // Context is auto-connected to output[PREV], no need to set size
        let result = self.inner.init().map_err(crate::error::gnomics_error_to_pyerr);

        // Swap forward again to keep py_input in sync
        if let Some(ref py_input) = self.py_input {
            let mut py_input_mut = py_input.borrow_mut(py);
            std::mem::swap(&mut self.inner.input, &mut py_input_mut.inner);
        }

        result
    }

    /// Compute predictions from input pattern.
    ///
    /// Uses previous output as context to predict current input.
    ///
    /// Args:
    ///     input: Input BitArray pattern (active columns)
    pub fn compute(&mut self, input: &PyBitArray) -> PyResult<()> {
        self.inner.input.state = input.as_rust().clone();
        self.inner.compute();
        Ok(())
    }

    /// Learn the temporal transition from previous to current pattern.
    pub fn learn(&mut self) -> PyResult<()> {
        self.inner.learn();
        Ok(())
    }

    /// Execute compute and learn in one step.
    ///
    /// Args:
    ///     input: Input BitArray pattern
    ///     learn_flag: Whether to learn from this example
    pub fn execute(&mut self, input: &PyBitArray, learn_flag: bool) -> PyResult<()> {
        self.inner.input.state = input.as_rust().clone();
        self.inner.compute();
        if learn_flag {
            self.inner.learn();
        }
        self.inner.store();
        self.inner.step();
        Ok(())
    }

    /// Clear all state and history.
    pub fn clear(&mut self) {
        self.inner.clear();
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
    /// Returns the active statelets predicted for the current time step.
    pub fn get_output_state(&self) -> PyBitArray {
        PyBitArray::from_rust(self.inner.output.borrow().state.clone())
    }

    /// Get output at a specific historical time step.
    ///
    /// Args:
    ///     time: Time offset (0=current, 1=previous, etc.)
    ///
    /// Returns:
    ///     BitArray at the specified time step
    pub fn output_at(&self, time: usize) -> PyResult<PyBitArray> {
        Ok(PyBitArray::from_rust(self.inner.output.borrow().get_bitarray(time).clone()))
    }

    /// Check if output has changed since last time step.
    ///
    /// Returns:
    ///     True if output differs from previous time step
    pub fn has_changed(&self) -> bool {
        self.inner.output.borrow().has_changed()
    }

    /// Get current anomaly score (0.0-1.0).
    ///
    /// Indicates the percentage of input columns that were unexpected.
    /// - 0.0 = All columns were predicted by previous pattern
    /// - 1.0 = All columns were unexpected (sequence broken)
    pub fn get_anomaly_score(&self) -> f64 {
        self.inner.get_anomaly_score()
    }

    /// Get count of statelets that have learned at least one transition.
    ///
    /// Indicates how many different temporal patterns have been learned.
    pub fn get_historical_count(&self) -> usize {
        self.inner.get_historical_count()
    }

    /// Get number of columns.
    pub fn num_c(&self) -> usize {
        self.inner.num_c()
    }

    /// Get statelets per column.
    pub fn num_spc(&self) -> usize {
        self.inner.num_spc()
    }

    /// Get dendrites per statelet.
    pub fn num_dps(&self) -> usize {
        self.inner.num_dps()
    }

    /// Get dendrite activation threshold.
    pub fn d_thresh(&self) -> u32 {
        self.inner.d_thresh()
    }

    /// Get memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.inner.memory_usage()
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        format!(
            "SequenceLearner(num_c={}, num_spc={}, num_dps={}, d_thresh={})",
            self.inner.num_c(),
            self.inner.num_spc(),
            self.inner.num_dps(),
            self.inner.d_thresh()
        )
    }
}
