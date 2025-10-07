//! Python bindings for gnomics::PatternPooler

use crate::bitarray::PyBitArray;
use crate::block_input::PyBlockInput;
use crate::block_output::PyBlockOutput;
use gnomics::{Block, PatternPooler as RustPatternPooler};
use pyo3::prelude::*;
use std::rc::Rc;

/// Unsupervised learning block for pattern recognition.
///
/// PatternPooler creates sparse distributed representations through competitive
/// winner-take-all learning. It learns to recognize patterns in input data
/// without supervision, making it useful for feature extraction and clustering.
#[pyclass(name = "PatternPooler", module = "gnomics.core", unsendable)]
pub struct PyPatternPooler {
    inner: RustPatternPooler,
    // Python wrapper for input to track child references
    py_input: Option<Py<PyBlockInput>>,
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
        Ok(PyPatternPooler {
            inner: pooler,
            py_input: None,
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

    /// Initialize the pooler based on connected inputs.
    ///
    /// Must be called after connecting inputs and before using compute or learn methods.
    ///
    /// Args:
    ///     num_i: Optional number of input bits. If not provided, uses connected input size.
    #[pyo3(signature = (num_i=None))]
    pub fn init(&mut self, py: Python, num_i: Option<usize>) -> PyResult<()> {
        // Sync from py_input if it exists by swapping back, then forward again
        if let Some(ref py_input) = self.py_input {
            let mut py_input_mut = py_input.borrow_mut(py);
            // Swap the BlockInput with children back into inner.input
            std::mem::swap(&mut self.inner.input, &mut py_input_mut.inner);
        }

        // Set input size if provided
        if let Some(size) = num_i {
            self.inner.input.state.resize(size);
        }

        let result = self.inner.init().map_err(crate::error::gnomics_error_to_pyerr);

        // Swap forward again to keep py_input in sync
        if let Some(ref py_input) = self.py_input {
            let mut py_input_mut = py_input.borrow_mut(py);
            std::mem::swap(&mut self.inner.input, &mut py_input_mut.inner);
        }

        result
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

    /// Execute the full computation pipeline.
    ///
    /// This performs: step() -> pull() -> compute() -> [learn()] -> store()
    /// Input is automatically pulled from connected BlockInput.
    ///
    /// Args:
    ///     learn_flag: Whether to perform learning
    pub fn execute(&mut self, py: Python, learn_flag: bool) -> PyResult<()> {
        // Sync from py_input if it exists
        if let Some(ref py_input) = self.py_input {
            let mut py_input_mut = py_input.borrow_mut(py);
            std::mem::swap(&mut self.inner.input, &mut py_input_mut.inner);
        }

        // Execute using the Rust Block trait's execute which handles pull()
        let result = self.inner.execute(learn_flag).map_err(crate::error::gnomics_error_to_pyerr);

        // Swap back to keep py_input in sync
        if let Some(ref py_input) = self.py_input {
            let mut py_input_mut = py_input.borrow_mut(py);
            std::mem::swap(&mut self.inner.input, &mut py_input_mut.inner);
        }

        result
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
    ///     BitArray with exactly num_as bits set
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
