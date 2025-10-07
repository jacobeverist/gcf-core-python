//! Python bindings for gnomics::PatternClassifier

use crate::bitarray::PyBitArray;
use gnomics::{Block, PatternClassifier as RustPatternClassifier};
use pyo3::prelude::*;

/// Supervised learning block for multi-class classification.
///
/// PatternClassifier learns to classify input patterns into discrete categories
/// through supervised learning. It produces probability distributions over
/// class labels and can predict the most likely label for new inputs.
#[pyclass(name = "PatternClassifier", module = "gnomics.core", unsendable)]
pub struct PyPatternClassifier {
    inner: RustPatternClassifier,
}

#[pymethods]
impl PyPatternClassifier {
    /// Create a new PatternClassifier for supervised classification.
    ///
    /// Args:
    ///     num_l: Number of labels/classes
    ///     num_s: Total number of statelets
    ///     num_as: Active statelets per label
    ///     perm_thr: Permanence threshold (0-99, typically 20)
    ///     perm_inc: Permanence increment (typically 2)
    ///     perm_dec: Permanence decrement (typically 1)
    ///     pct_pool: Pooling percentage (typically 0.8)
    ///     pct_conn: Initial connectivity (typically 0.5)
    ///     pct_learn: Learning percentage (typically 0.3)
    ///     num_t: History depth (minimum 2)
    ///     seed: Random seed for reproducibility
    #[new]
    #[pyo3(signature = (num_l, num_s, num_as, perm_thr, perm_inc, perm_dec, pct_pool, pct_conn, pct_learn, num_t, seed))]
    pub fn new(
        num_l: usize,
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
        let classifier = RustPatternClassifier::new(
            num_l, num_s, num_as, perm_thr, perm_inc, perm_dec, pct_pool, pct_conn, pct_learn,
            num_t, seed,
        );
        Ok(PyPatternClassifier {
            inner: classifier,
        })
    }

    /// Initialize the classifier with input size.
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

    /// Set the ground truth label for learning.
    ///
    /// Must be called before learn() to specify which label to train.
    ///
    /// Args:
    ///     label: Label index (0 to num_l-1)
    pub fn set_label(&mut self, label: usize) {
        self.inner.set_label(label);
    }

    /// Compute output and probabilities from input pattern.
    ///
    /// Args:
    ///     input: Input BitArray pattern
    pub fn compute(&mut self, input: &PyBitArray) {
        // Copy input to internal state
        self.inner.input.state = input.as_rust().clone();

        // Compute overlaps and activate winners per label
        self.inner.compute();
    }

    /// Learn from current input pattern using the set label.
    ///
    /// Must call set_label() first to specify ground truth.
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
    /// If learn_flag is True, must call set_label() first.
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

    /// Get probability distribution over all labels.
    ///
    /// Returns:
    ///     List of probabilities for each label (sums to 1.0)
    pub fn get_probabilities(&self) -> Vec<f64> {
        self.inner.get_probabilities()
    }

    /// Get the predicted label (highest probability).
    ///
    /// Returns:
    ///     Label index with highest probability
    pub fn get_predicted_label(&self) -> usize {
        self.inner.get_predicted_label()
    }

    /// Get all label indices.
    ///
    /// Returns:
    ///     List of label indices [0, 1, 2, ..., num_l-1]
    pub fn get_labels(&self) -> Vec<usize> {
        self.inner.get_labels()
    }

    /// Get the output BitArray containing active dendrites.
    ///
    /// Returns:
    ///     BitArray with num_l * num_as bits set
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

    /// Clear all internal state and reset the classifier.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Get the number of labels.
    ///
    /// Returns:
    ///     Number of labels
    pub fn num_l(&self) -> usize {
        self.inner.num_l()
    }

    /// Get the number of statelets.
    ///
    /// Returns:
    ///     Total number of statelets
    pub fn num_s(&self) -> usize {
        self.inner.num_s()
    }

    /// Get the number of active statelets per label.
    ///
    /// Returns:
    ///     Active statelets per label
    pub fn num_as(&self) -> usize {
        self.inner.num_as()
    }

    /// Get the number of statelets per label.
    ///
    /// Returns:
    ///     Statelets per label (num_s / num_l)
    pub fn num_spl(&self) -> usize {
        self.inner.num_spl()
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
            "PatternClassifier(num_l={}, num_s={}, num_as={})",
            self.inner.num_l(), self.inner.num_s(), self.inner.num_as()
        )
    }
}
