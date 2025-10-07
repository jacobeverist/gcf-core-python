//! Python bindings for gnomics::BlockInput

use crate::bitarray::PyBitArray;
use crate::block_output::PyBlockOutput;
use gnomics::BlockInput as RustBlockInput;
use pyo3::prelude::*;
use std::rc::Rc;

/// Manages block inputs with lazy copying from child outputs.
///
/// BlockInput concatenates multiple child BlockOutputs into a single input
/// BitArray with lazy copying optimization - data is only copied from children
/// that have changed.
#[pyclass(name = "BlockInput", module = "gnomics.core", unsendable)]
pub struct PyBlockInput {
    inner: RustBlockInput,
    // Keep Python references alive to prevent GC
    py_children: Vec<Py<PyBlockOutput>>,
}

#[pymethods]
impl PyBlockInput {
    /// Create a new BlockInput.
    #[new]
    pub fn new() -> Self {
        PyBlockInput {
            inner: RustBlockInput::new(),
            py_children: Vec::new(),
        }
    }

    /// Add a child BlockOutput at a specific time offset.
    ///
    /// Data is NOT copied during this call - only metadata is stored.
    /// Actual copying happens during pull() and only for changed children.
    ///
    /// Args:
    ///     child: BlockOutput to add as a child
    ///     time: Time offset (0=current, 1=previous, etc.)
    pub fn add_child(&mut self, py: Python, child: Py<PyBlockOutput>, time: usize) {
        // Clone the Rc (cheap - just increments reference count)
        let rc_child = {
            let child_ref = child.borrow(py);
            Rc::clone(&child_ref.inner)
        }; // child_ref is dropped here

        // Add to Rust BlockInput
        self.inner.add_child(rc_child, time);

        // Keep Python reference alive
        self.py_children.push(child);
    }

    /// Pull data from child outputs (with lazy copying optimization).
    ///
    /// Only copies data from children that have changed. Unchanged children
    /// are skipped, providing significant performance benefits.
    pub fn pull(&mut self) {
        self.inner.pull();
    }

    /// Check if any child has changed.
    ///
    /// Returns immediately on first change found (short-circuit evaluation).
    /// Enables downstream blocks to skip computation when no inputs changed.
    ///
    /// Returns:
    ///     True if any child has changed
    pub fn children_changed(&self) -> bool {
        self.inner.children_changed()
    }

    /// Clear all bits in state to 0.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Get number of children.
    ///
    /// Returns:
    ///     Number of child outputs
    pub fn num_children(&self) -> usize {
        self.inner.num_children()
    }

    /// Get total number of bits in concatenated state.
    ///
    /// Returns:
    ///     Total bits
    pub fn num_bits(&self) -> usize {
        self.inner.num_bits()
    }

    /// Get unique input ID.
    ///
    /// Returns:
    ///     Input ID
    pub fn id(&self) -> u32 {
        self.inner.id()
    }

    /// Get the concatenated input state.
    ///
    /// Returns:
    ///     BitArray containing the concatenated state
    pub fn state(&self) -> PyBitArray {
        PyBitArray::from_rust(self.inner.state.clone())
    }

    /// Estimate memory usage in bytes.
    ///
    /// Returns:
    ///     Memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.inner.memory_usage()
    }

    fn __repr__(&self) -> String {
        format!(
            "BlockInput(num_children={}, num_bits={}, id={})",
            self.inner.num_children(),
            self.inner.num_bits(),
            self.inner.id()
        )
    }
}
