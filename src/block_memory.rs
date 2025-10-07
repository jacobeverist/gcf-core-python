//! Python bindings for gnomics::BlockMemory

use crate::bitarray::PyBitArray;
use gnomics::BlockMemory as RustBlockMemory;
use pyo3::prelude::*;

/// Synaptic learning with permanence-based connections.
///
/// BlockMemory implements biologically-inspired learning through "dendrites"
/// with multiple "receptors" that have adjustable permanence values (0-99).
/// Learning strengthens matching connections and weakens non-matching ones.
#[pyclass(name = "BlockMemory", module = "gnomics")]
pub struct PyBlockMemory {
    inner: RustBlockMemory,
}

#[pymethods]
impl PyBlockMemory {
    /// Create a new BlockMemory with learning parameters.
    ///
    /// Args:
    ///     num_d: Number of dendrites
    ///     num_rpd: Number of receptors per dendrite
    ///     perm_thr: Permanence threshold for "connected" state (0-99)
    ///     perm_inc: Permanence increment on learning (typically 1-5)
    ///     perm_dec: Permanence decrement on unlearning (typically 1-2)
    ///     pct_learn: Percentage of receptors to learn/move (0.0-1.0)
    #[new]
    #[pyo3(signature = (num_d, num_rpd, perm_thr, perm_inc, perm_dec, pct_learn))]
    pub fn new(
        num_d: usize,
        num_rpd: usize,
        perm_thr: u8,
        perm_inc: u8,
        perm_dec: u8,
        pct_learn: f64,
    ) -> Self {
        PyBlockMemory {
            inner: RustBlockMemory::new(num_d, num_rpd, perm_thr, perm_inc, perm_dec, pct_learn),
        }
    }

    /// Initialize the memory with a specific input size.
    ///
    /// Must be called before using overlap, learn, or other methods.
    ///
    /// Args:
    ///     num_i: Number of input bits
    ///     seed: Random seed for initialization
    pub fn init(&mut self, num_i: usize, seed: u64) {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        self.inner.init(num_i, &mut rng);
    }

    /// Calculate overlap score for a specific dendrite.
    ///
    /// Overlap is the count of connected receptors (permanence >= threshold)
    /// that match active bits in the input.
    ///
    /// Args:
    ///     d: Dendrite index
    ///     input: Input BitArray pattern
    ///
    /// Returns:
    ///     Overlap score
    pub fn overlap(&self, d: usize, input: &PyBitArray) -> usize {
        self.inner.overlap(d, input.as_rust())
    }

    /// Learn from an input pattern for a specific dendrite.
    ///
    /// Strengthens connections to active input bits and weakens connections
    /// to inactive bits. Requires an RNG for the learn_move operation.
    ///
    /// Args:
    ///     d: Dendrite index
    ///     input: Input BitArray pattern
    ///     seed: Random seed for any receptor movement
    pub fn learn(&mut self, d: usize, input: &PyBitArray, seed: u64) {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        self.inner.learn(d, input.as_rust(), &mut rng);
    }

    /// Punish (weaken) connections for a specific dendrite.
    ///
    /// Decrements permanence for receptors that match the input pattern,
    /// implementing negative learning.
    ///
    /// Args:
    ///     d: Dendrite index
    ///     input: Input BitArray pattern
    ///     seed: Random seed for any receptor movement
    pub fn punish(&mut self, d: usize, input: &PyBitArray, seed: u64) {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        self.inner.punish(d, input.as_rust(), &mut rng);
    }

    /// Move "dead" receptors (zero permanence) to new random positions.
    ///
    /// This helps maintain dendrite coverage across the input space by
    /// relocating receptors that have been completely unlearned.
    ///
    /// Args:
    ///     d: Dendrite index
    ///     input: Input BitArray pattern (used to determine connected receptors)
    ///     seed: Random seed for receptor placement
    pub fn learn_move(&mut self, d: usize, input: &PyBitArray, seed: u64) {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        self.inner.learn_move(d, input.as_rust(), &mut rng);
    }

    /// Get the number of dendrites.
    ///
    /// Returns:
    ///     Number of dendrites
    pub fn num_dendrites(&self) -> usize {
        self.inner.num_dendrites()
    }

    /// Get receptor addresses for a specific dendrite.
    ///
    /// Args:
    ///     d: Index of the dendrite
    ///
    /// Returns:
    ///     List of input bit indices that receptors connect to
    pub fn addrs(&self, d: usize) -> Vec<usize> {
        self.inner.addrs(d)
    }

    /// Get receptor permanence values for a specific dendrite.
    ///
    /// Args:
    ///     d: Index of the dendrite
    ///
    /// Returns:
    ///     List of permanence values (0-99)
    pub fn perms(&self, d: usize) -> Vec<u8> {
        self.inner.perms(d)
    }

    /// Get connected receptors as a BitArray for a specific dendrite.
    ///
    /// Args:
    ///     d: Index of the dendrite
    ///
    /// Returns:
    ///     BitArray indicating which receptors are connected (permanence >= threshold),
    ///     or None if not computed yet
    pub fn conns(&self, d: usize) -> Option<PyBitArray> {
        self.inner.conns(d).map(|ba| PyBitArray::from_rust(ba.clone()))
    }

    fn __repr__(&self) -> String {
        format!(
            "BlockMemory(num_dendrites={})",
            self.inner.num_dendrites()
        )
    }
}
