mod bitarray;
mod block_memory;
mod block_output;
mod context_learner;
mod discrete_transformer;
mod error;
mod pattern_classifier;
mod pattern_pooler;
mod persistence_transformer;
mod scalar_transformer;

use bitarray::PyBitArray;
use block_memory::PyBlockMemory;
use block_output::PyBlockOutput;
use context_learner::PyContextLearner;
use discrete_transformer::PyDiscreteTransformer;
use pattern_classifier::PyPatternClassifier;
use pattern_pooler::PyPatternPooler;
use persistence_transformer::PyPersistenceTransformer;
use scalar_transformer::PyScalarTransformer;
use pyo3::prelude::*;

/// Python bindings for the Gnomic Computing Framework (GCF)
#[pymodule]
fn _gcf_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyBitArray>()?;
    m.add_class::<PyBlockOutput>()?;
    m.add_class::<PyBlockMemory>()?;
    m.add_class::<PyScalarTransformer>()?;
    m.add_class::<PyDiscreteTransformer>()?;
    m.add_class::<PyPersistenceTransformer>()?;
    m.add_class::<PyPatternPooler>()?;
    m.add_class::<PyPatternClassifier>()?;
    m.add_class::<PyContextLearner>()?;
    Ok(())
}
