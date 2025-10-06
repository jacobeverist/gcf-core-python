//! Error handling for Python bindings

use pyo3::exceptions::{PyIndexError, PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Convert a gnomics::GnomicsError into a Python exception
pub fn gnomics_error_to_pyerr(err: gnomics::GnomicsError) -> PyErr {
    match err {
        gnomics::GnomicsError::NotInitialized => {
            PyRuntimeError::new_err(format!("{}", err))
        }
        gnomics::GnomicsError::InvalidInputSize { .. } => {
            PyValueError::new_err(format!("{}", err))
        }
        gnomics::GnomicsError::InvalidParameter(_) => {
            PyValueError::new_err(format!("{}", err))
        }
        gnomics::GnomicsError::IndexOutOfBounds { .. } => {
            PyIndexError::new_err(format!("{}", err))
        }
        gnomics::GnomicsError::Io(_) => PyIOError::new_err(format!("{}", err)),
        gnomics::GnomicsError::Serialization(_) => {
            PyRuntimeError::new_err(format!("{}", err))
        }
        gnomics::GnomicsError::Other(_) => PyRuntimeError::new_err(format!("{}", err)),
    }
}
