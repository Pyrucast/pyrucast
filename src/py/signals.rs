//! Cancellation token bridging Python's signal handling to the PyO3-free
//! operator core ([`crate::interrupt::Cancel`]).
//!
//! This is the single place where Python interruption meets the operators:
//! they take a `&dyn Cancel`, never `Python`, so the core stays usable from
//! pure Rust. See the book chapter "Interrompre une fonction".

use crate::error::{PyrucastError, Result};
use crate::interrupt::Cancel;
use pyo3::prelude::*;

/// A [`Cancel`] token that polls Python's pending signals: a `Ctrl+C`
/// (`SIGINT`) raised during a long operator turns into
/// [`PyrucastError::Interrupted`] → `KeyboardInterrupt`.
pub struct PySignals<'py>(pub Python<'py>);

impl Cancel for PySignals<'_> {
    fn check(&self) -> Result<()> {
        self.0
            .check_signals()
            .map_err(|_| PyrucastError::Interrupted)
    }
}
