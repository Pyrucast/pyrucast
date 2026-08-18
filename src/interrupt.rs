//! Cooperative cancellation for long-running operators.
//!
//! Long operators (meshers, solvers, refinement loops) can run for a while.
//! To let a caller stop them early, they take a [`Cancel`] token and poll it
//! periodically; returning [`PyrucastError::Interrupted`] aborts the
//! computation cleanly (no partial result is committed).
//!
//! The key design point is the **Rust/Python boundary**: [`Cancel`] is a
//! plain Rust trait with **no dependency on PyO3**. The *operator* never
//! knows how cancellation is signalled — that is the frontend's job:
//!
//! - **Pure Rust** callers pass [`NoCancel`] (never cancels), a
//!   [`Deadline`] (timeout), or an [`std::sync::atomic::AtomicBool`] flipped
//!   by their own `Ctrl+C` handler (e.g. the `ctrlc` crate) or a control
//!   thread. The same `AtomicBool` is the natural way to cancel
//!   shared-memory parallel work, where each worker polls the flag.
//! - The **Python** binding (under the `python-api` feature, in `src/py`)
//!   provides its own token that calls `Python::check_signals`, turning a
//!   `Ctrl+C` (`SIGINT`) into [`PyrucastError::Interrupted`] →
//!   `KeyboardInterrupt`. That token lives entirely in the FFI layer, so the
//!   operator core stays PyO3-free and usable from a pure Rust program.

use crate::error::{PyrucastError, Result};

/// A cancellation check polled periodically by long-running operators.
///
/// `check` returns `Ok(())` to continue, or `Err(PyrucastError::Interrupted)`
/// to request a clean stop. Implementations decide *how* cancellation is
/// signalled, keeping operators free of any frontend concern.
// ANCHOR: trait
pub trait Cancel {
    /// Poll the cancellation state. Called frequently, so keep it cheap.
    fn check(&self) -> Result<()>;
}
// ANCHOR_END: trait

/// Forward through shared references, so `&token` is itself a [`Cancel`].
impl<C: Cancel + ?Sized> Cancel for &C {
    fn check(&self) -> Result<()> {
        (**self).check()
    }
}

/// Never cancels. The default for Rust callers that don't need it.
///
/// ```
/// # use pyrucast::interrupt::{Cancel, NoCancel};
/// // Le jeton qui ne s'arme jamais : ce que passent les fonctions publiques.
/// assert!(NoCancel.check().is_ok());
/// ```
pub struct NoCancel;

impl Cancel for NoCancel {
    #[inline]
    fn check(&self) -> Result<()> {
        Ok(())
    }
}

/// `()` also works as a no-op token, for the terse call site `&()`.
impl Cancel for () {
    #[inline]
    fn check(&self) -> Result<()> {
        Ok(())
    }
}

/// Cancel as soon as the flag reads `true` — set it from a signal handler, a
/// control thread, or (later) the supervisor of a parallel region.
impl Cancel for std::sync::atomic::AtomicBool {
    #[inline]
    fn check(&self) -> Result<()> {
        if self.load(std::sync::atomic::Ordering::Relaxed) {
            Err(PyrucastError::Interrupted)
        } else {
            Ok(())
        }
    }
}

/// Cancel once a deadline has passed (a wall-clock timeout).
pub struct Deadline(pub std::time::Instant);

impl Deadline {
    /// A deadline `dur` from now.
    ///
    /// ```
    /// # use pyrucast::interrupt::{Cancel, Deadline};
    /// # use std::time::Duration;
    /// // Échéance déjà passée : le premier point de contrôle s'arrête.
    /// let echu = Deadline::after(Duration::from_millis(0));
    /// std::thread::sleep(Duration::from_millis(1));
    /// assert!(echu.check().is_err());
    /// ```
    pub fn after(dur: std::time::Duration) -> Self {
        Deadline(std::time::Instant::now() + dur)
    }
}

impl Cancel for Deadline {
    #[inline]
    fn check(&self) -> Result<()> {
        if std::time::Instant::now() >= self.0 {
            Err(PyrucastError::Interrupted)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn no_cancel_never_trips() {
        assert!(NoCancel.check().is_ok());
        assert!(().check().is_ok());
    }

    #[test]
    fn atomic_flag_trips_when_set() {
        let flag = AtomicBool::new(false);
        assert!(flag.check().is_ok());
        flag.store(true, Ordering::Relaxed);
        assert!(matches!(flag.check(), Err(PyrucastError::Interrupted)));
        // Through a shared reference too.
        let r = &flag;
        assert!(matches!(r.check(), Err(PyrucastError::Interrupted)));
    }

    #[test]
    fn deadline_trips_in_the_past() {
        let past = Deadline(std::time::Instant::now() - std::time::Duration::from_secs(1));
        assert!(matches!(past.check(), Err(PyrucastError::Interrupted)));
        let future = Deadline::after(std::time::Duration::from_secs(3600));
        assert!(future.check().is_ok());
    }
}
