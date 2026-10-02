//! Helpers shared by the integration tests: a file under `tests/` that needs
//! them declares `mod common;`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A fresh directory under the system's temporary directory, removed with
/// everything in it when dropped: at the end of the test, and also while a
/// failing test unwinds.
///
/// Every call gets a directory of its own (process id plus a counter), so
/// tests running in parallel never share one, and none can remove another's.
pub struct TempDir(PathBuf);

impl TempDir {
    /// Create `<temp dir>/<prefix>_<pid>_<n>`.
    pub fn new(prefix: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("{prefix}_{}_{n}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl std::ops::Deref for TempDir {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        // Best effort: a directory that cannot be removed must not turn a
        // passing test into a failing one.
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
