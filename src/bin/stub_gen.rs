//! Generate `pyrucast.pyi` — Python stub file with signatures and
//! docstrings — so IDEs (Pylance/Pyright/PyCharm) can show typed
//! completion and hover docs for the compiled extension module.
//!
//! Run with:
//!
//! ```sh
//! PYO3_PYTHON=/usr/bin/python3.13 \
//!     cargo run --bin stub_gen --features stub-gen
//! ```
//!
//! Output: `pyrucast.pyi` at the repository root.

#[cfg(feature = "stub-gen")]
fn main() -> pyo3_stub_gen::Result<()> {
    let stub = pyrucast::stub_info()?;
    stub.generate()?;
    Ok(())
}

#[cfg(not(feature = "stub-gen"))]
fn main() {
    eprintln!(
        "stub_gen requires the `stub-gen` cargo feature. \
         Rerun with: cargo run --bin stub_gen --features stub-gen"
    );
    std::process::exit(2);
}
