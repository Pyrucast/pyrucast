#!/usr/bin/env bash
# Full Phase verification: Rust + doctests + Python + mdbook.
# Activates the venv so pyo3 and maturin find the right interpreter.

set -euo pipefail

cd "$(dirname "$0")/.."

# shellcheck source=/dev/null
. ./.venv/bin/activate

step() {
    echo ">>> $1"
    shift
    "$@"
}

step "cargo fmt --check"                         cargo fmt --check
step "ruff format --check (Python)"              ruff format --check .
step "cargo test (default features)"             cargo test
step "cargo test --doc (explicit)"               cargo test --doc
step "cargo test --features viz"                 cargo test --features viz
step "cargo build --features viz-interactive"    cargo build --features viz-interactive
step "maturin develop --features extension-module,viz" maturin develop --features extension-module,viz
step "pytest"                                    python -m pytest
step "cargo doc --no-deps --lib"                 cargo doc --no-deps --lib
step "mdbook build"                              mdbook build book
step "mdbook test"                               mdbook test book

echo "OK: all checks passed."
