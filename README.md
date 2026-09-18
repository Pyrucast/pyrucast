# pyrucast

A finite element library in Rust, inspired by cast3m, exposed to Python — and
usable as is in **pure Rust** (`pyo3` is an optional dependency: a default build
pulls in neither `pyo3` nor `libpython`).

- **Full documentation**: the mdbook in [`book/`](book/) (architecture, memory
  model, meshing, fields, physics…) — `mdbook build book` for the HTML.
- **Rust API reference**: `cargo doc --no-deps --lib --open`.

## Prerequisites

| Tool | Version | Installation |
|---|---|---|
| Rust | ≥ 1.89 (the crate's `rust-version`), edition 2024 | [`rustup`](https://rustup.rs) |
| Python | ≥ 3.11 | *Python API only* — not needed in pure Rust |
| Python headers | — | *Python API only* — **Linux**: `python3-dev` (Debian/Ubuntu) or `python3-devel` (Fedora/RHEL). Windows: included in the official installer. |

## Use in pure Rust (without Python)

```toml
[dependencies]
pyrucast = { git = "…", default-features = false }   # no pyo3, no libpython
```

```rust
use pyrucast::atoms::ElementType;
let mesh = pyrucast::ops::mesh::triangulate_surface(&contour, ElementType::TRI3, Some(1.0))?;
```

`cargo build` / `cargo test` (without any feature) compile the core in pure Rust
— neither Python nor a venv required. The Python API (`#[pyclass]`) lives behind
the `python-api` feature, enabled automatically by `maturin`.

## Building in a venv (Python API)

For builds **with the Python API** (`maturin`, or `cargo --features
python-api`), `pyo3` locates the interpreter through `VIRTUAL_ENV`: **always
activate the venv before `cargo` or `maturin`**, otherwise the build fails with
`error: failed to run the Python interpreter at ...`. (A plain `cargo build` does
not have this constraint.)

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip maturin
maturin develop --release
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip maturin
maturin develop --release
```

`maturin develop` compiles the module and installs it into the venv. Check:

```bash
python -c "import pyrucast; c = pyrucast.Coords(2); print(c)"
```

After any change to the Rust, simply re-run `maturin develop --release`.

## Development

```bash
pip install pytest ruff     # on top of maturin, in the venv
cargo build                 # pure Rust core (without pyo3; no venv needed)
cargo test                  # unit + integration tests + doctests (pure Rust)
maturin develop && python -m pytest   # Python tests
cargo fmt && ruff format .  # standard formatting (Rust + Python)
bash script/check_all.sh    # chains all the checks (formatting included)
bash script/check_doc.sh    # or a single block: format / rust / python / examples / doc
```

The book's chapter [Building and testing](book/src/compilation.md) details the
Cargo features (`viz`, `viz-interactive`, `stub-gen`…), the generation of the
`python/pyrucast/_pyrucast/__init__.pyi` stub and the common troubleshooting.

## License

Distributed under the **[Mozilla Public License 2.0](LICENSE)** (MPL-2.0).

It is a *per-file* copyleft: any modification of a pyrucast source file must be
republished under the MPL, but one can freely build proprietary code **on top
of** the library and combine it with it. The MPL is moreover GPL/CeCILL
compatible ("Secondary Licenses", §3.3).
