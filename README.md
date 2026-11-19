# pyrucast

A finite-element library written in Rust, inspired by cast3m, used from
Python on **Linux, Windows and macOS** with a single `pip install`. It also
works as a **pure Rust** crate, with no Python involved.

[PyPI](https://pypi.org/project/pyrucast/) ·
[crates.io](https://crates.io/crates/pyrucast) ·
[Documentation (French)](https://pyrucast.github.io/pyrucast/) ·
[Rust API](https://docs.rs/pyrucast)

## Install

```bash
pip install pyrucast
```

PyPI ships **prebuilt wheels**: nothing to compile, no Rust, no C compiler.

| System | Processor | Wheel |
|---|---|---|
| Linux, glibc ≥ 2.17 | x86_64 | `manylinux2014_x86_64` |
| Windows 64-bit | x86_64 | `win_amd64` |
| macOS ≥ 10.12 (Intel), ≥ 11 (Apple Silicon) | x86_64 and arm64 | `universal2`, one wheel for both |

**The only prerequisite is CPython ≥ 3.11.** Each wheel targets Python's
stable ABI (`abi3`), so one wheel per platform serves Python 3.11, 3.12,
3.13 and later versions. pyrucast has no required Python dependency.

What the wheels contain, and what they expect from the system:

- **Visualization is included**: `plot()` exports PNG and SVG, and opens an
  interactive window when called without `save=`.
- **Linux**: the wheel bundles its own fontconfig and freetype. It needs only
  glibc, libgcc, zlib and expat from the system, which every common
  distribution provides. The interactive window needs a display (X11 or Wayland). Export
  to PNG/SVG works without one, on a server, in CI or in a container.
- **Windows, macOS**: nothing to install. The wheel uses only system
  libraries and the runtime that comes with Python.

Check the installation:

```bash
python -c "import pyrucast; print(pyrucast.__version__, pyrucast.__features__)"
```

`viz` in `__features__` means plotting is compiled in.

**Other platforms** (Linux ARM, musl/Alpine, BSD…): `pip` finds no wheel and
compiles the source distribution itself. This needs Rust
([rustup](https://rustup.rs)), and the result has **no visualization**
(`__features__` lacks `viz`, `plot()` does not exist). To get plotting there,
build from the repository, as described in [Build from source](#build-from-source).

## A first analysis

A 10 cm steel plate, heated throughout, its edge held at 0 °C:

```python
import pyrucast as pc

coords = pc.Coords(2)
a = coords.add_node([0.0, 0.0])
b = coords.add_node([0.10, 0.0])
plate = pc.mesh.extrude(pc.mesh.line(a, b, 20), [0.0, 0.10], 20)  # 20 × 20 QUA4
edge = pc.mesh.to_poi1(pc.mesh.border(plate))

fes = pc.FiniteElementSpace(plate)
model = pc.model.heat_conduction(fes)
model = model | pc.model.dirichlet(edge, model, "T")
model = model | pc.model.flux(fes, model, "q")
materials = pc.element_field.material_field(model, [("k", 50.0), ("phi_q", 1.0e6)])

T = pc.solver.solve(
    pc.matrix.stiffness(model, materials),
    pc.node_field.external_forces(model, materials),
)
print(f"T max = {T.max('T'):.2f} °C")  # T max = 14.76 °C
plate.plot(field=T, component="T", save="plate.png")  # omit save= for a window
```

The [tutorial](https://pyrucast.github.io/pyrucast/tutoriel/debutant.html)
(in French) builds on this, step by step, from meshing to nonlinear
mechanics. Complete scripts are in
[`examples/`](https://github.com/Pyrucast/pyrucast/tree/master/examples) and
[`tutoriel/`](https://github.com/Pyrucast/pyrucast/tree/master/tutoriel).

## Documentation

- **The book**, in French: <https://pyrucast.github.io/pyrucast/>. It covers
  installation, the tutorial, the object model, meshing, fields, physics and
  development. It is built from `master`, so it can describe features newer
  than the latest release.
- **Rust API reference**: <https://docs.rs/pyrucast> for the published
  versions, <https://pyrucast.github.io/pyrucast/rust/> for `master`, or
  locally `cargo doc --no-deps --lib --open`.
- **In Python**: `help(pyrucast.mesh.line)`. The wheel ships a type stub, so
  editors show signatures and docstrings.

## Use from Rust

```bash
cargo add pyrucast
```

```rust
use pyrucast::atoms::ElementType;
// `contour`: a closed line mesh around the area to fill.
let mesh = pyrucast::ops::mesh::triangulate_surface(&contour, ElementType::TRI3, Some(1.0))?;
```

No feature is on by default, so the crate is pure Rust: it pulls in neither
`pyo3` nor `libpython`, and needs no Python to build or test. It requires
Rust ≥ 1.89 (edition 2024). The optional features are `viz` (PNG/SVG export)
and `viz-interactive` (adds the window). The Python binding lives behind
`python-api`, which `maturin` turns on.

## Build from source

You only need this for the development version, or to get plotting on a
platform without a wheel.

| Tool | Version | Notes |
|---|---|---|
| Git | — | to clone the repository |
| Rust | ≥ 1.89, edition 2024 | via [`rustup`](https://rustup.rs) |
| Python | ≥ 3.11 | |
| Linux only | — | `pkg-config`, plus the fontconfig and freetype headers for the visualization: `libfontconfig1-dev libfreetype6-dev` (Debian/Ubuntu) or `fontconfig-devel freetype-devel` (Fedora/RHEL). Add the Python headers, `python3-dev` / `python3-devel`, for builds that link `libpython` (`cargo --features python-api`, the stub generator). |

Windows and macOS need nothing beyond the first three lines. `maturin`
requires an **activated venv**: without one it fails with "Couldn't find a
virtualenv or conda environment".

Linux / macOS (bash):

```bash
git clone https://github.com/Pyrucast/pyrucast
cd pyrucast
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip maturin
maturin develop --release --features extension-module,viz-interactive
```

Windows (PowerShell):

```powershell
git clone https://github.com/Pyrucast/pyrucast
cd pyrucast
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip maturin
maturin develop --release --features extension-module,viz-interactive
```

`maturin develop` compiles the module and installs it into the venv, in
editable mode. After any change to the Rust code, re-run **the same
command**: a plain `maturin develop` builds without the visualization.
`bash script/dev.sh` (or `.\script\dev.ps1`) does the same build and also
regenerates the type stub.

## Development

Every script in `script/` exists in bash (`.sh`) and PowerShell (`.ps1`).

```bash
pip install pytest ruff==0.16.4   # on top of maturin, in the venv
cargo build                       # pure Rust core (no pyo3, no venv needed)
cargo test                        # unit + integration tests + doctests
maturin develop --release --features extension-module,viz && python -m pytest
cargo fmt && ruff format .        # formatting (Rust + Python)
bash script/check_all.sh          # every check, formatting included
bash script/check_rust.sh         # or one block: format / rust / python / examples / doc
```

The book's chapter
[Compilation et tests](https://pyrucast.github.io/pyrucast/compilation.html)
details the Cargo features, the scripts, the generation of the `.pyi` stub and
common troubleshooting.

## License

Distributed under the
**[Mozilla Public License 2.0](https://github.com/Pyrucast/pyrucast/blob/master/LICENSE)**
(MPL-2.0).

The copyleft applies file by file. If you distribute a modified pyrucast
source file, it stays under the MPL and its source must be made available.
Your own code, built on top of the library or combined with it, can be under
any license, proprietary included. The MPL also allows combining pyrucast with
code under the GPL, LGPL or AGPL ("Secondary Licenses", §3.3).
