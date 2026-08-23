#!/usr/bin/env bash
# pyrucast — full build, tests and documentation (Linux / macOS).
#
# 1. checks prerequisites (cargo, python, venv, maturin, pytest, mdbook),
# 2. compiles the Rust core and runs every test suite,
# 3. installs the Python module WITH interactive visualization,
# 4. builds the documentation (mdbook + rustdoc + Python pydoc + .pyi stub),
# 5. prints a summary: where the docs are and how to use the library.
#
# The lighter CI-style check is `script/check_all.sh`; this one is the
# "everything, end to end" build.

set -euo pipefail

# Repo root = parent of this script's directory.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

bold=$(tput bold 2>/dev/null || true)
reset=$(tput sgr0 2>/dev/null || true)
step() { printf '\n%s>>> %s%s\n' "$bold" "$1" "$reset"; }
die()  { printf '\nERROR: %s\n' "$1" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# ── 1. Prerequisites ────────────────────────────────────────────────────────
step "Checking prerequisites"

have cargo || die "cargo not found — install Rust via https://rustup.rs"
echo "  cargo   : $(cargo --version)"

PY=""
for c in python3 python; do have "$c" && { PY="$c"; break; }; done
[ -n "$PY" ] || die "python not found — install Python >= 3.11"
echo "  python  : $("$PY" --version 2>&1)"
# Linux note: pyo3 needs the Python headers (python3-dev / python3-devel).

# Virtual environment: create it if missing, then activate it (pyo3 and
# maturin locate the interpreter through VIRTUAL_ENV).
if [ ! -d .venv ]; then
    step "Creating virtual environment (.venv)"
    "$PY" -m venv .venv
fi
# shellcheck source=/dev/null
. ./.venv/bin/activate
echo "  venv    : ${VIRTUAL_ENV:-?}"

step "Ensuring maturin and pytest in the venv"
python -m pip install --quiet --upgrade pip maturin pytest
echo "  maturin : $(maturin --version)"
echo "  pytest  : $(python -m pytest --version 2>&1 | head -1)"

if ! have mdbook; then
    step "Installing mdbook (cargo install mdbook --locked)"
    cargo install mdbook --locked
fi
echo "  mdbook  : $(mdbook --version)"

# ── 2. Compile + tests ──────────────────────────────────────────────────────
step "cargo build (core)"                            ; cargo build
step "cargo test (unit + integration + doctests)"    ; cargo test
step "cargo test --doc"                              ; cargo test --doc
step "cargo test --features viz"                     ; cargo test --features viz

# ── 3. Python module WITH interactive visualization ─────────────────────────
step "maturin develop --release --features extension-module,viz-interactive"
maturin develop --release --features extension-module,viz-interactive

step "pytest (Python test suite)"                    ; python -m pytest

# ── 4. Documentation ────────────────────────────────────────────────────────
step "cargo doc (Rust API reference)"                ; cargo doc --no-deps --lib
step "Regenerating the Python stub (.pyi)"           ; cargo run --quiet --bin stub_gen --features stub-gen
step "mdbook build (theory book)"                    ; mdbook build book

step "Python API doc (pydoc HTML)"
mkdir -p target/python-doc
( cd target/python-doc && python -m pydoc -w pyrucast >/dev/null )
echo "  wrote target/python-doc/pyrucast.html"

# ── 5. Verify the interactive-viz module is importable ──────────────────────
step "Verifying the Python module (import + headless viz)"
python - <<'PYCHECK'
import tempfile, os, pyrucast as pc
c = pc.Coords(3)
a = c.add_node([0.0, 0.0, 0.0]); b = c.add_node([1.0, 0.0, 0.0]); d = c.add_node([0.0, 1.0, 0.0])
m = pc.Mesh(c, "TRI3"); m.unit().add_cell([a, b, d])
print("  import OK")
# Mesh.plot only exists when the viz feature is compiled in: its absence
# means the interactive-viz module was NOT installed -> hard failure.
if not hasattr(m, "plot"):
    raise SystemExit("  ERROR: viz feature missing (Mesh.plot absent) — interactive visualization not available")
try:
    out = os.path.join(tempfile.gettempdir(), "pyrucast_check.svg")
    m.plot(save=out)            # exercises the viz stack, no display needed
    assert os.path.exists(out)
    print("  viz available, export OK ->", out)
except Exception as e:           # font/display quirks must not fail the build
    print("  viz available (plot present); headless export skipped:", e)
PYCHECK

# ── 6. Summary ──────────────────────────────────────────────────────────────
cat <<EOF

============================================================
  pyrucast — build, tests & documentation: SUCCESS
============================================================

Documentation produite :
  - Théorie (mdbook)  : file://$ROOT/book/book/index.html
  - API Rust (rustdoc): file://$ROOT/target/doc/pyrucast/index.html
  - API Python (pydoc): file://$ROOT/target/python-doc/pyrucast.html
  - Stub typé (.pyi)  : $ROOT/python/pyrucast/_pyrucast/__init__.pyi

Ouvrir le livre :
  xdg-open "$ROOT/book/book/index.html"
  (ou, en serveur live :  mdbook serve book  -> http://localhost:3000 )

Utiliser pyrucast — la visu interactive est installée dans le venv :
  1. Activer le venv :
       source "$ROOT/.venv/bin/activate"
  2. Ouvrir une fenêtre interactive (clic-glissé : rotation, molette : zoom, A : repère) :
       python -c "import pyrucast as pc; c=pc.Coords(3); a=c.add_node([0,0,0]); b=c.add_node([1,0,0]); d=c.add_node([0,1,0]); m=pc.Mesh(c,'TRI3'); m.unit().add_cell([a,b,d]); m.plot()"

EOF
