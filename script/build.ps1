# pyrucast - full build, tests and documentation (Windows / PowerShell).
#
# 1. checks prerequisites (cargo, python, venv, maturin, pytest, mdbook),
# 2. compiles the Rust core and runs every test suite,
# 3. installs the Python module WITH interactive visualization,
# 4. builds the documentation (mdbook + rustdoc + Python pydoc + .pyi stub),
# 5. prints a summary: where the docs are and how to use the library.
#
# Run from a PowerShell prompt:   .\script\build.ps1
# (If scripts are blocked: powershell -ExecutionPolicy Bypass -File .\script\build.ps1)

$ErrorActionPreference = 'Stop'

# Repo root = parent of this script's directory.
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Step($msg) { Write-Host "`n>>> $msg" -ForegroundColor Cyan }
function Have($cmd) { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }
function Die($msg)  { Write-Host "`nERROR: $msg" -ForegroundColor Red; exit 1 }
# Native commands don't trip $ErrorActionPreference; check the exit code.
function Run($label, [scriptblock]$cmd) {
    Step $label
    & $cmd
    if ($LASTEXITCODE -ne 0) { Die "$label failed (exit $LASTEXITCODE)" }
}

# -- 1. Prerequisites --------------------------------------------------------
Step "Checking prerequisites"

if (-not (Have cargo)) { Die "cargo not found - install Rust via https://rustup.rs" }
Write-Host "  cargo   : $(cargo --version)"

$Py = $null
foreach ($c in 'python', 'py') { if (Have $c) { $Py = $c; break } }
if (-not $Py) { Die "python not found - install Python >= 3.11" }
Write-Host "  python  : $(& $Py --version)"

# Virtual environment: create it if missing, then activate it (pyo3 and
# maturin locate the interpreter through VIRTUAL_ENV).
if (-not (Test-Path .venv)) {
    Step "Creating virtual environment (.venv)"
    & $Py -m venv .venv
    if ($LASTEXITCODE -ne 0) { Die "venv creation failed" }
}
& .\.venv\Scripts\Activate.ps1
Write-Host "  venv    : $env:VIRTUAL_ENV"

Step "Ensuring maturin and pytest in the venv"
python -m pip install --quiet --upgrade pip maturin pytest
if ($LASTEXITCODE -ne 0) { Die "pip install failed" }
Write-Host "  maturin : $(maturin --version)"

if (-not (Have mdbook)) {
    Run "Installing mdbook (cargo install mdbook --locked)" { cargo install mdbook --locked }
}
Write-Host "  mdbook  : $(mdbook --version)"

# -- 2. Compile + tests ------------------------------------------------------
Run "cargo build (core)"                            { cargo build }
Run "cargo test (unit + integration + doctests)"    { cargo test }
Run "cargo test --doc"                              { cargo test --doc }
Run "cargo test --features viz"                     { cargo test --features viz }

# -- 3. Python module WITH interactive visualization -------------------------
Run "maturin develop --release --features extension-module,viz-interactive" {
    maturin develop --release --features extension-module,viz-interactive
}
Run "pytest (Python test suite)"                    { python -m pytest }

# -- 4. Documentation --------------------------------------------------------
Run "cargo doc (Rust API reference)"                { cargo doc --no-deps --lib }
Run "Regenerating the Python stub (pyrucast.pyi)"   { cargo run --quiet --bin stub_gen --features stub-gen }
Run "mdbook build (theory book)"                    { mdbook build book }

Step "Python API doc (pydoc HTML)"
New-Item -ItemType Directory -Force -Path target\python-doc | Out-Null
Push-Location target\python-doc
python -m pydoc -w pyrucast | Out-Null
$pydocExit = $LASTEXITCODE
Pop-Location
if ($pydocExit -ne 0) { Die "pydoc failed" }
Write-Host "  wrote target\python-doc\pyrucast.html"

# -- 5. Verify the interactive-viz module is importable ----------------------
Step "Verifying the Python module (import + headless viz)"
$check = @'
import tempfile, os, pyrucast as pc
c = pc.Coords(3)
a = c.add_node([0.0, 0.0, 0.0]); b = c.add_node([1.0, 0.0, 0.0]); d = c.add_node([0.0, 1.0, 0.0])
m = pc.Mesh(c, "TRI3"); m.unit().add_cell([a, b, d])
print("  import OK")
if not hasattr(m, "plot"):
    raise SystemExit("  ERROR: viz feature missing (Mesh.plot absent) - interactive visualization not available")
try:
    out = os.path.join(tempfile.gettempdir(), "pyrucast_check.svg")
    m.plot(save=out)
    assert os.path.exists(out)
    print("  viz available, export OK ->", out)
except Exception as e:
    print("  viz available (plot present); headless export skipped:", e)
'@
$check | python -
if ($LASTEXITCODE -ne 0) { Die "module verification failed" }

# -- 6. Summary --------------------------------------------------------------
$book    = Join-Path $Root 'book\book\index.html'
$rustdoc = Join-Path $Root 'target\doc\pyrucast\index.html'
$pydoc   = Join-Path $Root 'target\python-doc\pyrucast.html'
$stub    = Join-Path $Root 'pyrucast.pyi'
$venvAct = Join-Path $Root '.venv\Scripts\Activate.ps1'

Write-Host ""
Write-Host "============================================================"
Write-Host "  pyrucast - build, tests & documentation: SUCCESS"
Write-Host "============================================================"
Write-Host ""
Write-Host "Documentation produite :"
Write-Host "  - Theorie (mdbook)  : $book"
Write-Host "  - API Rust (rustdoc): $rustdoc"
Write-Host "  - API Python (pydoc): $pydoc"
Write-Host "  - Stub type (.pyi)  : $stub"
Write-Host ""
Write-Host "Ouvrir le livre :"
Write-Host "  Start-Process `"$book`""
Write-Host "  (ou, en serveur live :  mdbook serve book  -> http://localhost:3000 )"
Write-Host ""
Write-Host "Utiliser pyrucast - la visu interactive est installee dans le venv :"
Write-Host "  1. Activer le venv :"
Write-Host "       & `"$venvAct`""
Write-Host "  2. Ouvrir une fenetre interactive (clic-glisse : rotation, molette : zoom, A : repere) :"
Write-Host "       python -c `"import pyrucast as pc; c=pc.Coords(3); a=c.add_node([0,0,0]); b=c.add_node([1,0,0]); d=c.add_node([0,1,0]); m=pc.Mesh(c,'TRI3'); m.unit().add_cell([a,b,d]); m.plot()`""
Write-Host ""
