# pyrucast - build minimal (Windows / PowerShell).
# Compile le module Python en release AVEC la visu interactive, puis
# regenere le stub type (python/pyrucast/_pyrucast/__init__.pyi). Rien d'autre.
# Pour le build complet (tests + doc), voir script\build.ps1.
#
# Lancer :   .\script\dev.ps1
# (si les scripts sont bloques : powershell -ExecutionPolicy Bypass -File .\script\dev.ps1)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

# Environnement virtuel (cree + active) : maturin y installe le module.
if (-not (Test-Path .venv)) { python -m venv .venv }
& .\.venv\Scripts\Activate.ps1
python -m pip install --quiet --upgrade maturin

# Compile + installe l'extension, puis regenere le stub .pyi.
maturin develop --release --features extension-module,viz-interactive
cargo run --quiet --bin stub_gen --features stub-gen

Write-Host "OK - pyrucast installe (visu interactive)."
Write-Host "Activer le venv :  .\.venv\Scripts\Activate.ps1"
