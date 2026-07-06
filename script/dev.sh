#!/usr/bin/env bash
# pyrucast — build minimal (Linux / macOS).
# Compile le module Python en release AVEC la visu interactive, puis
# régénère le stub typé (pyrucast.pyi). Rien d'autre.
# Pour le build complet (tests + doc), voir script/build.sh.

set -euo pipefail
cd "$(dirname "$0")/.."

# Environnement virtuel (créé + activé) : maturin y installe le module.
[ -d .venv ] || python3 -m venv .venv
# shellcheck source=/dev/null
. ./.venv/bin/activate
python -m pip install --quiet --upgrade maturin

# Compile + installe l'extension, puis régénère pyrucast.pyi.
maturin develop --release --features extension-module,viz-interactive
cargo run --quiet --bin stub_gen --features stub-gen

echo "OK — pyrucast installé (visu interactive)."
echo "Activer le venv :  source .venv/bin/activate"
