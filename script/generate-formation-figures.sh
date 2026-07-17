#!/usr/bin/env bash
# pyrucast — régénère les figures SVG embarquées dans les pages de la
# formation débutant (book/src/formation/*.md).
#
# Les figures sont des artefacts pré-générés, commités avec le livre :
# `mdbook build` ne connaît pas Python et ne les régénère jamais tout seul.
# Après modification d'un script formation/*.py qui appelle `.plot(...)`,
# relancer ce script et committer les SVG mis à jour.
#
# Prérequis : le module Python compilé avec la feature `viz` (export
# PNG/SVG headless) :
#     maturin develop --release --features extension-module,viz

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

bold=$(tput bold 2>/dev/null || true)
reset=$(tput sgr0 2>/dev/null || true)
step() { printf '\n%s>>> %s%s\n' "$bold" "$1" "$reset"; }
die()  { printf '\nERROR: %s\n' "$1" >&2; exit 1; }

for c in python3 python; do command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }; done
[ -n "${PY:-}" ] || die "python not found"

python -c "import pyrucast" 2>/dev/null \
    || die "pyrucast n'est pas importable — 'maturin develop --release --features extension-module,viz' d'abord"

export PYRUCAST_FORMATION_IMG_DIR="$ROOT/book/src/formation/img"
mkdir -p "$PYRUCAST_FORMATION_IMG_DIR"

for f in maillage thermique mecanique plasticite contact; do
    step "formation/$f.py"
    "$PY" "$ROOT/formation/$f.py"
done

step "Done"
echo "  Figures : $PYRUCAST_FORMATION_IMG_DIR/"
ls -la "$PYRUCAST_FORMATION_IMG_DIR"
