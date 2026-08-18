#!/usr/bin/env bash
# Formatage — Rust et Python. Ne formate rien : il vérifie.
#
# À lancer APRÈS `cargo fmt` et `.venv/bin/ruff format .`, jamais avant :
# le but est d'attraper ce qui part en commit mal formaté.

. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

step "cargo fmt --check"            cargo fmt --check
step "ruff format --check (Python)" ruff format --check .

echo "OK : formatage."
