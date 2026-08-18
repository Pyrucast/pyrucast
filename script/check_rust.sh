#!/usr/bin/env bash
# Cœur Rust — tests unitaires, d'intégration, doctests, et la compilation
# des couches optionnelles (visualisation) qu'aucun test ne couvre.

. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

step "cargo test (features par défaut)"      cargo test
step "cargo test --doc (explicite)"          cargo test --doc
step "cargo test --features viz"             cargo test --features viz
step "cargo build --features viz-interactive" cargo build --features viz-interactive

echo "OK : Rust."
