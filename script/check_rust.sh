#!/usr/bin/env bash
# Cœur Rust — tests unitaires, d'intégration et doctests, toutes couches
# optionnelles compilées, plus un contrôle du build sans feature.

. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# `viz-interactive` implique `viz`, et `cargo test` exécute déjà les doctests :
# un seul jeu de features couvre ce que quatre pas couvraient, sans les
# recompiler trois fois ni relancer les ~890 doctests à chaque fois.
#
# Le `check` par défaut est le filet qui reste : la bibliothèque se veut du
# **Rust pur** sans feature, et sans lui plus rien ne compilerait cette
# configuration-là. Il coûte 25 s, contre 200 s pour un `test` complet.
step "cargo check (features par défaut)"      cargo check --all-targets
step "cargo test --features viz-interactive"  cargo test --features viz-interactive

echo "OK : Rust."
