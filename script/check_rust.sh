#!/usr/bin/env bash
# Cœur Rust — tests unitaires, d'intégration et doctests, toutes couches
# optionnelles compilées, plus un contrôle du build sans feature.

. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Trois pas, un rôle chacun, et le choix du jeu de features n'est pas neutre.
#
# `cargo test` exécute déjà les doctests : les relancer par un `--doc` explicite
# était une pure répétition, qui coûtait quatre secondes quand il y en avait 31
# et deux cents aujourd'hui.
#
# Les doctests tournent sous `viz`, **pas** sous `viz-interactive`. Chacun est
# un binaire séparé, lié contre tout le graphe de dépendances : 120 crates sous
# `viz`, 178 sous `viz-interactive` — winit, Wayland, X11, tiny-skia. Mesuré sur
# un même lot, 2,88 s contre 4,68 s ; sur les ~890, 237 s contre 364 s.
#
# La couche interactive est donc **compilée sans être testée** : elle ne porte
# aucun test (960 lignes de fenêtre winit, que rien ne peut exercer sans écran),
# et un `build` suffit à garantir qu'elle compile encore. Il coûte 6 s quand les
# deux jeux sont en cache, 30 s au premier basculement.
#
# Le `check` par défaut, enfin : la bibliothèque se veut du **Rust pur** sans
# feature, et sans lui plus rien ne compilerait cette configuration-là.
step "cargo check (features par défaut)"       cargo check --all-targets
step "cargo test --features viz"               cargo test --features viz
step "cargo build --features viz-interactive"  cargo build --features viz-interactive

echo "OK : Rust."
