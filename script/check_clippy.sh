#!/usr/bin/env bash
# Clippy — le même code relu sous quatre jeux de features.

. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Quatre passes, parce qu'un avertissement peut n'exister que dans l'un des
# jeux : le code derrière un `cfg` n'est compilé que s'il est demandé. La
# 0.3.1 l'a montré — `cargo clippy --fix` lancé sur le jeu par défaut avait
# laissé intacts les `if` imbriqués de `src/viz/`, que la quatrième passe a
# rattrapés.
#
# Ce bloc n'appartient pas à `check_all` : il coûte trop cher pour la boucle
# quotidienne. Il est appelé aux deux moments où l'on pose une version —
# `set_new_version.sh` en local, et le job `verify` de `release.yml` en CI.
step "cargo clippy (défaut) -D warnings" \
    cargo clippy --all-targets -- -D warnings
step "cargo clippy --features viz -D warnings" \
    cargo clippy --all-targets --features viz -- -D warnings
step "cargo clippy --features extension-module,viz -D warnings" \
    cargo clippy --all-targets --features extension-module,viz -- -D warnings
step "cargo clippy --features extension-module,viz,viz-interactive,abi3 -D warnings" \
    cargo clippy --all-targets --features extension-module,viz,viz-interactive,abi3 -- -D warnings

echo "OK : clippy."
