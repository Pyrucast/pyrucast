#!/usr/bin/env bash
# Documentation — rustdoc sans warning, garde-fous du book, puis rendu.
#
# Il n'y a plus de pas « compiler les extraits » : depuis que **tout** bloc du
# book est un `{{#include}}`, le code affiché est celui d'un test ou d'un
# exemple, et c'est `check_rust` / `check_python` / `check_examples` qui
# l'exécutent. `mdbook test` a disparu pour la même raison — il ne compilait
# rien, tous les blocs Rust étant `rust,ignore` par construction du mécanisme
# d'inclusion.

. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

step "cargo doc --no-deps --lib (sans warning)" \
    env RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --lib
step "mdbook build"                             mdbook build book
# Six garde-fous de texte : includes qui résolvent, aucune page qui possède de
# code, prose sans symbole disparu, cliquet de couverture des doctests Rust,
# cliquet de citation de l'API Python, équations intactes. Après `cargo doc` et
# `mdbook build`, dont le quatrième et le sixième lisent la sortie ; le
# cinquième, lui, importe le module Python installé.
step "garde-fous de la documentation"           python script/doc_lint.py

echo "OK : documentation."
