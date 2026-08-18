#!/usr/bin/env bash
# Documentation — rustdoc sans warning, extraits Rust du book compilés,
# puis mdbook.
#
# Le pas qui n'existait pas : `mdbook test` ne compile **rien**, tous les
# blocs Rust du book étant `rust,ignore`. Un module renommé ou une signature
# changée pouvait donc y pourrir indéfiniment. `cargo run --bin book_blocks`
# rassemble les extraits qui sont de vrais programmes dans un fichier de test
# jetable, que l'on type-vérifie ici.
#
# Les extraits restent des extraits : beaucoup nomment une variable que la
# page ne définit jamais. On ignore donc les codes d'erreur qui signalent un
# fragment, et on échoue sur tout le reste — dont les erreurs de syntaxe, qui
# rendraient le garde-fou aveugle si on les laissait passer.

. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Codes tolérés (le bloc est un extrait, pas l'API qui a bougé) :
#   E0425 valeur/fonction inconnue      E0412 type inconnu
#   E0405 trait inconnu                 E0422 structure inconnue
#   E0433 module non résolu             E0252 import en double
#   E0277 `?` hors d'une fn -> Result   E0423 macro prise pour une fonction
FRAGMENT='error\[(E0425|E0412|E0405|E0422|E0433|E0252|E0277|E0423)\]'

check_book_blocks() {
    cargo run --quiet --bin book_blocks
    local out
    out=$(cargo check --test book_blocks --features viz,book-check --message-format short 2>&1 || true)
    local fatal
    fatal=$(printf '%s\n' "$out" | grep -E '^tests/book_blocks\.rs.*error' | grep -vE "$FRAGMENT" || true)
    if [ -n "$fatal" ]; then
        echo "L'API a bougé sous les extraits du book :" >&2
        printf '%s\n' "$fatal" >&2
        echo >&2
        echo "Chaque ligne pointe tests/book_blocks.rs ; le commentaire '// --- page:ligne'" >&2
        echo "qui précède donne la page du book et la ligne d'origine." >&2
        return 1
    fi
}

step "cargo doc --no-deps --lib (sans warning)" \
    env RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --lib
step "extraits Rust du book (compilés)"         check_book_blocks
step "mdbook build"                             mdbook build book
step "mdbook test"                              mdbook test book

echo "OK : documentation."
