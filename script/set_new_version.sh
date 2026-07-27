#!/usr/bin/env bash
# pyrucast — passe le projet à une nouvelle version.
#
# 1. vérifie que tout est vert et SANS WARNING (fmt, clippy, tests Rust,
#    doctests, tests Python, build de la doc rustdoc/mdbook) ;
# 2. demande le nouveau numéro de version ;
# 3. le reporte dans Cargo.toml et pyproject.toml ;
# 4. commit + tag annoté `vX.Y.Z` sur master.
#
# Ne pousse rien : `git push && git push origin vX.Y.Z` reste un geste manuel.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

bold=$(tput bold 2>/dev/null || true)
reset=$(tput sgr0 2>/dev/null || true)
step() { printf '\n%s>>> %s%s\n' "$bold" "$1" "$reset"; }
die()  { printf '\nERREUR: %s\n' "$1" >&2; exit 1; }

# ── 0. Préconditions git ─────────────────────────────────────────────────────
branch="$(git rev-parse --abbrev-ref HEAD)"
[ "$branch" = "master" ] || die "il faut être sur master (actuellement: $branch)"
[ -z "$(git status --porcelain)" ] || die "arbre de travail non propre — commit/stash d'abord :
$(git status --porcelain)"

# shellcheck source=/dev/null
. ./.venv/bin/activate

# ── 1. Vérifications complètes, sans warning ────────────────────────────────
step "cargo fmt --check"                                        ; cargo fmt --check
step "ruff format --check (Python)"                              ; ruff format --check .

step "cargo clippy (défaut) -D warnings"
cargo clippy --all-targets -- -D warnings
step "cargo clippy --features viz -D warnings"
cargo clippy --all-targets --features viz -- -D warnings
step "cargo clippy --features extension-module,viz -D warnings"
cargo clippy --all-targets --features extension-module,viz -- -D warnings
step "cargo clippy --features extension-module,viz,viz-interactive,abi3 -D warnings"
cargo clippy --all-targets --features extension-module,viz,viz-interactive,abi3 -- -D warnings

step "cargo test (défaut)"                                       ; cargo test
step "cargo test --doc"                                          ; cargo test --doc
step "cargo test --features viz"                                 ; cargo test --features viz
step "cargo build --features viz-interactive"                    ; cargo build --features viz-interactive

step "maturin develop --features extension-module,viz"          ; maturin develop --features extension-module,viz
step "pytest"                                                     ; python -m pytest

step "cargo doc --no-deps --lib (sans warning)"
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --lib
step "mdbook build"                                               ; mdbook build book
step "mdbook test"                                                ; mdbook test book

echo
echo "OK: tout est vert, sans warning."

# ── 2. Nouveau numéro de version ────────────────────────────────────────────
current="$(sed -nE 's/^version = "(.*)"/\1/p' Cargo.toml | head -1)"

echo
echo "Version actuelle : $current"
read -rp "Nouvelle version : " new_version

[ -n "$new_version" ] || die "version vide"
[[ "$new_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.]+)?$ ]] \
    || die "format de version invalide (attendu: X.Y.Z) : $new_version"
git rev-parse -q --verify "refs/tags/v$new_version" >/dev/null \
    && die "le tag v$new_version existe déjà"

read -rp "Confirmer le passage de $current à $new_version sur master ? [o/N] " confirm
[[ "$confirm" =~ ^[oO]$ ]] || die "annulé"

# ── 3. Mise à jour des fichiers ─────────────────────────────────────────────
step "Mise à jour de Cargo.toml et pyproject.toml"
sed -i -E "s/^version = \".*\"/version = \"$new_version\"/" Cargo.toml
sed -i -E "s/^version = \".*\"/version = \"$new_version\"/" pyproject.toml

step "cargo check (régénère Cargo.lock)"
cargo check --quiet

# ── 4. Commit + tag ──────────────────────────────────────────────────────────
step "git commit + tag v$new_version"
git add Cargo.toml pyproject.toml
git commit -m "chore: version $new_version"
git tag -a "v$new_version" -m "pyrucast v$new_version"

echo
echo "Fait : commit + tag v$new_version créés sur master (en local uniquement)."
echo "Pour publier : git push && git push origin v$new_version"
