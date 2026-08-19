#!/usr/bin/env bash
# pyrucast — passe le projet à une nouvelle version.
#
# 1. vérifie que tout est vert et SANS WARNING : `check_all.sh` en entier,
#    plus les quatre passes de clippy qu'aucun bloc ne couvre ;
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
# Les vérifications **appellent** les blocs au lieu de les recopier. La copie
# précédente avait divergé, et pas seulement en durée : elle relançait les
# doctests trois fois, mais surtout elle ne lançait ni les cinq garde-fous de
# documentation ni les exemples de bout en bout — au moment précis où l'on pose
# un tag.
step "check_all (formatage, Rust, Python, exemples, documentation)"
bash script/check_all.sh

# Clippy, en revanche, n'appartient à aucun bloc : les quatre passes coûtent
# trop cher pour la boucle quotidienne, et c'est ici qu'elles ont leur place.
# Quatre jeux de features, parce qu'un avertissement peut n'exister que dans
# l'un d'eux — le code derrière un `cfg` n'est compilé que s'il est demandé.
step "cargo clippy (défaut) -D warnings"
cargo clippy --all-targets -- -D warnings
step "cargo clippy --features viz -D warnings"
cargo clippy --all-targets --features viz -- -D warnings
step "cargo clippy --features extension-module,viz -D warnings"
cargo clippy --all-targets --features extension-module,viz -- -D warnings
step "cargo clippy --features extension-module,viz,viz-interactive,abi3 -D warnings"
cargo clippy --all-targets --features extension-module,viz,viz-interactive,abi3 -- -D warnings

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
