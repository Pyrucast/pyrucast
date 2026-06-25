#!/usr/bin/env bash
# pyrucast — build the book + rustdoc and publish them to Codeberg Pages
# by hand.
#
# Builds the mdBook (book/) and the Rust API reference (rustdoc), then
# force-pushes the combined site to the `pages` branch, which Codeberg
# serves at:
#     https://gauthier.codeberg.page/pyrucast/          (book, racine)
#     https://gauthier.codeberg.page/pyrucast/rust/...  (rustdoc)
#
# This is the manual alternative to the Forgejo Actions workflow
# (.forgejo/workflows/pages.yml), useful until a Forgejo runner is
# available. It pushes with your own git credentials (the `origin`
# remote) — no CI token required.
#
# The build output is staged in a throwaway temporary repository, so
# neither the working tree nor book/book/ is touched by git.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

bold=$(tput bold 2>/dev/null || true)
reset=$(tput sgr0 2>/dev/null || true)
step() { printf '\n%s>>> %s%s\n' "$bold" "$1" "$reset"; }
die()  { printf '\nERROR: %s\n' "$1" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

have mdbook || die "mdbook not found — install it via 'cargo install mdbook'"
have cargo  || die "cargo not found — install Rust via rustup"
have git    || die "git not found"

REMOTE_URL="$(git remote get-url --push origin)" \
    || die "no 'origin' remote configured"

step "Building the book"
mdbook build book

step "Building the Rust API reference (rustdoc)"
# Pure Rust (default features): neither pyo3 nor libpython is required.
cargo doc --no-deps --lib

step "Publishing the combined site (book + rust/) to the 'pages' branch"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cp -a book/book/. "$TMP"/
mkdir "$TMP"/rust
cp -a target/doc/. "$TMP"/rust/
# cargo doc ne génère pas d'index.html à la racine de target/doc : on dépose
# une redirection pour que /rust/ mène directement à la crate.
printf '<!doctype html><meta http-equiv="refresh" content="0; url=pyrucast/index.html">\n' \
    > "$TMP"/rust/index.html
(
    cd "$TMP"
    git init -q
    git checkout -q -b pages
    git config user.name  "$(git -C "$ROOT" config user.name  || echo pyrucast)"
    git config user.email "$(git -C "$ROOT" config user.email || echo ci@pyrucast)"
    git add -A
    git commit -q -m "Déploiement du book + rustdoc"
    git push -f "$REMOTE_URL" pages
)

step "Done"
echo "  Book   : https://gauthier.codeberg.page/pyrucast/"
echo "  Rustdoc: https://gauthier.codeberg.page/pyrucast/rust/"
