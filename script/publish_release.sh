#!/usr/bin/env bash
# pyrucast — publie le dernier tag git (vX.Y.Z) sur crates.io et PyPI.
#
# 1. repère le dernier tag vX.Y.Z, checkout dessus (détaché) ;
# 2. cargo publish sur crates.io (sauté si cette version y est déjà) ;
# 3. build sdist + wheel Linux (features extension-module,viz,viz-interactive,
#    abi3 — un seul wheel cp39-abi3 valable pour tout Python >= 3.9, plutôt
#    qu'un wheel lié à une version cp3XX précise) et upload sur PyPI
#    (fichiers déjà présents sautés individuellement) ;
# 4. revient sur la branche de départ.
#
# Tag manylinux : plotters (feature "ttf", pour le texte) tire font-kit/
# fontconfig et libpng, liés dynamiquement contre les versions système de
# ce poste. maturin (via patchelf) embarque ces .so dans le wheel
# (pyrucast.libs/) — pas de dépendance runtime sur le système cible —
# mais leur *version de compilation* fixe quand même le plancher glibc
# du wheel. Sans Docker (pour builder dans le conteneur manylinux2014
# officiel, aux libs plus anciennes), le tag reste manylinux_2_38 :
# portable sur les distros Linux récentes (glibc >= 2.38, ~2023+), pas
# sur les plus anciennes (Ubuntu 22.04, Debian 12…).
#
# Ne construit PAS le wheel Windows : voir script/publish_release.ps1,
# à lancer depuis une machine Windows pour ce fichier-là.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

bold=$(tput bold 2>/dev/null || true)
reset=$(tput sgr0 2>/dev/null || true)
step() { printf '\n%s>>> %s%s\n' "$bold" "$1" "$reset"; }
die()  { printf '\nERREUR: %s\n' "$1" >&2; exit 1; }

CRATE_NAME="$(sed -nE 's/^name = "(.*)"/\1/p' Cargo.toml | head -1)"
UA="User-Agent: ${CRATE_NAME}-publish-script (script/publish_release.sh)"

# ── 0. Préconditions git ─────────────────────────────────────────────────────
[ -z "$(git status --porcelain)" ] || die "arbre de travail non propre — commit/stash d'abord :
$(git status --porcelain)"

tag="$(git tag -l 'v*' --sort=-v:refname | head -1)"
[ -n "$tag" ] || die "aucun tag vX.Y.Z trouvé — lance d'abord script/set_new_version.sh"
version="${tag#v}"
echo "Dernier tag : $tag (version $version)"

orig_branch="$(git rev-parse --abbrev-ref HEAD)"
restore_branch() { git checkout --quiet "$orig_branch" 2>/dev/null || true; }
trap restore_branch EXIT

step "git checkout $tag"
git checkout --quiet "$tag"

file_version="$(sed -nE 's/^version = "(.*)"/\1/p' Cargo.toml | head -1)"
[ "$file_version" = "$version" ] \
    || die "Cargo.toml annonce la version $file_version au tag $tag (attendu $version) — incohérence"

# shellcheck source=/dev/null
. ./.venv/bin/activate

# ── 1. crates.io ─────────────────────────────────────────────────────────────
step "Vérification crates.io ($CRATE_NAME $version)"
crates_code="$(curl -s -o /dev/null -w '%{http_code}' -H "$UA" "https://crates.io/api/v1/crates/$CRATE_NAME/$version")"
if [ "$crates_code" = "200" ]; then
    echo "  déjà publié — sauté."
else
    step "cargo publish"
    cargo publish
fi

# ── 2. PyPI : sdist + wheel Linux ────────────────────────────────────────────
step "Vérification des fichiers déjà présents sur PyPI"
existing_files="$(curl -s "https://pypi.org/pypi/$CRATE_NAME/$version/json" \
    | python3 -c "import json,sys
try:
    d = json.load(sys.stdin)
    print('\n'.join(u['filename'] for u in d.get('urls', [])))
except Exception:
    pass" 2>/dev/null || true)"

python -m pip install --quiet --upgrade ziglang patchelf

step "maturin sdist"
maturin sdist
step "maturin build --release --zig --manylinux 2_38 --features extension-module,viz,viz-interactive,abi3"
maturin build --release --zig --manylinux 2_38 --features extension-module,viz,viz-interactive,abi3

to_upload=()
while IFS= read -r -d '' f; do
    name="$(basename "$f")"
    if grep -qxF "$name" <<< "$existing_files"; then
        echo "  $name déjà sur PyPI — sauté."
    else
        to_upload+=("$f")
    fi
done < <(find target/wheels -maxdepth 1 -name "${CRATE_NAME}-${version}*" -print0)

if [ "${#to_upload[@]}" -eq 0 ]; then
    echo "Rien à uploader — tout est déjà sur PyPI."
else
    step "maturin upload (${#to_upload[@]} fichier(s))"
    maturin upload "${to_upload[@]}"
fi

echo
echo "OK: $tag publié (crates.io + PyPI, wheel Linux). Wheel Windows : script/publish_release.ps1 sur une machine Windows."
