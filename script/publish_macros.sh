#!/usr/bin/env bash
# pyrucast — vérifie puis publie `pyrucast-macros` sur crates.io.
#
# Geste **manuel**, et rare : l'outillage de macros bouge bien plus lentement
# que la bibliothèque, et `release.yml` ne publie que le paquet racine. Tant
# que la version déclarée dans `Cargo.toml` n'existe pas sur crates.io,
# `cargo package` / `cargo publish` de `pyrucast` échouent — la dépendance est
# en `path` **et** en version, donc cargo la cherche dans l'index.
#
#   bash script/publish_macros.sh            # publie la version courante
#   bash script/publish_macros.sh 0.1.1      # la passe à 0.1.1, puis publie
#
# Avec un numéro, le script met à jour `macros/Cargo.toml`, celui de la
# dépendance dans `Cargo.toml`, et commit — les deux ne peuvent pas diverger,
# sans quoi la publication ne débloquerait rien.
#
# Ne pousse rien, et ne touche pas au tag de `pyrucast` : après publication,
# `script/set_new_version.sh` reprend la main.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CRATE=pyrucast-macros
MANIFEST=macros/Cargo.toml

bold=$(tput bold 2>/dev/null || true)
reset=$(tput sgr0 2>/dev/null || true)
step() { printf '\n%s>>> %s%s\n' "$bold" "$1" "$reset"; }
die()  { printf '\nERREUR: %s\n' "$1" >&2; exit 1; }

version_of() { sed -nE 's/^version = "(.*)"/\1/p' "$1" | head -1; }

# La version que `pyrucast` exige de sa dépendance. Publier un autre numéro ne
# débloquerait pas son `cargo package` : c'est celui-ci que cargo cherchera.
required_version() {
    sed -nE "s/^$CRATE = \{.*version = \"([^\"]+)\".*/\1/p" Cargo.toml | head -1
}

# 200 = déjà publiée, 404 = numéro libre. Tout le reste — proxy, panne,
# coupure — est une **réponse inconnue**, pas une autorisation : la lire comme
# « libre » ferait publier un numéro déjà pris, ou taguer sur une publication
# jamais partie. Un numéro déjà pris ne se reprend pas, même à contenu
# identique.
http_status() {
    curl -sS -o /dev/null -w '%{http_code}' \
        "https://crates.io/api/v1/crates/$CRATE/$1" 2>/dev/null || echo 000
}
published() {
    local code
    code="$(http_status "$1")"
    case "$code" in
        200) return 0 ;;
        404) return 1 ;;
        *) die "crates.io répond $code pour $CRATE $1 — impossible de savoir si \
le numéro est libre" ;;
    esac
}

# ── 0. Préconditions ─────────────────────────────────────────────────────────
branch="$(git rev-parse --abbrev-ref HEAD)"
[ "$branch" = master ] || die "il faut être sur master (actuellement: $branch)"
[ -z "$(git status --porcelain)" ] || die "arbre de travail non propre — commit/stash d'abord :
$(git status --porcelain)"

[ -n "${CARGO_REGISTRY_TOKEN:-}" ] || [ -f "${CARGO_HOME:-$HOME/.cargo}/credentials.toml" ] \
    || die "aucun jeton crates.io — 'cargo login', ou CARGO_REGISTRY_TOKEN dans l'environnement"

current="$(version_of "$MANIFEST")"
new_version="${1:-$current}"
[[ "$new_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.]+)?$ ]] \
    || die "format de version invalide (attendu: X.Y.Z) : $new_version"

step "crates.io connaît-il déjà $CRATE $new_version ?"
if published "$new_version"; then
    die "$CRATE $new_version est déjà publiée — donner un numéro plus haut :
    bash script/publish_macros.sh <X.Y.Z>"
fi
echo "non — le numéro est libre."

# ── 1. Le numéro, des deux côtés ─────────────────────────────────────────────
if [ "$new_version" != "$current" ]; then
    step "Mise à jour de $MANIFEST et de la dépendance ($current → $new_version)"
    sed -i -E "0,/^version = \".*\"/s//version = \"$new_version\"/" "$MANIFEST"
    sed -i -E "s/^($CRATE = \{.*version = )\"[^\"]+\"/\1\"$new_version\"/" Cargo.toml
    # Un sed qui ne mord pas ne dit rien : sans relecture, l'écart ne se
    # verrait qu'à la publication, numéro déjà brûlé.
    [ "$(version_of "$MANIFEST")" = "$new_version" ] \
        || die "$MANIFEST annonce $(version_of "$MANIFEST") après mise à jour"
fi

required="$(required_version)"
[ "$required" = "$new_version" ] || die "Cargo.toml exige $CRATE $required, et l'on publierait \
$new_version — publier l'un ne débloquerait pas l'autre"

# ── 2. Vérifications ─────────────────────────────────────────────────────────
# La crate seule d'abord, puis son consommateur : une macro procédurale ne
# prouve rien tant qu'un appelant ne l'a pas expansée, et `pyrucast` ne la
# compile que sous `python-api`.
step "cargo fmt --check"
cargo fmt -p "$CRATE" -- --check

step "cargo clippy -D warnings"
cargo clippy -p "$CRATE" --all-targets -- -D warnings

step "cargo test"
cargo test -p "$CRATE"

# `-D warnings` : `check_doc.sh` ne documente que la lib racine, donc rien
# ailleurs ne relit la rustdoc de cette crate-ci.
step "cargo doc --no-deps -D warnings"
RUSTDOCFLAGS="-D warnings" cargo doc -p "$CRATE" --no-deps

step "cargo check --features python-api (le consommateur expanse la macro)"
cargo check --features python-api --all-targets

# `cargo package` déballe le .crate et le compile : la seule passe qui voie ce
# que l'empaquetage a emporté, et ce qu'il a laissé.
step "cargo package (le paquet déballé compile)"
cargo package -p "$CRATE"

step "cargo publish --dry-run"
cargo publish -p "$CRATE" --dry-run

# ── 3. Publication ───────────────────────────────────────────────────────────
echo
echo "OK : $CRATE $new_version est vert, sans warning, paquet compris."
echo "crates.io est une porte à sens unique : un numéro publié ne se reprend pas."
read -rp "Publier $CRATE $new_version sur crates.io ? [o/N] " confirm
[[ "$confirm" =~ ^[oO]$ ]] || die "annulé"

if [ "$new_version" != "$current" ]; then
    step "git commit (version $new_version)"
    git add "$MANIFEST" Cargo.toml Cargo.lock
    git commit -m "chore(macros): version $new_version"
fi

step "cargo publish"
cargo publish -p "$CRATE"

# ── 4. Attendre l'index ──────────────────────────────────────────────────────
# `cargo publish` rend la main avant que l'index ne serve la version. Tant
# qu'il ne la sert pas, `cargo package` de `pyrucast` échoue exactement comme
# avant — l'attendre ici évite de croire la publication ratée.
step "Attente de l'index crates.io"
served=false
for _ in $(seq 1 60); do
    if [ "$(http_status "$new_version")" = 200 ]; then
        served=true
        echo "crates.io sert $CRATE $new_version."
        break
    fi
    sleep 5
done
[ "$served" = true ] || die "crates.io ne sert toujours pas $new_version après 5 minutes \
— vérifier https://crates.io/crates/$CRATE avant de taguer pyrucast"

echo
echo "Fait : $CRATE $new_version publiée."
if [ "$new_version" != "$current" ]; then
    echo "Commit local créé, non poussé : git push"
fi
echo "Suite : bash script/set_new_version.sh"
