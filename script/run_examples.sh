#!/usr/bin/env bash
# Exécute tous les exemples Python et les scripts de formation de bout en bout.
#
# `pytest` couvre l'API unité par unité ; ces scripts couvrent autre chose :
# des chaînes de calcul complètes, écrites comme un utilisateur les écrirait.
# C'est ce qui manquait quand `add_submesh` a disparu sans que rien ne
# l'attrape — la suite était verte, trois exemples étaient morts.
#
# Aucun n'ouvre de fenêtre : la visualisation passe par `plot(save=…)`, dirigée
# vers un répertoire temporaire.

set -euo pipefail

cd "$(dirname "$0")/.."

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
export PYRUCAST_FORMATION_IMG_DIR="$tmp"
export PYRUCAST_IMG_DIR="$tmp"

fail=0
for f in examples/*.py formation/*.py; do
    if python "$f" >"$tmp/out.log" 2>&1; then
        printf '  ok   %s\n' "$f"
    else
        printf '  FAIL %s\n' "$f"
        tail -n 15 "$tmp/out.log" | sed 's/^/       /'
        fail=1
    fi
done

exit "$fail"
