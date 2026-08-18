#!/usr/bin/env bash
# Toutes les vérifications, dans l'ordre. C'est le script à brancher en CI.
#
# L'ordre n'est pas indifférent : le formatage d'abord (il échoue en une
# seconde), le cœur Rust ensuite, puis Python — qui (ré)installe l'extension
# dont les exemples ont besoin —, les exemples, et la documentation en
# dernier, la plus lente.
#
# Chaque bloc se lance aussi seul :  bash script/check_rust.sh
#
# ⚠ Ne jamais piper ce script (`| tail`, `| grep`) : le code de retour
# deviendrait celui du dernier maillon du tube, et un échec passerait pour un
# succès. La sortie est longue, c'est le prix.

set -euo pipefail
cd "$(dirname "$0")/.."

CHECKS=(format rust python examples doc)

for c in "${CHECKS[@]}"; do
    printf '\n═══ check_%s ═══\n' "$c"
    bash "script/check_$c.sh"
done

echo
echo "OK : toutes les vérifications passent."
