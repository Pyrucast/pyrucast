#!/usr/bin/env bash
# Le tour rapide — celui de la boucle de commit.
#
# Formatage et cœur Rust : ~3 min, contre ~5 min 40 pour `check_all`. C'est ce
# qui attrape la quasi-totalité des régressions, parce que c'est là que vit la
# quasi-totalité du code.
#
# Ce qu'il ne couvre PAS, et qu'il faut lancer avant de pousser :
#   - la liaison Python et pytest       → check_python
#   - les exemples de bout en bout      → check_examples
#   - le book et ses cinq garde-fous    → check_doc
# ou, plus simplement, `check_all` qui enchaîne les cinq.
#
# Si l'on n'a touché qu'un domaine, plus étroit encore : le bloc seul.

set -euo pipefail
cd "$(dirname "$0")/.."

for c in format rust; do
    printf '\n═══ check_%s ═══\n' "$c"
    bash "script/check_$c.sh"
done

echo
echo "OK : tour rapide. Avant de pousser : check_python, check_examples, check_doc."
