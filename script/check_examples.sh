#!/usr/bin/env bash
# Exemples et scripts de formation, de bout en bout.
#
# `pytest` couvre l'API unité par unité ; ceux-ci couvrent autre chose : des
# chaînes de calcul complètes, écrites comme un utilisateur les écrirait.
# C'est ce qui manquait quand `add_submesh` a disparu sans que rien ne
# l'attrape — la suite était verte, trois exemples étaient morts.

. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

require_module
step "exemples + formation (bout en bout)" ./script/run_examples.sh

echo "OK : exemples."
