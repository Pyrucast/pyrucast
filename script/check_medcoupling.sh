#!/usr/bin/env bash
# Interface medcoupling — les tests qui exigent le module medcoupling installé.
#
# Ce bloc n'appartient pas à `check_all` : un développeur qui ne touche pas à
# cette interface n'a pas à installer medcoupling. Il est appelé quand on y touche,
# et par le job `verify` de `release.yml`.
#
# Il ne recompile pas l'extension : `check_python.sh` s'en charge.

. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

require_module

# Sans cette garde, `pytest -m medcoupling` ne trouverait que des tests
# *skippés* et rendrait 0 : la vérification passerait au vert sans rien avoir
# vérifié.
python -c "import medcoupling" 2>/dev/null || {
    echo "ERREUR : le module medcoupling manque — 'pip install medcoupling' dans le venv." >&2
    echo "         Roues publiées pour Linux x86_64 et Windows seulement." >&2
    exit 2
}

step "pytest -m medcoupling" python -m pytest -m medcoupling

echo "OK : medcoupling."
