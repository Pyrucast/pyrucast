#!/usr/bin/env bash
# Interface gmsh — les tests qui exigent le module gmsh installé.
#
# Ce bloc n'appartient pas à `check_all` : un développeur qui ne touche pas à
# cette interface n'a pas à installer gmsh. Il est appelé quand on y touche,
# et par le job `verify` de `release.yml`.
#
# Il ne recompile pas l'extension : `check_python.sh` s'en charge.

. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

require_module

# Sans cette garde, `pytest -m gmsh` ne trouverait que des tests *skippés* et
# rendrait 0 : la vérification passerait au vert sans rien avoir vérifié.
python -c "import gmsh" 2>/dev/null || {
    echo "ERREUR : le module gmsh manque — 'pip install gmsh' dans le venv." >&2
    echo "         Linux : la roue gmsh embarque un libgmsh lié à OpenGL, donc" >&2
    echo "         les paquets système 'libglu1-mesa' et 'libopengl0' aussi." >&2
    exit 2
}

step "pytest -m gmsh" python -m pytest -m gmsh

echo "OK : gmsh."
