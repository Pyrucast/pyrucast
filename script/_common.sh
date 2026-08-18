#!/usr/bin/env bash
# Socle commun des `check_*.sh` : racine du dépôt, venv, affichage des étapes.
# Ce fichier se *source*, il ne s'exécute pas.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[ -f .venv/bin/activate ] || {
    echo "ERREUR : pas de venv — lancer d'abord 'bash script/dev.sh'." >&2
    exit 2
}
# shellcheck source=/dev/null
. ./.venv/bin/activate

# Une étape : son nom, puis la commande. S'arrête à la première erreur
# (`set -e`), donc l'étape affichée en dernier est celle qui a échoué.
step() {
    echo ">>> $1"
    shift
    "$@"
}

# Le module compilé est-il importable ? Les vérifications qui *utilisent*
# pyrucast sans le reconstruire s'en servent pour donner un message clair.
require_module() {
    python -c 'import pyrucast' 2>/dev/null || {
        echo "ERREUR : pyrucast n'est pas importable — lancer 'bash script/check_python.sh'" >&2
        echo "         (ou 'bash script/dev.sh') pour compiler et installer le module." >&2
        exit 2
    }
}
