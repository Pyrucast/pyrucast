#!/usr/bin/env bash
# Surface Python — recompile l'extension puis déroule pytest.
#
# C'est ce script qui (ré)installe le module dans le venv : les autres
# vérifications qui ont besoin de pyrucast supposent qu'il est déjà passé.

. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

step "maturin develop --features extension-module,viz" \
    maturin develop --features extension-module,viz
step "pytest"                                          python -m pytest

echo "OK : Python."
