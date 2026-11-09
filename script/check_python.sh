#!/usr/bin/env bash
# Surface Python — recompile l'extension puis déroule pytest.
#
# C'est ce script qui (ré)installe le module dans le venv : les autres
# vérifications qui ont besoin de pyrucast supposent qu'il est déjà passé.
#
# En release, comme la roue que `pip install pyrucast` installe : les scripts
# du tutoriel et leurs garde-fous enchaînent des calculs pas à pas sur des
# HEX20, 13 s chacun en release contre plus de 7 min en debug. Les tests
# cargo, eux, restent en debug (`check_rust`).

. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

step "maturin develop --release --features extension-module,viz" \
    maturin develop --release --features extension-module,viz
step "pytest"                                          python -m pytest

echo "OK : Python."
