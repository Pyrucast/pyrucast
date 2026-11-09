# Surface Python - recompile l'extension puis deroule pytest.
#
# C'est ce script qui (re)installe le module dans le venv : les autres
# verifications qui ont besoin de pyrucast supposent qu'il est deja passe.
#
# En release, comme la roue que `pip install pyrucast` installe : les scripts
# du tutoriel et leurs garde-fous enchainent des calculs pas a pas sur des
# HEX20, 13 s chacun en release contre plus de 7 min en debug. Les tests
# cargo, eux, restent en debug (`check_rust`).
#
# Lancer :   .\script\check_python.ps1

. "$PSScriptRoot\_common.ps1"

Step "maturin develop --release --features extension-module,viz" {
    maturin develop --release --features extension-module,viz
}
Step "pytest" { python -m pytest }

Write-Host "OK : Python."
