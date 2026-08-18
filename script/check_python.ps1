# Surface Python - recompile l'extension puis deroule pytest.
#
# C'est ce script qui (re)installe le module dans le venv : les autres
# verifications qui ont besoin de pyrucast supposent qu'il est deja passe.
#
# Lancer :   .\script\check_python.ps1

. "$PSScriptRoot\_common.ps1"

Step "maturin develop --features extension-module,viz" {
    maturin develop --features extension-module,viz
}
Step "pytest" { python -m pytest }

Write-Host "OK : Python."
