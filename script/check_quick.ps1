# Le tour rapide - celui de la boucle de commit.
#
# Formatage et coeur Rust : ~3 min, contre ~5 min 40 pour check_all. C'est ce
# qui attrape la quasi-totalite des regressions, parce que c'est la que vit la
# quasi-totalite du code.
#
# Ce qu'il ne couvre PAS, et qu'il faut lancer avant de pousser :
#   - la liaison Python et pytest       -> check_python
#   - les exemples de bout en bout      -> check_examples
#   - le book et ses cinq garde-fous    -> check_doc
#
# Lancer :   .\script\check_quick.ps1

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

foreach ($c in @("format", "rust")) {
    Write-Host "`n=== check_$c ===" -ForegroundColor Cyan
    & powershell -NoProfile -File "script\check_$c.ps1"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "`nOK : tour rapide. Avant de pousser : check_python, check_examples, check_doc."
