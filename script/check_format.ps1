# Formatage - Rust et Python. Ne formate rien : il verifie.
#
# A lancer APRES `cargo fmt` et `ruff format .`, jamais avant : le but est
# d'attraper ce qui part en commit mal formate.
#
# Lancer :   .\script\check_format.ps1

. "$PSScriptRoot\_common.ps1"

Step "cargo fmt --check"            { cargo fmt --check }
Step "ruff format --check (Python)" { ruff format --check . }

Write-Host "OK : formatage."
