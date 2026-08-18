# Socle commun des check_*.ps1 : racine du depot, venv, affichage des etapes.
# Ce fichier se *source* (dot-source), il ne s'execute pas :
#     . "$PSScriptRoot\_common.ps1"

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Test-Path .venv\Scripts\Activate.ps1)) {
    Write-Error "Pas de venv - lancer d'abord .\script\dev.ps1"
}
& .\.venv\Scripts\Activate.ps1

# Une etape : son nom, puis la commande. PowerShell ne propage pas le code de
# retour des programmes externes, il faut tester $LASTEXITCODE a la main.
function Step {
    param([string]$Name, [scriptblock]$Body)
    Write-Host ">>> $Name"
    & $Body
    if ($LASTEXITCODE -ne 0) { Write-Error "echec : $Name (code $LASTEXITCODE)" }
}

# Le module compile est-il importable ? Les verifications qui *utilisent*
# pyrucast sans le reconstruire s'en servent pour donner un message clair.
function Require-Module-Pyrucast {
    python -c 'import pyrucast' 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "pyrucast n'est pas importable - lancer .\script\check_python.ps1 (ou .\script\dev.ps1)."
    }
}
