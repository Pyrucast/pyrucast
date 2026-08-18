# Toutes les verifications, dans l'ordre. Equivalent Windows de check_all.sh.
#
# L'ordre n'est pas indifferent : le formatage d'abord (il echoue en une
# seconde), le coeur Rust ensuite, puis Python - qui (re)installe l'extension
# dont les exemples ont besoin -, les exemples, et la documentation en dernier,
# la plus lente.
#
# Chaque bloc se lance aussi seul :   .\script\check_rust.ps1
#
# Lancer :   .\script\check_all.ps1
# (si les scripts sont bloques :
#  powershell -ExecutionPolicy Bypass -File .\script\check_all.ps1)

$ErrorActionPreference = 'Stop'

$checks = @('format', 'rust', 'python', 'examples', 'doc')

foreach ($c in $checks) {
    Write-Host ""
    Write-Host "=== check_$c ==="
    & (Join-Path $PSScriptRoot "check_$c.ps1")
}

Write-Host ""
Write-Host "OK : toutes les verifications passent."
