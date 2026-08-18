# Exemples et scripts de formation, de bout en bout.
#
# pytest couvre l'API unite par unite ; ceux-ci couvrent autre chose : des
# chaines de calcul completes, ecrites comme un utilisateur les ecrirait.
# C'est ce qui manquait quand `add_submesh` a disparu sans que rien ne
# l'attrape - la suite etait verte, trois exemples etaient morts.
#
# Lancer :   .\script\check_examples.ps1

. "$PSScriptRoot\_common.ps1"

Require-Module-Pyrucast
Step "exemples + formation (bout en bout)" { & "$PSScriptRoot\run_examples.ps1" }

Write-Host "OK : exemples."
