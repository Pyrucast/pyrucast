# Interface medcoupling - les tests qui exigent le module medcoupling installe.
#
# Lancer :   .\script\check_medcoupling.ps1
#
# Ce bloc n'appartient pas a `check_all` : un developpeur qui ne touche pas a
# cette interface n'a pas a installer medcoupling. Il est appele quand on y touche,
# et par le job `verify` de `release.yml`.
#
# Il ne recompile pas l'extension : `check_python.ps1` s'en charge.

. "$PSScriptRoot\_common.ps1"

Require-Module-Pyrucast

# Sans cette garde, `pytest -m medcoupling` ne trouverait que des tests *skippes* et
# rendrait 0 : la verification passerait au vert sans rien avoir verifie.
python -c "import medcoupling"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Le module medcoupling manque - 'pip install medcoupling' dans le venv."
    exit 2
}

Step "pytest -m medcoupling" { python -m pytest -m medcoupling }

Write-Host "OK : medcoupling."
