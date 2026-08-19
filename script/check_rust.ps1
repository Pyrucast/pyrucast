# Coeur Rust - tests unitaires, d'integration et doctests, toutes couches
# optionnelles compilees, plus un controle du build sans feature.
#
# Lancer :   .\script\check_rust.ps1

. "$PSScriptRoot\_common.ps1"

# `viz-interactive` implique `viz`, et `cargo test` execute deja les doctests :
# un seul jeu de features couvre ce que quatre pas couvraient, sans les
# recompiler trois fois ni relancer les ~890 doctests a chaque fois.
#
# Le `check` par defaut est le filet qui reste : la bibliotheque se veut du
# Rust pur sans feature, et sans lui plus rien ne compilerait cette
# configuration-la. Il coute 25 s, contre 200 s pour un `test` complet.
Step "cargo check (features par defaut)"      { cargo check --all-targets }
Step "cargo test --features viz-interactive"  { cargo test --features viz-interactive }

Write-Host "OK : Rust."
