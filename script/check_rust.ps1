# Coeur Rust - tests unitaires, d'integration, doctests, et la compilation
# des couches optionnelles (visualisation) qu'aucun test ne couvre.
#
# Lancer :   .\script\check_rust.ps1

. "$PSScriptRoot\_common.ps1"

Step "cargo test (features par defaut)"       { cargo test }
Step "cargo test --doc (explicite)"           { cargo test --doc }
Step "cargo test --features viz"              { cargo test --features viz }
Step "cargo build --features viz-interactive" { cargo build --features viz-interactive }

Write-Host "OK : Rust."
