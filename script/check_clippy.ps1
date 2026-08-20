# Clippy - le meme code relu sous quatre jeux de features.
#
# Lancer :   .\script\check_clippy.ps1

. "$PSScriptRoot\_common.ps1"

# Quatre passes, parce qu'un avertissement peut n'exister que dans l'un des
# jeux : le code derriere un `cfg` n'est compile que s'il est demande. La 0.3.1
# l'a montre - `cargo clippy --fix` lance sur le jeu par defaut avait laisse
# intacts les `if` imbriques de `src/viz/`, que la quatrieme passe a rattrapes.
#
# Ce bloc n'appartient pas a `check_all` : il coute trop cher pour la boucle
# quotidienne. Il est appele aux deux moments ou l'on pose une version -
# `set_new_version.sh` en local, et le job `verify` de `release.yml` en CI.
Step "cargo clippy (defaut) -D warnings" `
    { cargo clippy --all-targets -- -D warnings }
Step "cargo clippy --features viz -D warnings" `
    { cargo clippy --all-targets --features viz -- -D warnings }
Step "cargo clippy --features extension-module,viz -D warnings" `
    { cargo clippy --all-targets --features extension-module,viz -- -D warnings }
Step "cargo clippy --features extension-module,viz,viz-interactive,abi3 -D warnings" `
    { cargo clippy --all-targets --features extension-module,viz,viz-interactive,abi3 -- -D warnings }

Write-Host "OK : clippy."
