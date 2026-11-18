# Coeur Rust - tests unitaires, d'integration et doctests, toutes couches
# optionnelles compilees, plus un controle du build sans feature.
#
# Lancer :   .\script\check_rust.ps1

. "$PSScriptRoot\_common.ps1"

# Trois pas, un role chacun, et le choix du jeu de features n'est pas neutre.
#
# `cargo test` execute deja les doctests : les relancer par un `--doc` explicite
# etait une pure repetition.
#
# Les doctests tournent sous `viz`, PAS sous `viz-interactive`. Chacun est un
# binaire separe, lie contre tout le graphe de dependances : 120 crates sous
# `viz`, 178 sous `viz-interactive`. Sur les ~890 doctests, 237 s contre 364 s.
#
# La couche interactive est donc compilee sans etre testee : elle ne porte aucun
# test (960 lignes de fenetre winit, que rien ne peut exercer sans ecran), et un
# `build` suffit a garantir qu'elle compile encore.
#
# Le `clippy` par defaut, enfin : la bibliotheque se veut du Rust pur sans
# feature, et sans lui plus rien ne compilerait cette configuration-la. C'est un
# `clippy` et non un `check` : il compile la meme chose, pour 1 s de plus
# mesuree, et refuse en prime ce que les lints du workspace refusent - un
# `Result` qui ne peut pas echouer, par exemple (`Cargo.toml`, `clippy.toml`).
# Les trois autres jeux de features ne passent sous clippy qu'a la release
# (`check_clippy.ps1`).
#
# `--workspace` : depuis que `macros/` existe, le depot tient deux paquets, et
# cargo ne construit que la racine par defaut. Sans ce drapeau la crate de
# macros passerait sous tous les garde-fous.
Step "cargo clippy (features par defaut) -D warnings" `
    { cargo clippy --workspace --all-targets -- -D warnings }
Step "cargo test --features viz"               { cargo test --workspace --features viz }
Step "cargo build --features viz-interactive"  { cargo build --features viz-interactive }

Write-Host "OK : Rust."
