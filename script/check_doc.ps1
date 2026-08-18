# Documentation - rustdoc sans warning, garde-fous du book, puis rendu.
#
# Il n'y a plus de pas « compiler les extraits » : depuis que tout bloc du book
# est un {{#include}}, le code affiche est celui d'un test ou d'un exemple, et
# c'est check_rust / check_python / check_examples qui l'executent. mdbook test
# a disparu pour la meme raison - il ne compilait rien, tous les blocs Rust
# etant rust,ignore par construction du mecanisme d'inclusion.
#
# Lancer :   .\script\check_doc.ps1

. "$PSScriptRoot\_common.ps1"

Step "cargo doc --no-deps --lib (sans warning)" {
    $env:RUSTDOCFLAGS = '-D warnings'
    cargo doc --no-deps --lib
    Remove-Item Env:\RUSTDOCFLAGS
}
# Quatre garde-fous de texte : includes qui resolvent, aucune page qui possede
# de code, prose sans symbole disparu, cliquet de couverture des doctests.
# Apres cargo doc, dont le dernier lit la sortie.
Step "garde-fous de la documentation" { python script\doc_lint.py }
Step "mdbook build"                     { mdbook build book }

Write-Host "OK : documentation."
