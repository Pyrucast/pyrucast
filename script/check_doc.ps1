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
# Les figures des fiches d'elements ne sont pas commitees : le script, en
# Python pur, les dessine avant chaque rendu.
Step "figures des elements"             { python script\elements_figures.py }
Step "mdbook build"                     { mdbook build book }
# Neuf garde-fous de texte : includes qui resolvent, aucune page qui possede de
# code, prose sans symbole disparu, cliquet de couverture des doctests Rust,
# cliquet de citation de l'API Python, equations intactes, reference des
# operateurs complete, stub fidele au module, cliquet d'ecriture des
# docstrings. Apres cargo doc et mdbook build, dont le quatrieme et le sixieme
# lisent la sortie ; le cinquieme et le huitieme importent le module Python
# installe.
Step "garde-fous de la documentation" { python script\doc_lint.py }

Write-Host "OK : documentation."
