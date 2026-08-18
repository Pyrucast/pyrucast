# Documentation - rustdoc sans warning, extraits Rust du book compiles,
# puis mdbook.
#
# Le pas qui n'existait pas : `mdbook test` ne compile RIEN, tous les blocs
# Rust du book etant `rust,ignore`. Un module renomme ou une signature changee
# pouvait donc y pourrir indefiniment. `cargo run --bin book_blocks` rassemble
# les extraits qui sont de vrais programmes dans un fichier de test jetable,
# que l'on type-verifie ici.
#
# Les extraits restent des extraits : beaucoup nomment une variable que la page
# ne definit jamais. On ignore donc les codes d'erreur qui signalent un
# fragment, et on echoue sur tout le reste - dont les erreurs de syntaxe, qui
# rendraient le garde-fou aveugle si on les laissait passer.
#
# Lancer :   .\script\check_doc.ps1

. "$PSScriptRoot\_common.ps1"

# Codes toleres (le bloc est un extrait, pas l'API qui a bouge) :
#   E0425 valeur/fonction inconnue      E0412 type inconnu
#   E0405 trait inconnu                 E0422 structure inconnue
#   E0433 module non resolu             E0252 import en double
#   E0277 `?` hors d'une fn -> Result   E0423 macro prise pour une fonction
$Fragment = 'error\[(E0425|E0412|E0405|E0422|E0433|E0252|E0277|E0423)\]'

function Check-BookBlocks {
    cargo run --quiet --bin book_blocks
    if ($LASTEXITCODE -ne 0) { Write-Error "book_blocks : generation en echec" }

    $out = cargo check --test book_blocks --features viz,book-check --message-format short 2>&1
    $fatal = $out |
        Select-String -Pattern '^tests[\\/]book_blocks\.rs.*error' |
        Where-Object { $_ -notmatch $Fragment }

    if ($fatal) {
        Write-Host "L'API a bouge sous les extraits du book :"
        $fatal | ForEach-Object { Write-Host $_ }
        Write-Host ""
        Write-Host "Chaque ligne pointe tests\book_blocks.rs ; le commentaire '// --- page:ligne'"
        Write-Host "qui precede donne la page du book et la ligne d'origine."
        Write-Error "extraits du book : l'API a bouge"
    }
    $global:LASTEXITCODE = 0
}

Step "cargo doc --no-deps --lib (sans warning)" {
    $env:RUSTDOCFLAGS = '-D warnings'
    cargo doc --no-deps --lib
    Remove-Item Env:\RUSTDOCFLAGS
}
Step "extraits Rust du book (compiles)" { Check-BookBlocks }
Step "mdbook build"                     { mdbook build book }
Step "mdbook test"                      { mdbook test book }

Write-Host "OK : documentation."
