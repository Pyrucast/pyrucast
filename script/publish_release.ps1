# pyrucast - construit le wheel Windows du dernier tag git (vX.Y.Z) et
# l'envoie sur PyPI. Ne touche PAS crates.io (deja fait cote Linux, la
# crate n'est pas liee a une plateforme) ni le sdist (deja envoye cote
# Linux par script/publish_release.sh).
#
# Run from a PowerShell prompt:   .\script\publish_release.ps1
# (If scripts are blocked: powershell -ExecutionPolicy Bypass -File .\script\publish_release.ps1)

$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Step($msg) { Write-Host "`n>>> $msg" -ForegroundColor Cyan }
function Die($msg)  { Write-Host "`nERROR: $msg" -ForegroundColor Red; exit 1 }
function Run($label, [scriptblock]$cmd) {
    Step $label
    & $cmd
    if ($LASTEXITCODE -ne 0) { Die "$label failed (exit $LASTEXITCODE)" }
}

function Get-TomlValue($path, $key) {
    $m = Select-String -Path $path -Pattern "^$key = `"(.*)`"" | Select-Object -First 1
    if (-not $m) { Die "cle '$key' introuvable dans $path" }
    return $m.Matches[0].Groups[1].Value
}

function Get-PyPiFiles($crate, $version) {
    try {
        $resp = Invoke-RestMethod -Uri "https://pypi.org/pypi/$crate/$version/json"
        return @($resp.urls | ForEach-Object { $_.filename })
    } catch {
        return @()
    }
}

# -- 0. Preconditions git -----------------------------------------------------
$dirty = git status --porcelain
if ($dirty) { Die "arbre de travail non propre - commit/stash d'abord :`n$dirty" }

$tag = git tag -l 'v*' --sort=-v:refname | Select-Object -First 1
if (-not $tag) { Die "aucun tag vX.Y.Z trouve - il doit deja exister (cree via script/set_new_version.sh)" }
$version = $tag.Substring(1)
Write-Host "Dernier tag : $tag (version $version)"

$origBranch = git rev-parse --abbrev-ref HEAD

Step "git checkout $tag"
git checkout --quiet $tag
if ($LASTEXITCODE -ne 0) { Die "checkout de $tag impossible" }

try {
    $crateName = Get-TomlValue 'Cargo.toml' 'name'
    $fileVersion = Get-TomlValue 'Cargo.toml' 'version'
    if ($fileVersion -ne $version) {
        Die "Cargo.toml annonce la version $fileVersion au tag $tag (attendu $version) - incoherence"
    }

    if (-not (Test-Path .venv)) { Die "venv absent - lance script/build.ps1 au moins une fois avant" }
    & .\.venv\Scripts\Activate.ps1

    Step "Fichiers deja presents sur PyPI pour $crateName $version"
    $existing = Get-PyPiFiles $crateName $version
    $existing | ForEach-Object { Write-Host "  deja present : $_" }

    Run "maturin build --release --features extension-module,viz,viz-interactive" {
        maturin build --release --features extension-module,viz,viz-interactive
    }

    $wheel = Get-ChildItem "target\wheels\$crateName-$version-*.whl" |
        Where-Object { $_.Name -match 'win' } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $wheel) { Die "aucun wheel Windows trouve dans target\wheels apres le build" }
    Write-Host "  wheel construit : $($wheel.Name)"

    if ($existing -contains $wheel.Name) {
        Write-Host "$($wheel.Name) deja sur PyPI - rien a envoyer."
    } else {
        Run "maturin upload $($wheel.Name)" { maturin upload $wheel.FullName }
    }

    Write-Host "`nOK: wheel Windows de $tag envoye sur PyPI."
}
finally {
    Step "retour sur $origBranch"
    git checkout --quiet $origBranch
}
