# Execute tous les exemples Python et les scripts de formation de bout en bout.
# Equivalent Windows de run_examples.sh.
#
# Aucun n'ouvre de fenetre : la visualisation passe par plot(save=...), dirigee
# vers un repertoire temporaire.

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("pyrucast-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $tmp | Out-Null
$env:PYRUCAST_FORMATION_IMG_DIR = $tmp
$env:PYRUCAST_IMG_DIR = $tmp

$fail = 0
try {
    foreach ($f in (Get-ChildItem examples\*.py, formation\*.py | Sort-Object FullName)) {
        $log = Join-Path $tmp 'out.log'
        python $f.FullName *> $log
        if ($LASTEXITCODE -eq 0) {
            Write-Host ("  ok   " + $f.Name)
        } else {
            Write-Host ("  FAIL " + $f.Name)
            Get-Content $log -Tail 15 | ForEach-Object { Write-Host ("       " + $_) }
            $fail = 1
        }
    }
} finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}

exit $fail
