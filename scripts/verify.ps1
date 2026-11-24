# Verification protocol V (script 1.5), fresh-process checks per milestone.
#   scripts\verify.ps1 -Milestone F0
#   scripts\verify.ps1 -Milestone F2
# F0: check.ps1, secret scan, spotcheck.py. F2: lint, format, tests, frozen
# gates and input hashes. Exit non-zero on any failure.
param(
    [Parameter(Mandatory = $true)][string]$Milestone
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PY = "B:\Applications\MiniConda\envs\docmind-ft\python.exe"
Set-Location $RepoRoot

function Invoke-Step([string]$Name, [scriptblock]$Cmd) {
    Write-Output "--- $Name ---"
    & $Cmd
    if ($LASTEXITCODE -ne 0) { throw "verify FAILED at step: $Name (exit $LASTEXITCODE)" }
}

if ($Milestone -eq "F0") {
    Invoke-Step "check.ps1" { powershell -NoProfile -File scripts\check.ps1 }

    Write-Output "--- secret scan ---"
    # Bare prefixes are written with a bracketed last char, so the pattern
    # still matches the real prefix but the scanner does not flag its own
    # pattern line (or this comment) for containing one.
    $patterns = "hf_[A-Za-z0-9]{20,}|gs[k]_|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|tvl[y]-|AIzaS[y]|AQ\.Ab"
    $tracked = git ls-files
    $hits = @()
    foreach ($f in $tracked) {
        $hits += Select-String -Path $f -Pattern $patterns -ErrorAction SilentlyContinue
    }
    if (Test-Path -LiteralPath "reports") {
        $hits += Get-ChildItem -Recurse -File "reports" | Select-String -Pattern $patterns -ErrorAction SilentlyContinue
    }
    if ($hits.Count -gt 0) {
        $hits | ForEach-Object { Write-Output "$($_.Path):$($_.LineNumber)" }
        throw "secret scan FAILED ($($hits.Count) hit(s))"
    }
    Write-Output "secret scan: 0 hits"

    Invoke-Step "spotcheck" { & $PY scripts\spotcheck.py --milestone F0 }

    Write-Output "verify.ps1 F0 PASSED"
    exit 0
}

if ($Milestone -eq "F2") {
    Invoke-Step "ruff check" { & $PY -m ruff check . }
    Invoke-Step "ruff format" { & $PY -m ruff format --check . }
    Invoke-Step "pytest" { & $PY -m pytest -q --basetemp=B:\p8\pytest-tmp-verify }
    Invoke-Step "gates schema" { & $PY scripts\check_gates.py --schema-only }
    Invoke-Step "gates freeze + context/split hashes" { & $PY scripts\check_gates.py --verify-freeze }

    Write-Output "verify.ps1 F2 PASSED"
    exit 0
}

throw "verify.ps1: milestone $Milestone not implemented yet"
