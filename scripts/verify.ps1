# Verification protocol V (script 1.5), fresh-process checks per milestone.
#   scripts\verify.ps1 -Milestone F0
# Runs check.ps1, the secret scan, the hashes check, and spotcheck.py.
# Exit non-zero on any failure. Filled per milestone; F0 implemented now.
param(
    [Parameter(Mandatory = $true)][string]$Milestone
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PY = "B:\Applications\MiniConda\envs\docmind-ft\python.exe"
Set-Location $RepoRoot

if ($Milestone -ne "F0") {
    throw "verify.ps1: milestone $Milestone not implemented yet"
}

function Invoke-Step([string]$Name, [scriptblock]$Cmd) {
    Write-Output "--- $Name ---"
    & $Cmd
    if ($LASTEXITCODE -ne 0) { throw "verify FAILED at step: $Name (exit $LASTEXITCODE)" }
}

Invoke-Step "check.ps1" { powershell -NoProfile -File scripts\check.ps1 }

Write-Output "--- secret scan ---"
$patterns = "hf_[A-Za-z0-9]{20,}|gsk_|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|tvly-|AIzaSy|AQ\.Ab"
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
