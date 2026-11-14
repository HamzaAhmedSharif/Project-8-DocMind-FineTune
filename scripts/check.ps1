# Pre-commit check (script 1.11): ruff, pytest, vendor check, gates schema.
# Laptop-only; CI runs its own steps. Run from the repo root or anywhere.
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PY = "B:\Applications\MiniConda\envs\docmind-ft\python.exe"
Set-Location $RepoRoot

function Invoke-Step([string]$Name, [scriptblock]$Cmd) {
    Write-Output "--- $Name ---"
    & $Cmd
    if ($LASTEXITCODE -ne 0) { throw "check FAILED at step: $Name (exit $LASTEXITCODE)" }
}

Invoke-Step "ruff check" { & $PY -m ruff check . }
Invoke-Step "ruff format" { & $PY -m ruff format --check . }
Invoke-Step "pytest" { & $PY -m pytest -q }
Invoke-Step "vendor check" { & $PY scripts\check_vendor.py }
Invoke-Step "gates schema" { & $PY scripts\check_gates.py --schema-only }
Write-Output "check.ps1 PASSED"
