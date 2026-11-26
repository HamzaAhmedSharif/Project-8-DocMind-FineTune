# F4 canary for the 2560 cutoff: 6 micro-steps at GA 1 on the 10 longest
# rows of the 2560-filtered F4 set (2,553-2,560 tokens). If these fit, the
# whole 2560 set should fit for training; if they OOM cleanly, the cutoff
# must go lower.
# Launched by scripts\run_detached.ps1; writes GPU/shared telemetry to
# reports\logs\probe_*_canary2560.csv and an environment snapshot.
$ErrorActionPreference = "Stop"
cd "A:\HDD\Work\VS Code - Python\GitHub\Project-8-DocMind-FineTune"

$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$py = "B:\Applications\MiniConda\envs\docmind-ft\python.exe"
$repo = (Get-Location).Path
$logDir = Join-Path $repo "reports\logs"
$Variant = "canary2560"
$gpuCsv = Join-Path $logDir "probe_gpu_$Variant.csv"
$sharedCsv = Join-Path $logDir "probe_shared_$Variant.csv"
$envTxt = Join-Path $logDir "probe_env_$Variant.txt"
$counterScript = "scripts\gpu_shared_counter.ps1"

function Write-Env([string]$Label) {
    "[$Label] $(Get-Date -Format o)" | Out-File -LiteralPath $envTxt -Append -Encoding utf8
    $freeMb = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024, 0)
    "free_ram_mb=$freeMb" | Out-File -LiteralPath $envTxt -Append -Encoding utf8
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv |
        Out-File -LiteralPath $envTxt -Append -Encoding utf8
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader |
        Out-File -LiteralPath $envTxt -Append -Encoding utf8
}

Write-Env "start"
$gpu = Start-Process -FilePath "nvidia-smi" `
    -ArgumentList "--query-gpu=timestamp,memory.used,memory.total,power.draw,power.limit,clocks.sm,clocks.max.sm,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active", "--format=csv", "-l", "5" `
    -RedirectStandardOutput $gpuCsv -NoNewWindow -PassThru
$shared = Start-Process -FilePath "powershell.exe" `
    -ArgumentList "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", $counterScript, "-Variant", $Variant `
    -NoNewWindow -PassThru

$code = 1
try {
    & $py -m docmind_ft.train --config configs/train_probe.yaml --max-steps 6 `
        --output-dir "runs/ft-canary-2560" --data "data/canary_2560.jsonl" --no-final-eval
    $code = $LASTEXITCODE
}
finally {
    Stop-Process -Id $gpu.Id -Force -ErrorAction SilentlyContinue
    Stop-Process -Id $shared.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3
    if (Test-Path -LiteralPath $sharedCsv) {
        $bytes = [IO.File]::ReadAllBytes($sharedCsv)
        if ($bytes.Length -gt 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
            [IO.File]::WriteAllBytes($sharedCsv, $bytes[3..($bytes.Length - 1)])
        }
    }
    Write-Env "end"
}
exit $code
