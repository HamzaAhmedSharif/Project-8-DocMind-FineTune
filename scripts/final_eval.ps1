# F5.2 / F5.3 sealed test evaluation for the fine-tuned model (FT-main).
# This is the ONLY place that sets DOCMIND_FT_ALLOW_TEST=1.
# Default (no flags): generate FT-main outputs on Q-TEST, HOLD, PAPER16
#   (detached + resumable via run_detached; finished ids are skipped by
#   run_eval.py through the I7 ledger guard).
# With -Compute: compute test_metrics.json from the generated outputs
#   (F5.3; only after generation is complete for every model+split).
# Usage:
#   scripts\final_eval.ps1                       (generate FT-main)
#   scripts\final_eval.ps1 -Compute              (metrics, F5.3)
#   scripts\final_eval.ps1 -Compute -RecomputeOnly  (verify, F5.7)
param(
    [switch]$Compute,
    [switch]$RecomputeOnly
)

$ErrorActionPreference = "Stop"
cd "A:\HDD\Work\VS Code - Python\GitHub\Project-8-DocMind-FineTune"

$env:HF_HOME = "B:\p8\hf-cache"
$env:HF_HUB_OFFLINE = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:DOCMIND_FT_ALLOW_TEST = "1"
$py = "B:\Applications\MiniConda\envs\docmind-ft\python.exe"

$modelId = "meta-llama/Llama-3.2-3B-Instruct"
$revision = "0cb88a4f764b7a12671c53f0838cd831a0843b95"
$adapter = "runs\ft-main"
$promptFile = "evaluation\baselines\b1_frozen.txt"

if ($Compute) {
    # F5.3: compute metrics from the generated outputs.
    # -RecomputeOnly re-derives every table and compares with the committed JSON.
    $args = @("scripts\final_eval_compute.py")
    if ($RecomputeOnly) { $args += "--recompute-only" }
    & $py @args
    exit $LASTEXITCODE
}

# F5.2: generate FT-main on the three sealed splits.
$splits = @(
    @{ Split = "qtest";   Out = "evaluation\results\ft_main_qtest.jsonl" },
    @{ Split = "hold";    Out = "evaluation\results\ft_main_hold.jsonl" },
    @{ Split = "paper16"; Out = "evaluation\results\ft_main_paper16.jsonl" }
)

foreach ($s in $splits) {
    Write-Host "=== FT-main $($s.Split) -> $($s.Out) ==="
    & $py scripts\run_eval.py --model FT-main --split $s.Split --backend hf `
        --model-id $modelId --revision $revision --adapter $adapter `
        --prompt-file $promptFile --out $s.Out
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FT-main $($s.Split) failed with exit $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host "FT-main sealed generation complete (qtest, hold, paper16)."
exit 0
