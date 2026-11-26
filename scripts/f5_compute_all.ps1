# F5.3 step 1: per-(model,split) metrics via scripts/compute_metrics.py.
# Writes evaluation/results/metrics_<model>_<split>.json for every model+split.
# Test splits need DOCMIND_FT_ALLOW_TEST=1 (set here, as final_eval.ps1 does).
$ErrorActionPreference = "Stop"
cd "A:\HDD\Work\VS Code - Python\GitHub\Project-8-DocMind-FineTune"
$env:HF_HOME = "B:\p8\hf-cache"
$env:HF_HUB_OFFLINE = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:DOCMIND_FT_ALLOW_TEST = "1"
$py = "B:\Applications\MiniConda\envs\docmind-ft\python.exe"

$pairs = @(
    @{ Model="b0";  Split="qtest";   Out="evaluation\results\b0_qtest.jsonl" },
    @{ Model="b0";  Split="hold";    Out="evaluation\results\b0_hold.jsonl" },
    @{ Model="b0";  Split="paper16"; Out="evaluation\results\b0_paper16.jsonl" },
    @{ Model="b1";  Split="qtest";   Out="evaluation\results\b1_qtest.jsonl" },
    @{ Model="b1";  Split="hold";    Out="evaluation\results\b1_hold.jsonl" },
    @{ Model="b1";  Split="paper16"; Out="evaluation\results\b1_paper16.jsonl" },
    @{ Model="ft_main"; Split="qtest";   Out="evaluation\results\ft_main_qtest.jsonl" },
    @{ Model="ft_main"; Split="hold";    Out="evaluation\results\ft_main_hold.jsonl" },
    @{ Model="ft_main"; Split="paper16"; Out="evaluation\results\ft_main_paper16.jsonl" },
    @{ Model="r";   Split="hold";    Out="evaluation\results\r_hold.jsonl" },
    @{ Model="r";   Split="paper16"; Out="evaluation\results\r_paper16.jsonl" }
)

foreach ($p in $pairs) {
    $metricsOut = "evaluation\results\metrics_$($p.Model)_$($p.Split).json"
    Write-Host "=== $($p.Model) $($p.Split) -> $metricsOut ==="
    & $py scripts\compute_metrics.py --split $p.Split --outputs $p.Out --out $metricsOut
    if ($LASTEXITCODE -ne 0) {
        Write-Host "$($p.Model) $($p.Split) FAILED exit $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}
Write-Host "All per-(model,split) metrics computed."
exit 0
