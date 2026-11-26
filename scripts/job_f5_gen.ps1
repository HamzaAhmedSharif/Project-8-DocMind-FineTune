# F5.2: generate FT-main on the two remaining sealed splits (qtest, hold).
# paper16 is already done (smoke test 2026-11-27 02:50, 16/16, ledger written).
# Run via run_detached so it survives disconnects; resumable (finished ids skipped).
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

Write-Host "=== FT-main qtest -> evaluation\results\ft_main_qtest.jsonl ==="
& $py scripts\run_eval.py --model FT-main --split qtest --backend hf --model-id $modelId --revision $revision --adapter $adapter --prompt-file $promptFile --out evaluation\results\ft_main_qtest.jsonl
if ($LASTEXITCODE -ne 0) { Write-Host "FT-main qtest failed exit $LASTEXITCODE" -ForegroundColor Red; exit $LASTEXITCODE }

Write-Host "=== FT-main hold -> evaluation\results\ft_main_hold.jsonl ==="
& $py scripts\run_eval.py --model FT-main --split hold --backend hf --model-id $modelId --revision $revision --adapter $adapter --prompt-file $promptFile --out evaluation\results\ft_main_hold.jsonl
if ($LASTEXITCODE -ne 0) { Write-Host "FT-main hold failed exit $LASTEXITCODE" -ForegroundColor Red; exit $LASTEXITCODE }

Write-Host "FT-main sealed generation complete (qtest + hold; paper16 was already done)."
exit 0
