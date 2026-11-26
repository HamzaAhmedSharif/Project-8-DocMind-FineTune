# F3.3 learn-test re-run (torch 2.6.0 baseline check): 64 steps at GA 1
# on learn8, constant_with_warmup, LR 2e-4. Output runs/ft-learn.
# Launched by scripts\run_detached.ps1. Re-verifies the learning path
# after the torch 2.5.1 -> 2.6.0 upgrade; the verified F3.3 artifacts
# were backed up to runs/ft-learn-f33 before this run.
$ErrorActionPreference = "Continue"
cd "A:\HDD\Work\VS Code - Python\GitHub\Project-8-DocMind-FineTune"
$py = "B:\Applications\MiniConda\envs\docmind-ft\python.exe"
& $py -m docmind_ft.train --config configs/train_learn.yaml --max-steps 64 --output-dir runs/ft-learn --data data/learn8.jsonl --no-final-eval
exit $LASTEXITCODE
