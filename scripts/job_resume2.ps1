# F3.4 resume step: resume from checkpoint-4 to step 12.
# Launched by scripts\run_detached.ps1. Output runs/ft-resume.
# The first 12-step run completed before the HO-2 kill landed, so the
# resume is run from the valid checkpoint-4 saved by that run.
$ErrorActionPreference = "Continue"
cd "A:\HDD\Work\VS Code - Python\GitHub\Project-8-DocMind-FineTune"
$py = "B:\Applications\MiniConda\envs\docmind-ft\python.exe"
& $py -m docmind_ft.train --config configs/train_resume.yaml --max-steps 12 --output-dir runs/ft-resume --data data/learn8.jsonl --no-final-eval --resume-from runs/ft-resume/checkpoint-4
exit $LASTEXITCODE
