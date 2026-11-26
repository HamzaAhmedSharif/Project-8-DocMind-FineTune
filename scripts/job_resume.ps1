# F3.4 resume test job: 12 micro-steps at GA 1 on learn8, save_steps 4.
# Launched by scripts\run_detached.ps1. Output runs/ft-resume.
# HO-2: Hamza kills the python process by hand at heartbeat step 6;
# Sol resumes with --resume-from runs/ft-resume/checkpoint-4 --max-steps 12.
$ErrorActionPreference = "Continue"
cd "A:\HDD\Work\VS Code - Python\GitHub\Project-8-DocMind-FineTune"
$py = "B:\Applications\MiniConda\envs\docmind-ft\python.exe"
& $py -m docmind_ft.train --config configs/train_resume.yaml --max-steps 12 --output-dir runs/ft-resume --data data/learn8.jsonl --no-final-eval
exit $LASTEXITCODE
