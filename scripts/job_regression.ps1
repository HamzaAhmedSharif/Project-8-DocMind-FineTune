$ErrorActionPreference = "Stop"
cd "A:\HDD\Work\VS Code - Python\GitHub\Project-8-DocMind-FineTune"
$env:HF_HOME = "B:\p8\hf-cache"
$env:HF_HUB_OFFLINE = "0"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:PYTHONIOENCODING = "utf-8"
B:\Applications\MiniConda\envs\docmind-ft\python.exe -m docmind_ft.regression
exit $LASTEXITCODE
