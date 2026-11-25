# Sample GPU shared-memory counters as clean CSV: timestamp,instance,bytes.
# Started by a job script and killed in its finally block.
param(
    [Parameter(Mandatory = $true)][string]$Variant
)
$ErrorActionPreference = "Stop"
$outPath = Join-Path (Split-Path $PSScriptRoot -Parent) "reports\logs\probe_shared_$Variant.csv"
Get-Counter '\GPU Adapter Memory(*)\Shared Usage' -SampleInterval 5 -Continuous |
    ForEach-Object {
        $_.CounterSamples | ForEach-Object {
            "{0},{1},{2}" -f $_.Timestamp.ToString("o"), $_.InstanceName, [int64]$_.CookedValue
        }
    } | Out-File -FilePath $outPath -Encoding utf8
