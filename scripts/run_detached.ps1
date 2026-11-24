# Run a long job detached (script 1.9): anything longer than 10 minutes goes
# through this runner, never through a held-open SSH session.
#
#   scripts\run_detached.ps1 -Name <unique> -Script <file> -Args <a> <b> [-TimeoutMinutes <n>] [-DryRun]
#
# -DryRun writes the args JSON and wrapper, then exits before scheduling the
# task (used by tests/test_run_detached.py).
#
# Behaviour: creates a one-time scheduled task p8-<Name> whose action is
# wscript.exe on a generated .vbs launcher; the launcher starts the job via
# WMI Win32_Process.Create with a SW_HIDE startup object, so no console
# window appears (plain Task Scheduler powershell.exe actions surface visible
# Windows Terminal windows on this laptop - 1.3 deviation, see DECISIONS.md).
# The generated wrapper reads the job script and its arguments from
# <name>-args.json (never from an interpolated command string), runs the job,
# writes a heartbeat line every 60 s to <name>.heartbeat.log, redirects job
# output to <name>.log (stderr to <name>.log.err), writes <name>.done or
# <name>.failed, and removes its own scheduled task in a finally block.
# -TimeoutMinutes (default 0 = no timeout) kills the child process tree after
# the deadline, writes <name>.failed, and exits non-zero. A second start with
# the same name refuses when <name>.done exists (idempotence guard).
#
# The task is registered with NO time trigger and started immediately with
# Start-ScheduledTask; this script then waits up to 60 s for the wrapper's
# heartbeat "start" line and throws (removing the task) if it never appears.
# A time trigger must NOT be used here: wscript.exe exits as soon as the WMI
# Create call returns, so the scheduler sees the task as Ready again within
# a second and a trigger would fire a second wrapper instance - F0.4 proved
# it (two "start" lines, two concurrent jobs). MultipleInstances IgnoreNew
# cannot dedupe because the scheduler never sees the job as running. As a
# second line of defence the wrapper also claims a named mutex
# (p8-job-<Name>) and exits silently if another instance holds it.
#
# Two more F0.4 findings are baked into the wrapper: (1) Start-Process
# -PassThru .ExitCode reads empty in this host (TOOLS.md), so the wrapper
# launches the job with raw .NET [Diagnostics.Process]::Start, streams
# stdout/stderr straight to the log files with CopyToAsync (no pipe buffer
# to fill), and reads .ExitCode from the .NET Process object.
# (2) Unregister-ScheduledTask silently failed from the WMI-spawned wrapper
# (the task survived both instances), so the finally block tries the cmdlet,
# falls back to schtasks.exe /Delete, and logs the outcome to the heartbeat.
# (3) The SSH session runs ELEVATED (high integrity), so a task created by
# it gets a DACL of user:(R) + Administrators:(F); the wrapper runs as the
# same user with a FILTERED token (UAC) and could not delete its own task
# ("Access is denied"). Registration therefore goes through the Task
# Scheduler COM API with an explicit SDDL granting the user full control.
#
# Naming mirrors docmind_ft/detached.py (single source of truth for the
# layout). All job files live under reports\logs\ (git-ignored).
param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$Script,
    [Alias("Args")][string[]]$JobArgs = @(),
    [ValidateRange(0, 1000000)][int]$TimeoutMinutes = 0,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LogsDir = Join-Path $RepoRoot "reports\logs"
if (-not (Test-Path -LiteralPath $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
}

if ($Name -notmatch "^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$") {
    throw "Invalid job name: $Name (letters, digits, '-' and '_', max 64 chars)"
}
$TaskName = "p8-$Name"
$Log = Join-Path $LogsDir "$Name.log"
$LogErr = Join-Path $LogsDir "$Name.log.err"
$Heartbeat = Join-Path $LogsDir "$Name.heartbeat.log"
$Done = Join-Path $LogsDir "$Name.done"
$Failed = Join-Path $LogsDir "$Name.failed"
$ArgsFile = Join-Path $LogsDir "$Name-args.json"
$Wrapper = Join-Path $LogsDir "$Name-wrapper.ps1"
$Launcher = Join-Path $LogsDir "$Name-launcher.vbs"

if (Test-Path -LiteralPath $Done) {
    throw "Job '$Name' already completed ($Done exists); refusing to start again."
}
$existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -ne $existingTask) {
    throw "Scheduled task '$TaskName' already exists; refusing to start again."
}

$JobScript = (Resolve-Path -LiteralPath $Script -ErrorAction Stop).Path

# Fresh run artifacts: drop a stale failure marker / logs from a previous
# attempt (named files only). A .done marker refuses above, never deleted.
# A locked stale file means a previous job's process is still alive (killing
# the wrapper alone orphans the cmd/job chain) - fail with a clear message
# instead of a raw IOException (F0.4 lesson).
foreach ($stale in @($Failed, $Log, $LogErr, $Heartbeat, $ArgsFile)) {
    if (Test-Path -LiteralPath $stale) {
        try { Remove-Item -LiteralPath $stale -Force -ErrorAction Stop }
        catch {
            throw "Cannot remove stale file $stale (locked - a previous '$Name' job process may still be running; kill it first). $($_.Exception.Message)"
        }
    }
}

function Write-TextFile([string]$Path, [string]$Text) {
    # BOM-free UTF-8 (PowerShell 5.1 Set-Content -Encoding utf8 writes a BOM).
    [IO.File]::WriteAllText($Path, $Text, (New-Object Text.UTF8Encoding $false))
}

# The wrapper reads this JSON; arguments are never interpolated into code or
# a command line by the parent.
$argsPayload = [ordered]@{
    script = $JobScript
    args = @($JobArgs)
    timeout_minutes = $TimeoutMinutes
}
Write-TextFile -Path $ArgsFile -Text (ConvertTo-Json -InputObject $argsPayload -Depth 3 -Compress)

$wrapperText = @"
# Generated wrapper for detached job '$Name'. Do not edit; created by scripts/run_detached.ps1.
`$ErrorActionPreference = 'Continue'
`$taskName = '$TaskName'
`$log = Join-Path `$PSScriptRoot '$Name.log'
`$logErr = Join-Path `$PSScriptRoot '$Name.log.err'
`$heartbeat = Join-Path `$PSScriptRoot '$Name.heartbeat.log'
`$done = Join-Path `$PSScriptRoot '$Name.done'
`$failed = Join-Path `$PSScriptRoot '$Name.failed'
`$argsFile = Join-Path `$PSScriptRoot '$Name-args.json'
# Single-instance guard: a second wrapper for the same job exits silently.
# (An abandoned mutex means the previous owner died; treat it as acquired.)
`$mutex = New-Object System.Threading.Mutex(`$false, 'p8-job-$Name')
`$acquired = `$false
try { `$acquired = `$mutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { `$acquired = `$true }
if (-not `$acquired) { exit 0 }
`$started = Get-Date
`$timedOut = `$false
try {
    if (Test-Path -LiteralPath `$done) { exit 0 }
    "[`$(Get-Date -Format o)] start task=`$taskName" | Out-File -LiteralPath `$heartbeat -Append -Encoding utf8
    `$job = Get-Content -LiteralPath `$argsFile -Raw | ConvertFrom-Json
    `$jobScript = [string]`$job.script
    `$jobArgs = @()
    if (`$null -ne `$job.args) { `$jobArgs = @(`$job.args) }
    `$timeoutMinutes = [int]`$job.timeout_minutes
    # Windows PowerShell 5.1 has no ProcessStartInfo.ArgumentList; quote each
    # value for CreateProcess instead of using cmd.exe.
    function Quote-Arg([string]`$Value) {
        `$escaped = `$Value -replace '(\\*)"', '`$1`$1\"'
        `$escaped = `$escaped -replace '(\\+)\z', '`$1`$1'
        return '"' + `$escaped + '"'
    }
    `$argList = @('-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', (Quote-Arg `$jobScript))
    foreach (`$arg in `$jobArgs) { `$argList += Quote-Arg ([string]`$arg) }
    `$psi = New-Object System.Diagnostics.ProcessStartInfo
    `$psi.FileName = 'powershell.exe'
    `$psi.Arguments = `$argList -join ' '
    `$psi.UseShellExecute = `$false
    `$psi.CreateNoWindow = `$true
    `$psi.RedirectStandardOutput = `$true
    `$psi.RedirectStandardError = `$true
    `$proc = [System.Diagnostics.Process]::Start(`$psi)
    `$stdoutStream = [IO.File]::Open(`$log, 'Create', 'Write', 'Read')
    `$stderrStream = [IO.File]::Open(`$logErr, 'Create', 'Write', 'Read')
    `$stdoutTask = `$proc.StandardOutput.BaseStream.CopyToAsync(`$stdoutStream)
    `$stderrTask = `$proc.StandardError.BaseStream.CopyToAsync(`$stderrStream)
    `$deadline = `$null
    if (`$timeoutMinutes -gt 0) { `$deadline = `$started.AddMinutes(`$timeoutMinutes) }
    `$nextHeartbeat = `$started.AddSeconds(60)
    while (-not `$proc.HasExited) {
        Start-Sleep -Seconds 5
        if (`$proc.HasExited) { break }
        if (`$null -ne `$deadline -and (Get-Date) -ge `$deadline) {
            "[`$(Get-Date -Format o)] timeout `${timeoutMinutes}m; killing process tree" | Out-File -LiteralPath `$heartbeat -Append -Encoding utf8
            try { & "`$env:SystemRoot\System32\taskkill.exe" /PID `$proc.Id /T /F 2>`$null | Out-Null } catch { }
            try { if (-not `$proc.HasExited) { `$proc.Kill() } } catch { }
            `$timedOut = `$true
            break
        }
        if ((Get-Date) -ge `$nextHeartbeat) {
            `$elapsed = [int]((Get-Date) - `$started).TotalSeconds
            "[`$(Get-Date -Format o)] heartbeat `${elapsed}s" | Out-File -LiteralPath `$heartbeat -Append -Encoding utf8
            `$nextHeartbeat = (Get-Date).AddSeconds(60)
        }
    }
    `$proc.WaitForExit()
    `$stdoutTask.Wait()
    `$stderrTask.Wait()
    `$stdoutStream.Dispose()
    `$stderrStream.Dispose()
    `$exitCode = `$proc.ExitCode
    `$ended = Get-Date
    "[`$(Get-Date -Format o)] end exit=`$exitCode" | Out-File -LiteralPath `$heartbeat -Append -Encoding utf8
    if (`$timedOut) {
        `$marker = "timeout_minutes=`$timeoutMinutes`nstarted=`$(`$started.ToString('o'))`nended=`$(`$ended.ToString('o'))`n"
        [IO.File]::WriteAllText(`$failed, `$marker, (New-Object Text.UTF8Encoding `$false))
    } else {
        `$marker = "exit_code=`$exitCode`nstarted=`$(`$started.ToString('o'))`nended=`$(`$ended.ToString('o'))`n"
        if (`$exitCode -eq 0) { [IO.File]::WriteAllText(`$done, `$marker, (New-Object Text.UTF8Encoding `$false)) }
        else { [IO.File]::WriteAllText(`$failed, `$marker, (New-Object Text.UTF8Encoding `$false)) }
    }
} catch {
    `$marker = "error: `$(`$_.Exception.Message)`n"
    [IO.File]::WriteAllText(`$failed, `$marker, (New-Object Text.UTF8Encoding `$false))
} finally {
    try { if (`$acquired) { `$mutex.ReleaseMutex() } } catch { }
    `$mutex.Dispose()
    `$unreg = 'ok'
    try {
        Unregister-ScheduledTask -TaskName `$taskName -Confirm:`$false -ErrorAction Stop
    } catch {
        `$unreg = "cmdlet failed: `$(`$_.Exception.Message)"
        try {
            schtasks.exe /Delete /TN `$taskName /F | Out-Null
            if (`$LASTEXITCODE -eq 0) { `$unreg = 'ok via schtasks (cmdlet failed)' }
            else { `$unreg = "`$unreg; schtasks exit=`$LASTEXITCODE" }
        } catch { `$unreg = "`$unreg; schtasks threw: `$(`$_.Exception.Message)" }
    }
    try { "[`$(Get-Date -Format o)] unregister: `$unreg" | Out-File -LiteralPath `$heartbeat -Append -Encoding utf8 } catch { }
}
if (`$timedOut) { exit 1 }
"@
Write-TextFile -Path $Wrapper -Text $wrapperText

# VBScript launcher: WMI Win32_Process.Create with a SW_HIDE startup object.
# Plain WMI Create can still show a pseudoconsole; the startup object is required.
$vbsCmd = "powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$Wrapper`""
$vbsText = "' Generated launcher for detached job '$Name'. Do not edit; created by scripts/run_detached.ps1." + "`r`n" +
    "Set svc = GetObject(`"winmgmts:\\.\root\cimv2`")" + "`r`n" +
    "Set startup = svc.Get(`"Win32_ProcessStartup`").SpawnInstance_" + "`r`n" +
    "startup.ShowWindow = 0" + "`r`n" +
    "Set proc = svc.Get(`"Win32_Process`")" + "`r`n" +
    "cmd = `"" + $vbsCmd.Replace("`"", "`"`"") + "`"" + "`r`n" +
    "rc = proc.Create(cmd, Null, startup, pid)" + "`r`n"
Write-TextFile -Path $Launcher -Text $vbsText

if ($DryRun) {
    Write-Output "dry-run: args=$ArgsFile wrapper=$Wrapper launcher=$Launcher"
    return
}

# No time trigger (see header): start immediately, then verify the wrapper
# actually launched by waiting for its heartbeat "start" line.
#
# Registration goes through the Task Scheduler COM API (not
# Register-ScheduledTask) so the task's security descriptor can be set
# explicitly: the SSH session runs ELEVATED, and a task created by the
# elevated token gets a DACL of user:(R) + Administrators:(F) - the wrapper
# runs as the same user but with a FILTERED token (UAC), so it could not
# delete its own task ("Access is denied", F0.4). The SDDL below grants the
# user full control, so the filtered-token wrapper can unregister.
$sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$sddl = "D:(A;;GA;;;$sid)(A;;GA;;;SY)(A;;GA;;;BA)"
$scheduleService = New-Object -ComObject Schedule.Service
$scheduleService.Connect()
$taskFolder = $scheduleService.GetFolder('\')
$taskDef = $scheduleService.NewTask(0)
$taskDef.RegistrationInfo.Description = "Project 8 detached job '$Name'"
$taskDef.Settings.MultipleInstances = 2  # TASK_INSTANCES_IGNORE_NEW
$taskDef.Settings.DisallowStartIfOnBatteries = $false  # (cmdlet: -AllowStartIfOnBatteries)
$taskDef.Settings.StopIfGoingOnBatteries = $false  # (cmdlet: -DontStopIfGoingOnBatteries)
$taskDef.Settings.StartWhenAvailable = $true
$taskDef.Settings.ExecutionTimeLimit = 'PT0S'  # no limit; the wrapper owns the lifecycle
$taskDef.Principal.UserId = $sid
$taskDef.Principal.LogonType = 3  # TASK_LOGON_INTERACTIVE_TOKEN
$taskDef.Principal.RunLevel = 0  # TASK_RUNLEVEL_LUA (filtered token; no elevation needed)
$taskAction = $taskDef.Actions.Create(0)  # TASK_ACTION_EXEC
$taskAction.Path = 'wscript.exe'
$taskAction.Arguments = "`"$Launcher`""
$taskFolder.RegisterTaskDefinition($TaskName, $taskDef, 6, $null, $null, 3, $sddl) | Out-Null
Start-ScheduledTask -TaskName $TaskName
$deadline = (Get-Date).AddSeconds(60)
$startedOk = $false
while ((Get-Date) -lt $deadline) {
    if (Test-Path -LiteralPath $Heartbeat) {
        $content = Get-Content -LiteralPath $Heartbeat -ErrorAction SilentlyContinue
        if ($content -match "start task=$TaskName") { $startedOk = $true; break }
    }
    Start-Sleep -Seconds 2
}
if (-not $startedOk) {
    try { Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop } catch { }
    throw "Job '$Name' failed to start within 60s (no heartbeat start line); task removed."
}

Write-Output "started job '$Name' task=$TaskName log=$Log heartbeat=$Heartbeat"
