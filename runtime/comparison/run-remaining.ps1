$ErrorActionPreference = "Stop"
$comparisonRoot = $PSScriptRoot
$projectRoot = Split-Path (Split-Path $comparisonRoot -Parent) -Parent
$runRoot = Join-Path $projectRoot "runs\model-comparison-20260913"
$queueLog = Join-Path $runRoot "remaining-queue-v2.log"
$queueState = Join-Path $runRoot "remaining-queue-v2-state.json"
$queuePid = Join-Path $runRoot "remaining-queue-v2.pid"
$srcPath = Join-Path $projectRoot "src"
$models = @(
    @{ Key = "smolvlm2"; Config = "configs\comparison-smolvlm2-2.2b-q4km.json"; Output = "smolvlm2-2.2b-q4km" },
    @{ Key = "internvl"; Config = "configs\comparison-internvl3.5-4b-q4km.json"; Output = "internvl3.5-4b-q4km" },
    @{ Key = "gemma3"; Config = "configs\comparison-gemma3-4b-it-q4km.json"; Output = "gemma3-4b-it-q4km" }
)

New-Item -ItemType Directory -Force $runRoot | Out-Null
if (Test-Path $queueLog) {
    throw "Refusing to overwrite existing queue log: $queueLog"
}
$PID | Set-Content -NoNewline $queuePid
@{ status = "running"; pid = $PID; started_at = [DateTimeOffset]::UtcNow.ToString("o") } |
    ConvertTo-Json | Set-Content -Encoding utf8 $queueState

function Write-QueueLog([string]$Message) {
    $line = "{0} {1}" -f [DateTimeOffset]::UtcNow.ToString("o"), $Message
    Add-Content -Encoding utf8 -Path $queueLog -Value $line
    Write-Output $line
}

$env:PYTHONPATH = $srcPath
$failures = 0
try {
    & (Join-Path $projectRoot "runtime\stop-local-vlm.ps1") | ForEach-Object { Write-QueueLog $_ }
    foreach ($entry in $models) {
        $serverStarted = $false
        try {
            Write-QueueLog "START model=$($entry.Key)"
            $logDirectory = "runs\model-comparison-20260913\server-logs\$($entry.Key)"
            & (Join-Path $comparisonRoot "start-model.ps1") -Model $entry.Key -LogDirectory $logDirectory |
                ForEach-Object { Write-QueueLog $_ }
            if ($LASTEXITCODE -ne 0) { throw "start-model failed with exit code $LASTEXITCODE" }
            $serverStarted = $true
            & python (Join-Path $comparisonRoot "run_checkpoint.py") `
                (Join-Path $projectRoot "datasets\expanded\cases.jsonl") `
                --config (Join-Path $projectRoot $entry.Config) `
                --model-key $entry.Key `
                --output (Join-Path $runRoot $entry.Output) |
                ForEach-Object { Write-QueueLog $_ }
            if ($LASTEXITCODE -ne 0) { throw "benchmark failed with exit code $LASTEXITCODE" }
            Write-QueueLog "COMPLETE model=$($entry.Key) cases=106"
        } catch {
            $failures += 1
            Write-QueueLog "FAILED model=$($entry.Key) error=$($_.Exception.Message)"
        } finally {
            if ($serverStarted -or (Test-Path (Join-Path $comparisonRoot "current.pid"))) {
                & (Join-Path $comparisonRoot "stop-model.ps1") | ForEach-Object { Write-QueueLog $_ }
            }
        }
    }
} finally {
    if (Test-Path (Join-Path $comparisonRoot "current.pid")) {
        & (Join-Path $comparisonRoot "stop-model.ps1") | ForEach-Object { Write-QueueLog $_ }
    }
    & (Join-Path $projectRoot "runtime\start-local-vlm.ps1") | ForEach-Object { Write-QueueLog $_ }
    Write-QueueLog "QUEUE_END failures=$failures original_qwen_restored=true"
    @{
        status = $(if ($failures -eq 0) { "complete" } else { "failed" })
        pid = $PID
        finished_at = [DateTimeOffset]::UtcNow.ToString("o")
        failures = $failures
        original_qwen_restored = $true
    } | ConvertTo-Json | Set-Content -Encoding utf8 $queueState
}
exit $failures
