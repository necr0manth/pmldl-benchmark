$ErrorActionPreference = "Stop"
$comparisonRoot = $PSScriptRoot
$pidPath = Join-Path $comparisonRoot "current.pid"
$statePath = Join-Path $comparisonRoot "current.json"
$serverPath = [IO.Path]::GetFullPath((Join-Path $comparisonRoot "..\llama.cpp-b10941-vulkan\llama-server.exe"))

if (-not (Test-Path $pidPath)) {
    Write-Output "No comparison PID file; nothing stopped."
    exit 0
}
$savedPid = [int](Get-Content -Raw $pidPath)
$process = Get-Process -Id $savedPid -ErrorAction SilentlyContinue
if (-not $process) {
    Remove-Item -LiteralPath $pidPath -Force
    Remove-Item -LiteralPath $statePath -Force -ErrorAction SilentlyContinue
    Write-Output "Stale comparison PID file removed; process $savedPid was not running."
    exit 0
}
if ($process.Path -ne $serverPath) {
    throw "PID $savedPid does not belong to the configured llama-server; refusing to stop it."
}
Stop-Process -Id $savedPid
$process.WaitForExit(10000) | Out-Null
Remove-Item -LiteralPath $pidPath -Force
Remove-Item -LiteralPath $statePath -Force -ErrorAction SilentlyContinue
Write-Output "Stopped comparison VLM PID=$savedPid."
