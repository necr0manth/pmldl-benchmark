param([int]$Port = 8080)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path $PSScriptRoot -Parent
$serverPath = Join-Path $PSScriptRoot "llama.cpp-b10941-vulkan\llama-server.exe"
$modelPath = Join-Path $projectRoot "models\Qwen3.5-4B-GGUF\Qwen3.5-4B-Q4_K_M.gguf"
$projectorPath = Join-Path $projectRoot "models\Qwen3.5-4B-GGUF\mmproj-F16.gguf"
$pidPath = Join-Path $PSScriptRoot "local-vlm.pid"
$stdoutPath = Join-Path $PSScriptRoot "logs\local-vlm.stdout.log"
$stderrPath = Join-Path $PSScriptRoot "logs\local-vlm.stderr.log"

if (Test-Path $pidPath) {
    $savedPid = [int](Get-Content -Raw $pidPath)
    $savedProcess = Get-Process -Id $savedPid -ErrorAction SilentlyContinue
    if ($savedProcess -and $savedProcess.Path -eq $serverPath) {
        Write-Output "Local VLM already running: PID=$savedPid port=$Port"
        exit 0
    }
    Remove-Item -LiteralPath $pidPath -Force
}

$listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
if ($listener) {
    throw "Port $Port is already owned by PID $($listener.OwningProcess)."
}

$arguments = @(
    "--model", $modelPath,
    "--mmproj", $projectorPath,
    "--alias", "qwen3.5-4b-q4km",
    "--host", "127.0.0.1",
    "--port", $Port,
    "--ctx-size", "8192",
    "--parallel", "1",
    "--n-gpu-layers", "all",
    "--reasoning", "off",
    "--jinja",
    "--log-timestamps"
)

$process = Start-Process -FilePath $serverPath -ArgumentList $arguments `
    -WorkingDirectory $projectRoot -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath -WindowStyle Hidden -PassThru
$process.Id | Set-Content -NoNewline $pidPath

$healthUrl = "http://127.0.0.1:$Port/health"
for ($attempt = 0; $attempt -lt 180; $attempt++) {
    if ($process.HasExited) {
        throw "llama-server exited with code $($process.ExitCode); see $stderrPath"
    }
    try {
        $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
        if ($health.status -eq "ok") {
            Write-Output "Local VLM ready: PID=$($process.Id) endpoint=http://127.0.0.1:$Port/v1 model=qwen3.5-4b-q4km"
            exit 0
        }
    } catch {
        Start-Sleep -Seconds 1
    }
}

throw "Timed out waiting for $healthUrl; PID=$($process.Id); see $stderrPath"
