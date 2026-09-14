param(
    [Parameter(Mandatory = $true)][ValidateSet("qwen", "smolvlm2", "internvl", "gemma3")][string]$Model,
    [Parameter(Mandatory = $true)][string]$LogDirectory,
    [int]$Port = 8080
)

$ErrorActionPreference = "Stop"
$comparisonRoot = $PSScriptRoot
$projectRoot = Split-Path (Split-Path $comparisonRoot -Parent) -Parent
$catalogPath = Join-Path $comparisonRoot "models.json"
$catalog = Get-Content -Raw $catalogPath | ConvertFrom-Json
$entry = $catalog.models.$Model
$serverPath = [IO.Path]::GetFullPath((Join-Path $comparisonRoot $catalog.runtime.server))
$modelPath = [IO.Path]::GetFullPath((Join-Path $comparisonRoot $entry.model))
$projectorPath = [IO.Path]::GetFullPath((Join-Path $comparisonRoot $entry.projector))
$pidPath = Join-Path $comparisonRoot "current.pid"
$statePath = Join-Path $comparisonRoot "current.json"
$resolvedLogDirectory = [IO.Path]::GetFullPath((Join-Path $projectRoot $LogDirectory))

function Get-ArtifactSha256([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    try {
        $sha = [Security.Cryptography.SHA256]::Create()
        try {
            return ([BitConverter]::ToString($sha.ComputeHash($stream))).Replace("-", "").ToLowerInvariant()
        } finally {
            $sha.Dispose()
        }
    } finally {
        $stream.Dispose()
    }
}

if (-not $resolvedLogDirectory.StartsWith($projectRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "LogDirectory must resolve inside $projectRoot"
}
New-Item -ItemType Directory -Force $resolvedLogDirectory | Out-Null
$stdoutPath = Join-Path $resolvedLogDirectory "$Model.stdout.log"
$stderrPath = Join-Path $resolvedLogDirectory "$Model.stderr.log"
if ((Test-Path $stdoutPath) -or (Test-Path $stderrPath)) {
    throw "Refusing to overwrite existing model logs in $resolvedLogDirectory"
}
if ((Test-Path $pidPath) -or (Get-Process llama-server -ErrorAction SilentlyContinue)) {
    throw "A llama-server may already be running. Stop it explicitly before starting $Model."
}
$listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
if ($listener) {
    throw "Port $Port is already owned by PID $($listener.OwningProcess)."
}
foreach ($artifact in @(
    @{ Path = $modelPath; Expected = $entry.model_sha256 },
    @{ Path = $projectorPath; Expected = $entry.projector_sha256 }
)) {
    if (-not (Test-Path $artifact.Path -PathType Leaf)) {
        throw "Missing artifact: $($artifact.Path)"
    }
    $actual = Get-ArtifactSha256 $artifact.Path
    if ($actual -ne $artifact.Expected) {
        throw "SHA-256 mismatch for $($artifact.Path): $actual"
    }
}

$arguments = @(
    "--model", $modelPath,
    "--mmproj", $projectorPath,
    "--alias", $entry.alias,
    "--host", "127.0.0.1",
    "--port", $Port,
    "--ctx-size", "8192",
    "--parallel", "1",
    "--n-gpu-layers", "all",
    "--jinja",
    "--log-timestamps"
)
if ($entry.reasoning_off) {
    $arguments += @("--reasoning", "off")
}
$process = Start-Process -FilePath $serverPath -ArgumentList $arguments `
    -WorkingDirectory $projectRoot -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath -WindowStyle Hidden -PassThru
$process.Id | Set-Content -NoNewline $pidPath
$state = @{
    model_key = $Model
    alias = $entry.alias
    pid = $process.Id
    port = $Port
    started_at = [DateTimeOffset]::UtcNow.ToString("o")
    server = $serverPath
    model = $modelPath
    projector = $projectorPath
    arguments = $arguments
    stdout = $stdoutPath
    stderr = $stderrPath
}
$state | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 $statePath
$state | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 (Join-Path $resolvedLogDirectory "$Model.state.json")

$healthUrl = "http://127.0.0.1:$Port/health"
for ($attempt = 0; $attempt -lt 180; $attempt++) {
    if ($process.HasExited) {
        throw "llama-server exited with code $($process.ExitCode); see $stderrPath"
    }
    try {
        $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
        if ($health.status -eq "ok") {
            $models = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/models" -TimeoutSec 5
            $served = $models.data[0].id
            if ($served -ne $entry.alias) {
                throw "Endpoint model mismatch: expected $($entry.alias), got $served"
            }
            $models | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 (Join-Path $resolvedLogDirectory "$Model.models.json")
            Write-Output "Comparison VLM ready: PID=$($process.Id) model=$served"
            exit 0
        }
    } catch {
        Start-Sleep -Seconds 1
    }
}
throw "Timed out waiting for $healthUrl; PID=$($process.Id); see $stderrPath"
