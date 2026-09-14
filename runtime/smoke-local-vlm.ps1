param(
    [string]$ImagePath = "datasets\pilot\assets\scene_01_two_facing.png",
    [int]$Port = 8080
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path $PSScriptRoot -Parent
$resolvedImage = (Resolve-Path (Join-Path $projectRoot $ImagePath)).Path
$extension = [IO.Path]::GetExtension($resolvedImage).TrimStart(".").ToLowerInvariant()
$mimeType = if ($extension -eq "jpg" -or $extension -eq "jpeg") { "image/jpeg" } else { "image/png" }
$encodedImage = [Convert]::ToBase64String([IO.File]::ReadAllBytes($resolvedImage))

$body = @{
    model = "qwen3.5-4b-q4km"
    temperature = 0
    max_tokens = 128
    messages = @(@{
        role = "user"
        content = @(
            @{ type = "image_url"; image_url = @{ url = "data:$mimeType;base64,$encodedImage" } }
            @{ type = "text"; text = "Count the visible people. Return only JSON with integer visible_count and a short evidence string." }
        )
    })
    response_format = @{
        type = "json_schema"
        json_schema = @{
            name = "vision_smoke"
            strict = $true
            schema = @{
                type = "object"
                properties = @{
                    visible_count = @{ type = "integer" }
                    evidence = @{ type = "string" }
                }
                required = @("visible_count", "evidence")
                additionalProperties = $false
            }
        }
    }
} | ConvertTo-Json -Depth 12 -Compress

$result = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$Port/v1/chat/completions" `
    -ContentType "application/json" -Body $body -TimeoutSec 180
$result | ConvertTo-Json -Depth 12
