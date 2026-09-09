param(
    [Parameter(Mandatory = $true)][string]$Session,
    [Parameter(Mandatory = $true)][string]$TrainingLog,
    [Parameter(Mandatory = $true)][string]$OutputDir
)

$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$watchLog = Join-Path $OutputDir 'urgent_export_watchdog.log'
$base64Path = Join-Path $OutputDir 'continuous_pair_distilbert.safetensors.b64'
$modelPath = Join-Path $OutputDir 'continuous_pair_distilbert.safetensors'
$streamScript = '/mnt/e/workspace/EmbeddiBERT/scripts/colab_stream_final_model.py'

function Write-Watch([string]$Message) {
    "$(Get-Date -Format o) $Message" | Add-Content -LiteralPath $watchLog
}

Write-Watch "watching $Session"
while ($true) {
    $content = if (Test-Path -LiteralPath $TrainingLog) {
        Get-Content -LiteralPath $TrainingLog -Raw
    } else { '' }
    if ($content -match 'FloatingPointError|Traceback') {
        Write-Watch 'training failure detected'
        exit 2
    }
    if ($content -match '"event": "complete"') { break }
    Start-Sleep -Seconds 1
}

Write-Watch 'completion detected; streaming final model immediately'
$drive = $base64Path.Substring(0, 1).ToLowerInvariant()
$base64Wsl = "/mnt/$drive/" + (($base64Path.Substring(3)) -replace '\\', '/')
$modelWsl = "/mnt/$drive/" + (($modelPath.Substring(3)) -replace '\\', '/')
$command = "mighty-colab --auth=oauth2 exec -s '$Session' -f '$streamScript' --timeout 300 > '$base64Wsl'"
wsl -e bash -lc $command
if ($LASTEXITCODE -ne 0) { Write-Watch 'stream failed'; exit 3 }
wsl -e bash -lc "base64 -d '$base64Wsl' > '$modelWsl'"
if ($LASTEXITCODE -ne 0) { Write-Watch 'decode failed'; exit 4 }

$actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $modelPath).Hash.ToLowerInvariant()
$completion = ($content -split "`n" | Where-Object { $_ -match '"event": "complete"' } | Select-Object -Last 1) | ConvertFrom-Json
$expectedHash = $completion.model.sha256.ToLowerInvariant()
$status = if ($actualHash -eq $expectedHash) { 'verified' } else { 'hash_mismatch' }
[ordered]@{
    status = $status
    expected_sha256 = $expectedHash
    actual_sha256 = $actualHash
    bytes = (Get-Item -LiteralPath $modelPath).Length
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputDir 'verification.json')
Write-Watch "export $status"
