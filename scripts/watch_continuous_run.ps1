param(
    [string]$Session = 'embeddibert-wiki727-qwen-pairs-20260909-r2',
    [string]$LogPath = 'E:\workspace\EmbeddiBERT\outputs\colab_continuous_training_r2.log',
    [string]$OutputDir = 'E:\workspace\EmbeddiBERT\outputs\continuous_wiki727',
    [string]$RemoteLog = ''
)

$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$watchLog = Join-Path $OutputDir 'watchdog.log'

function Write-WatchLog([string]$Message) {
    "$(Get-Date -Format o) $Message" | Add-Content -LiteralPath $watchLog
}

function ConvertTo-WslPath([string]$WindowsPath) {
    $drive = $WindowsPath.Substring(0, 1).ToLowerInvariant()
    return "/mnt/$drive/" + (($WindowsPath.Substring(3)) -replace '\\', '/')
}

Write-WatchLog "watchdog started for $Session"
while ($true) {
    if ($RemoteLog) {
        $localWsl = ConvertTo-WslPath $LogPath
        $downloadLog = "mighty-colab --auth=oauth2 download -s '$Session' '$RemoteLog' '$localWsl'"
        wsl -e bash -lc $downloadLog | Add-Content -LiteralPath $watchLog
        if ($LASTEXITCODE -ne 0) {
            Write-WatchLog 'remote log snapshot failed; will retry'
        }
    }
    $content = if (Test-Path -LiteralPath $LogPath) {
        Get-Content -LiteralPath $LogPath -Raw
    } else { '' }
    if ($content -match '"event": "complete"') { break }
    if ($content -match 'Traceback|RuntimeError|FloatingPointError|Incomplete .* cache') {
        Write-WatchLog 'failure marker found in training log'
        exit 2
    }
    Start-Sleep -Seconds 60
}

Write-WatchLog 'completion event found; downloading result artifacts'
$wslSession = $Session
$remoteFiles = @(
    @{ Remote = '/content/embeddibert_continuous/result.json'; Local = (Join-Path $OutputDir 'result.json') },
    @{ Remote = '/content/embeddibert_continuous/continuous_pair_distilbert.safetensors'; Local = (Join-Path $OutputDir 'continuous_pair_distilbert.safetensors') },
    @{ Remote = '/content/embeddibert_continuous/events.jsonl'; Local = (Join-Path $OutputDir 'events.jsonl') }
)
foreach ($file in $remoteFiles) {
    $localWsl = ConvertTo-WslPath $file.Local
    $download = "mighty-colab --auth=oauth2 download -s '$wslSession' '$($file.Remote)' '$localWsl'"
    Write-WatchLog "downloading $($file.Remote)"
    wsl -e bash -lc $download | Add-Content -LiteralPath $watchLog
    if ($LASTEXITCODE -ne 0) { Write-WatchLog "download failed: $($file.Remote)"; exit 3 }
}

$result = Get-Content -LiteralPath (Join-Path $OutputDir 'result.json') -Raw | ConvertFrom-Json
$actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $OutputDir 'continuous_pair_distilbert.safetensors')).Hash.ToLowerInvariant()
$expectedHash = $result.model.sha256.ToLowerInvariant()
$verification = [ordered]@{
    status = if ($actualHash -eq $expectedHash) { 'verified' } else { 'hash_mismatch' }
    expected_sha256 = $expectedHash
    actual_sha256 = $actualHash
    model_bytes = (Get-Item -LiteralPath (Join-Path $OutputDir 'continuous_pair_distilbert.safetensors')).Length
    result_status = $result.status
    epochs = $result.training.epochs.Count
    output_dir = $OutputDir
}
$verification | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $OutputDir 'verification.json')
Write-WatchLog "verification status: $($verification.status)"
