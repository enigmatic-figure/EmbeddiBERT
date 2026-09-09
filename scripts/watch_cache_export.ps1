param(
    [string]$Session = 'embeddibert-r2-final-rescue',
    [string]$OutputDir = 'E:\workspace\EmbeddiBERT\outputs\wiki727_train_cache_backup'
)

$ErrorActionPreference = 'Stop'
$expectedEmbeddingBytes = 46971032064L
$expectedBase64Bytes = 62628042752L
$expectedSha256 = 'bf7943b17279d64cd4e3eca392d6d6550bcd27adb48f8ce6a6502a64f64bf5b9'
$streamScript = '/mnt/e/workspace/EmbeddiBERT/scripts/colab_stream_cache_file.py'
$watchLog = Join-Path $OutputDir 'cache_export_watchdog.log'
$embeddingBase64 = Join-Path $OutputDir 'embeddings.f16.b64'
$embeddingBinary = Join-Path $OutputDir 'embeddings.f16'

function Write-Watch([string]$Message) {
    "$(Get-Date -Format o) $Message" | Add-Content -LiteralPath $watchLog
}

function ConvertTo-WslPath([string]$WindowsPath) {
    $drive = $WindowsPath.Substring(0, 1).ToLowerInvariant()
    return "/mnt/$drive/" + (($WindowsPath.Substring(3)) -replace '\\', '/')
}

function Export-SmallCacheFile([string]$Name) {
    $base64Path = Join-Path $OutputDir "$Name.b64"
    $binaryPath = Join-Path $OutputDir $Name
    $base64Wsl = ConvertTo-WslPath $base64Path
    $binaryWsl = ConvertTo-WslPath $binaryPath
    $command = "mighty-colab --auth=oauth2 exec -s '$Session' -f '$streamScript' --timeout 300 --env EMBEDDIBERT_CACHE_FILE=$Name > '$base64Wsl'"
    Write-Watch "streaming $Name"
    wsl -e bash -lc $command
    if ($LASTEXITCODE -ne 0) { throw "$Name stream failed" }
    wsl -e bash -lc "base64 -d '$base64Wsl' > '$binaryWsl'"
    if ($LASTEXITCODE -ne 0) { throw "$Name decode failed" }
    Write-Watch "$Name decoded"
}

Write-Watch "watching embeddings stream; expected base64 bytes=$expectedBase64Bytes"
while ($true) {
    $length = if (Test-Path -LiteralPath $embeddingBase64) {
        (Get-Item -LiteralPath $embeddingBase64).Length
    } else { 0L }
    Write-Watch "base64 progress=$length/$expectedBase64Bytes"
    if ($length -ge $expectedBase64Bytes) { break }
    Start-Sleep -Seconds 30
}

Start-Sleep -Seconds 5
$embeddingBase64Wsl = ConvertTo-WslPath $embeddingBase64
$embeddingBinaryWsl = ConvertTo-WslPath $embeddingBinary
Write-Watch 'decoding embeddings.f16'
wsl -e bash -lc "base64 -d '$embeddingBase64Wsl' > '$embeddingBinaryWsl'"
if ($LASTEXITCODE -ne 0) { throw 'embedding decode failed' }
$actualBytes = (Get-Item -LiteralPath $embeddingBinary).Length
$actualSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $embeddingBinary).Hash.ToLowerInvariant()
if ($actualBytes -ne $expectedEmbeddingBytes -or $actualSha256 -ne $expectedSha256) {
    throw "embedding verification failed: bytes=$actualBytes sha256=$actualSha256"
}
Write-Watch "embeddings verified sha256=$actualSha256"

Export-SmallCacheFile 'labels.u8'
Export-SmallCacheFile 'valid.u8'

$manifest = [ordered]@{
    status = 'verified'
    source_session = $Session
    embeddings = [ordered]@{
        bytes = $actualBytes
        sha256 = $actualSha256
    }
    labels = [ordered]@{
        bytes = (Get-Item -LiteralPath (Join-Path $OutputDir 'labels.u8')).Length
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $OutputDir 'labels.u8')).Hash.ToLowerInvariant()
    }
    valid = [ordered]@{
        bytes = (Get-Item -LiteralPath (Join-Path $OutputDir 'valid.u8')).Length
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $OutputDir 'valid.u8')).Hash.ToLowerInvariant()
    }
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $OutputDir 'verification.json')
Write-Watch 'complete cache export verified'
