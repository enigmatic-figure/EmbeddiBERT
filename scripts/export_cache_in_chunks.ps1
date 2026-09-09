param(
    [string]$Session = 'embeddibert-r2-final-rescue',
    [string]$OutputDir = 'E:\workspace\EmbeddiBERT\outputs\wiki727_train_cache_backup',
    [long]$ExpectedBytes = 46971032064L,
    [long]$ChunkBytes = 267386880L
)

$ErrorActionPreference = 'Stop'
$streamScript = '/mnt/e/workspace/EmbeddiBERT/scripts/colab_stream_cache_file.py'
$base64Path = Join-Path $OutputDir 'embeddings.f16.b64'
$transferLog = Join-Path $OutputDir 'chunked_transfer.log'
$drive = $base64Path.Substring(0, 1).ToLowerInvariant()
$base64Wsl = "/mnt/$drive/" + (($base64Path.Substring(3)) -replace '\\', '/')
$chunkWsl = "/mnt/$drive/" + (($OutputDir.Substring(3)) -replace '\\', '/') + '/current_chunk.b64'

function Write-Transfer([string]$Message) {
    "$(Get-Date -Format o) $Message" | Tee-Object -FilePath $transferLog -Append | Write-Host
}

$encodedLength = (Get-Item -LiteralPath $base64Path).Length
$alignedEncodedLength = $encodedLength - ($encodedLength % 4)
if ($alignedEncodedLength -ne $encodedLength) {
    wsl -e bash -lc "truncate -s $alignedEncodedLength '$base64Wsl'"
}
$offset = [long](($alignedEncodedLength / 4) * 3)
if ($offset % 3 -ne 0) { throw "Resume offset is not divisible by three: $offset" }
Write-Transfer "resuming at raw byte $offset/$ExpectedBytes"

while ($offset -lt $ExpectedBytes) {
    $length = [Math]::Min($ChunkBytes, $ExpectedBytes - $offset)
    $expectedEncoded = [long](4 * [Math]::Ceiling($length / 3.0))
    $command = "mighty-colab --auth=oauth2 exec -s '$Session' -f '$streamScript' --timeout 900 --env EMBEDDIBERT_CACHE_FILE=embeddings.f16 --env EMBEDDIBERT_CACHE_OFFSET_BYTES=$offset --env EMBEDDIBERT_CACHE_LENGTH_BYTES=$length > '$chunkWsl'"
    Write-Transfer "requesting offset=$offset length=$length"
    wsl -e bash -lc $command
    if ($LASTEXITCODE -ne 0) { throw "Remote chunk failed at offset $offset" }
    $chunkPath = Join-Path $OutputDir 'current_chunk.b64'
    $actualEncoded = (Get-Item -LiteralPath $chunkPath).Length
    if ($actualEncoded -ne $expectedEncoded) {
        throw "Chunk length mismatch at offset ${offset}: $actualEncoded != $expectedEncoded"
    }
    wsl -e bash -lc "cat '$chunkWsl' >> '$base64Wsl'"
    if ($LASTEXITCODE -ne 0) { throw "Appending chunk failed at offset $offset" }
    $offset += $length
    $percent = [Math]::Round(100 * $offset / $ExpectedBytes, 2)
    Write-Transfer "committed raw byte $offset/$ExpectedBytes ($percent%)"
}

Write-Transfer 'all embedding chunks transferred'
