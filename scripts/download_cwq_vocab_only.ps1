param(
    [string]$RogCwqDataset = "rmanluo/RoG-cwq",
    [string]$HfCacheDir = "",
    [string]$PythonBin = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$downloadScript = Join-Path $ScriptDir "download_data.ps1"

& $downloadScript -Dataset cwq -DataSource rog_hf -CwqVocabOnly -RogCwqDataset $RogCwqDataset -HfCacheDir $HfCacheDir -PythonBin $PythonBin
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "[done] CWQ vocab-only download complete"
Write-Host "  - data/CWQ/entities.txt"
Write-Host "  - data/CWQ/relations.txt"
