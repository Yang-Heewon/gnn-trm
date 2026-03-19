param(
    [string]$RogCwqDataset = "rmanluo/RoG-cwq",
    [string]$HfCacheDir = "",
    [string]$PythonBin = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$downloadScript = Join-Path $ScriptDir "download_data.ps1"

& $downloadScript -Dataset cwq -DataSource rog_hf -RogCwqDataset $RogCwqDataset -HfCacheDir $HfCacheDir -PythonBin $PythonBin
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "[done] CWQ HF download complete"
Write-Host "  - data/CWQ/train_split.jsonl"
Write-Host "  - data/CWQ/dev_split.jsonl"
Write-Host "  - data/CWQ/test_split.jsonl"
Write-Host "  - data/CWQ/entities.txt"
Write-Host "  - data/CWQ/relations.txt"
Write-Host "  - data/CWQ/embeddings_output/CWQ/e5/entity_ids.txt"
Write-Host "  - data/CWQ/embeddings_output/CWQ/e5/relation_ids.txt"
