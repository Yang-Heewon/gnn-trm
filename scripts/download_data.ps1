param(
    [ValidateSet("all", "cwq", "webqsp")]
    [string]$Dataset = "all",
    [ValidateSet("rog_hf", "hf_rog", "rog")]
    [string]$DataSource = "rog_hf",
    [switch]$CwqVocabOnly,
    [string]$RogCwqDataset = "rmanluo/RoG-cwq",
    [string]$RogWebqspDataset = "rmanluo/RoG-webqsp",
    [string]$HfCacheDir = "",
    [string]$PythonBin = ""
)

$ErrorActionPreference = "Stop"

function Resolve-Python {
    param([string]$ExplicitPython)
    if ($ExplicitPython) {
        return @{ Exe = $ExplicitPython; Prefix = @() }
    }
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return @{ Exe = $python.Source; Prefix = @() }
    }
    $python3 = Get-Command python3 -ErrorAction SilentlyContinue
    if ($python3) {
        return @{ Exe = $python3.Source; Prefix = @() }
    }
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return @{ Exe = $py.Source; Prefix = @("-3") }
    }
    throw "Python not found. Set -PythonBin or install python/python3/py."
}

function Invoke-Python {
    param(
        [hashtable]$PythonSpec,
        [string[]]$Arguments
    )
    & $PythonSpec.Exe @($PythonSpec.Prefix) @Arguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

function Require-Path {
    param([string]$PathToCheck)
    if (-not (Test-Path -LiteralPath $PathToCheck -PathType Leaf)) {
        Write-Host "[missing] $PathToCheck"
        return $false
    }
    Write-Host "[ok] $PathToCheck"
    return $true
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

$pythonSpec = Resolve-Python -ExplicitPython $(if ($PythonBin) { $PythonBin } else { $env:PYTHON_BIN })
$dataDir = Join-Path $RepoRoot "data"
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $dataDir ".downloads") | Out-Null

$targetDataset = $Dataset.ToLowerInvariant()
$targetSource = $DataSource.ToLowerInvariant()

if ($CwqVocabOnly -and $targetDataset -ne "cwq") {
    throw "CwqVocabOnly requires -Dataset cwq"
}

Write-Host "[step] preparing RoG datasets from Hugging Face"
$prepArgs = @(
    "scripts/prepare_rog_hf_data.py",
    "--dataset", $targetDataset,
    "--cwq_name", $RogCwqDataset,
    "--webqsp_name", $RogWebqspDataset,
    "--out_root", $RepoRoot
)
if ($CwqVocabOnly) {
    $prepArgs += "--cwq_vocab_only"
}
if ($HfCacheDir) {
    $prepArgs += @("--cache_dir", $HfCacheDir)
}
Invoke-Python -PythonSpec $pythonSpec -Arguments $prepArgs

$status = $true
if ($targetDataset -in @("webqsp", "all")) {
    $status = (Require-Path (Join-Path $dataDir "webqsp/train.json")) -and $status
    $status = (Require-Path (Join-Path $dataDir "webqsp/dev.json")) -and $status
    $status = (Require-Path (Join-Path $dataDir "webqsp/test.json")) -and $status
    $status = (Require-Path (Join-Path $dataDir "webqsp/entities.txt")) -and $status
    $status = (Require-Path (Join-Path $dataDir "webqsp/relations.txt")) -and $status
}
if ($targetDataset -in @("cwq", "all")) {
    $status = (Require-Path (Join-Path $dataDir "CWQ/entities.txt")) -and $status
    $status = (Require-Path (Join-Path $dataDir "CWQ/relations.txt")) -and $status
    if (-not $CwqVocabOnly) {
        $status = (Require-Path (Join-Path $dataDir "CWQ/train_split.jsonl")) -and $status
        $status = (Require-Path (Join-Path $dataDir "CWQ/dev_split.jsonl")) -and $status
        $status = (Require-Path (Join-Path $dataDir "CWQ/test_split.jsonl")) -and $status
        $status = (Require-Path (Join-Path $dataDir "CWQ/embeddings_output/CWQ/e5/entity_ids.txt")) -and $status
        $status = (Require-Path (Join-Path $dataDir "CWQ/embeddings_output/CWQ/e5/relation_ids.txt")) -and $status
        $status = (Require-Path (Join-Path $dataDir "data/CWQ/test.json")) -and $status
    }
}

if (-not $status) {
    Write-Host "[done] RoG HF data preparation incomplete."
    exit 2
}

Write-Host "[done] RoG HF data is ready"
