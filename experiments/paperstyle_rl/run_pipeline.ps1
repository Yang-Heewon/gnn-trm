param(
    [ValidateSet("download", "preprocess", "embed", "phase1", "phase2", "test", "all")]
    [string]$Stage = "all",
    [switch]$SkipDownload
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ScriptPath = Join-Path $ScriptDir "run_pipeline.py"

$LaunchArgs = @($ScriptPath, "--stage", $Stage)
if ($SkipDownload) {
    $LaunchArgs += "--skip-download"
}

$Python = Get-Command python -ErrorAction SilentlyContinue
if ($Python) {
    & $Python.Source @LaunchArgs
    exit $LASTEXITCODE
}

$Py = Get-Command py -ErrorAction SilentlyContinue
if ($Py) {
    & $Py.Source -3 @LaunchArgs
    exit $LASTEXITCODE
}

Write-Error "python/py launcher not found."
exit 1
