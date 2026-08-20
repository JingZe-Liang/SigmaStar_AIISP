param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
& $Python -m unittest discover -s (Join-Path $ProjectRoot "tests") -v

