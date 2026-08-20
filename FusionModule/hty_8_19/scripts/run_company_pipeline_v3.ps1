[CmdletBinding()]
param(
    [string]$Python = 'D:\AI\python.exe',
    [ValidateSet('cpu', 'cuda')]
    [string]$Device = 'cuda',
    [int]$Epochs = 6,
    [int]$StepsPerEpoch = 120
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$env:PYTHONPATH = Join-Path $root 'src'
$config = Join-Path $root 'configs\company.yaml'
$checkpoint = Join-Path $root 'outputs\checkpoints\joint_v3\best.pt'
$temp = Join-Path $env:TEMP 'dnr_fusion_delivery_v3_ascii'
New-Item -ItemType Directory -Path $temp -Force | Out-Null

function Invoke-Python {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

Invoke-Python @('-m', 'dnr_fusion.audit', '--config', $config)
Invoke-Python @('-m', 'dnr_fusion.train_joint_v3', '--config', $config, '--device', $Device, '--epochs', "$Epochs", '--steps-per-epoch', "$StepsPerEpoch")

foreach ($scene in @('645x', '128x')) {
    Invoke-Python @('-m', 'dnr_fusion.infer_v2', '--config', $config, '--scene', $scene, '--checkpoint', $checkpoint, '--device', $Device, '--rise-alpha', '0.08', '--fall-alpha', '1.0', '--overwrite')
    Invoke-Python @('-m', 'dnr_fusion.evaluate', '--config', $config, '--scene', $scene, '--device', $Device)
    Invoke-Python @('-m', 'dnr_fusion.stability', '--config', $config, '--scene', $scene, '--device', $Device)

    $video = Join-Path $temp "${scene}_comparison.mp4"
    $sheet = Join-Path $temp "${scene}_comparison_frame_0050.png"
    Invoke-Python @('-m', 'dnr_fusion.video', '--config', $config, '--scene', $scene, '--output', $video, '--contact-sheet', $sheet, '--overwrite')

    Copy-Item -LiteralPath $video -Destination (Join-Path $root "outputs\videos\${scene}_comparison.mp4") -Force
    Copy-Item -LiteralPath ($video -replace '\.mp4$', '.json') -Destination (Join-Path $root "outputs\videos\${scene}_comparison.json") -Force
    Copy-Item -LiteralPath $sheet -Destination (Join-Path $root "outputs\images\${scene}_comparison_frame_0050.png") -Force
}

Write-Host "Completed v3 pipeline. Final checkpoint: $checkpoint"
