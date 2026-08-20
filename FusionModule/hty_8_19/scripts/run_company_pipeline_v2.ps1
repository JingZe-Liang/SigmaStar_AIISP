param(
    [string]$Python = "D:\AI\python.exe",
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$Config = Join-Path $ProjectRoot "configs\company.yaml"
$Checkpoint = Join-Path $ProjectRoot "outputs\checkpoints\joint_v2\best.pt"
$OutputRoot = Join-Path $ProjectRoot "outputs"

Push-Location $ProjectRoot
try {
    & $Python -m dnr_fusion.audit --config $Config
    & $Python -m dnr_fusion.train_joint --config $Config --device $Device

    foreach ($Scene in @("645x", "128x")) {
        & $Python -m dnr_fusion.infer_v2 --config $Config --scene $Scene `
            --checkpoint $Checkpoint --device $Device --rise-alpha 0.08 --fall-alpha 1.0 --overwrite
        & $Python -m dnr_fusion.evaluate --config $Config --scene $Scene --device $Device
        & $Python -m dnr_fusion.stability --config $Config --scene $Scene --device $Device
    }

    # OpenCV builds on Windows may not write directly to a path containing Chinese characters.
    $VideoTempRoot = Join-Path ([System.IO.Path]::GetTempPath()) "dnr_fusion_v2_video"
    New-Item -ItemType Directory -Force -Path $VideoTempRoot | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $OutputRoot "videos"), (Join-Path $OutputRoot "images") | Out-Null
    foreach ($Scene in @("645x", "128x")) {
        $TempVideo = Join-Path $VideoTempRoot "$Scene`_comparison.mp4"
        $TempSheet = Join-Path $VideoTempRoot "$Scene`_comparison_frame_0050.png"
        & $Python -m dnr_fusion.video --config $Config --scene $Scene `
            --output $TempVideo --contact-sheet $TempSheet --overwrite
        Copy-Item -LiteralPath $TempVideo -Destination (Join-Path $OutputRoot "videos\$Scene`_comparison.mp4") -Force
        Copy-Item -LiteralPath ($TempVideo -replace '\.mp4$', '.json') -Destination (Join-Path $OutputRoot "videos\$Scene`_comparison.json") -Force
        Copy-Item -LiteralPath $TempSheet -Destination (Join-Path $OutputRoot "images\$Scene`_comparison_frame_0050.png") -Force
    }

    & $Python -m unittest discover -s (Join-Path $ProjectRoot "tests") -v
    Write-Host "Pipeline complete: $OutputRoot"
}
finally {
    Pop-Location
}
