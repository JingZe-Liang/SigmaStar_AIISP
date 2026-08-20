param(
    [string]$Python = "python",
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$Config = Join-Path $ProjectRoot "configs\company.yaml"

& $Python -m dnr_fusion.audit --config $Config

& $Python -m dnr_fusion.train --config $Config --fold A --device $Device
& $Python -m dnr_fusion.train --config $Config --fold B --device $Device

& $Python -m dnr_fusion.infer --config $Config --scene 645x `
    --checkpoint (Join-Path $ProjectRoot "outputs\checkpoints\fold_A\best.pt") `
    --device $Device --overwrite
& $Python -m dnr_fusion.infer --config $Config --scene 128x `
    --checkpoint (Join-Path $ProjectRoot "outputs\checkpoints\fold_B\best.pt") `
    --device $Device --overwrite

& $Python -m dnr_fusion.evaluate --config $Config --scene 645x --device $Device
& $Python -m dnr_fusion.evaluate --config $Config --scene 128x --device $Device

& $Python -m dnr_fusion.video --config $Config --scene 645x --overwrite
& $Python -m dnr_fusion.video --config $Config --scene 128x --overwrite

Write-Host "Pipeline complete: $ProjectRoot\outputs"

