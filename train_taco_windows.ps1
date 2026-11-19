param(
    [string]$TacoDir = (Join-Path $PSScriptRoot 'data\TACO'),
    [int]$Epochs = 12
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath $TacoDir)) {
        git clone --depth 1 https://github.com/pedropro/TACO.git $TacoDir
    }
    if (-not (Test-Path -LiteralPath (Join-Path $TacoDir 'data\annotations.json'))) {
        Push-Location $TacoDir
        try { python .\download.py } finally { Pop-Location }
    }
    python (Join-Path $PSScriptRoot 'prepare_taco_crops.py') --root $TacoDir --output (Join-Path $PSScriptRoot 'data\taco_crops')
    python (Join-Path $PSScriptRoot 'train_classifier.py') (Join-Path $PSScriptRoot 'data\taco_crops') $Epochs
    Write-Host 'Model saved in models\waste.onnx and models\labels.txt. Review held-out test accuracy before deployment.'
} finally {
    Pop-Location
}
