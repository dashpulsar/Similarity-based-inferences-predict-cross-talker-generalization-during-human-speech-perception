param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("AN19", "X21", "B23")]
    [string]$Dataset,

    [Parameter(Mandatory = $true)]
    [string]$Store,

    [string[]]$Features = @(),
    [string[]]$Measures = @(),
    [int]$Jobs = 8,
    [string]$Environment = "cross-talker-generalization"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$Config = Join-Path $ProjectRoot "configs\project.json"
$Profile = Join-Path $ProjectRoot "configs\confirmatory.json"
$Derived = Join-Path $ProjectRoot "artifacts\derived"
$Models = Join-Path $ProjectRoot "artifacts\models"
$Figures = Join-Path $ProjectRoot "artifacts\figures"
if ($Store -match "_hubert_(base|ft)_tsne$") {
    $VariantLabel = $Matches[1]
} else {
    $VariantLabel = $Store
}
$RunId = "$Dataset-HVE-$VariantLabel"
$Exposure = Join-Path $Derived "$Dataset-exposure"
$Folds = Join-Path $Derived "$Dataset-folds.csv"
$Variability = Join-Path $Derived "$RunId-values.csv"
$ModelInput = Join-Path $Derived "$RunId-model-input.csv"
$ModelOutput = Join-Path $Models $RunId
$Standardizers = Join-Path $Derived "$Store-standardizers"

function Invoke-Ctg {
    param([string[]]$Arguments)
    & conda run --no-capture-output -n $Environment python -m ctg.cli @Arguments
    if ($LASTEXITCODE -ne 0) { throw "ctg command failed with exit code $LASTEXITCODE" }
}

Invoke-Ctg @("build-exposure", "--project", $Config, "--dataset", $Dataset, "--output", $Exposure)
Invoke-Ctg @("make-folds", "--project", $Config, "--dataset", $Dataset, "--output", $Folds)

$NeedsScaling = $Store.EndsWith("_full") -or $Store.EndsWith("_acoustic")
if ($NeedsScaling) {
    $ScaleArgs = @("fit-standardizers", "--project", $Config, "--store", $Store,
        "--jobs", "$Jobs", "--output", $Standardizers)
    if ($Features.Count -gt 0) { $ScaleArgs += @("--features") + $Features }
    Invoke-Ctg $ScaleArgs
}

$VariabilityArgs = @("compute-variability", "--project", $Config, "--profile", $Profile,
    "--tasks", (Join-Path $Exposure "exposure_tasks.csv"),
    "--pools", (Join-Path $Exposure "exposure_pools.csv"), "--store", $Store,
    "--jobs", "$Jobs", "--output", $Variability)
if ($Features.Count -gt 0) { $VariabilityArgs += @("--features") + $Features }
if ($NeedsScaling) { $VariabilityArgs += @("--standardizer-dir", $Standardizers) }
Invoke-Ctg $VariabilityArgs

$InputArgs = @("make-variability-input", "--project", $Config, "--dataset", $Dataset,
    "--variability", $Variability, "--participant-pools", (Join-Path $Exposure "participant_pools.csv"),
    "--folds", $Folds, "--output", $ModelInput)
if ($Measures.Count -gt 0) { $InputArgs += @("--measures") + $Measures }
Invoke-Ctg $InputArgs
Invoke-Ctg @("fit-glmm-parallel", "--input", $ModelInput, "--output", $ModelOutput,
    "--jobs", "$Jobs", "--predictor-column", "predictor_value", "--direction", "1",
    "--term", "variability_z")
Invoke-Ctg @("plot-profile", "--model-dir", $ModelOutput,
    "--output", (Join-Path $Figures "$RunId-profile"))

Write-Host "Completed $RunId"
