param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("AN19", "X21", "B23")]
    [string]$Dataset,

    [Parameter(Mandatory = $true)]
    [string]$Store,

    [string[]]$Features = @(),
    [int]$Jobs = 8,
    [string]$Environment = "cross-talker-generalization"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RepositoryRoot = Split-Path -Parent $ProjectRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$Config = Join-Path $ProjectRoot "configs\project.json"
$Profile = Join-Path $ProjectRoot "configs\confirmatory.json"
$Derived = Join-Path $ProjectRoot "artifacts\derived"
$Models = Join-Path $ProjectRoot "artifacts\models"
$Figures = Join-Path $ProjectRoot "artifacts\figures"
$RunId = "$Dataset-$Store-confirmatory"
$PairDirectory = Join-Path $Derived "$Dataset-pairs"
$Folds = Join-Path $Derived "$Dataset-folds.csv"
$Distances = Join-Path $Derived "$RunId-distances.csv"
$Predictors = Join-Path $Derived "$RunId-predictors.csv"
$ModelInput = Join-Path $Derived "$RunId-model-input.csv"
$ModelOutput = Join-Path $Models $RunId
$Standardizers = Join-Path $Derived "$Store-standardizers"

function Invoke-Ctg {
    param([string[]]$Arguments)
    & conda run --no-capture-output -n $Environment python -m ctg.cli @Arguments
    if ($LASTEXITCODE -ne 0) { throw "ctg command failed with exit code $LASTEXITCODE" }
}

Invoke-Ctg @("build-pairs", "--project", $Config, "--dataset", $Dataset, "--output", $PairDirectory)
Invoke-Ctg @("make-folds", "--project", $Config, "--dataset", $Dataset, "--output", $Folds)

$NeedsScaling = $Store.EndsWith("_full") -or $Store.EndsWith("_acoustic")
if ($NeedsScaling) {
    $ScaleArgs = @("fit-standardizers", "--project", $Config, "--store", $Store,
        "--jobs", "$Jobs", "--output", $Standardizers)
    if ($Features.Count -gt 0) { $ScaleArgs += @("--features") + $Features }
    Invoke-Ctg $ScaleArgs
}

$DistanceArgs = @("compute-distances", "--project", $Config, "--profile", $Profile,
    "--pairs", (Join-Path $PairDirectory "pairs.csv"), "--store", $Store,
    "--jobs", "$Jobs", "--output", $Distances)
if ($Features.Count -gt 0) { $DistanceArgs += @("--features") + $Features }
if ($NeedsScaling) { $DistanceArgs += @("--standardizer-dir", $Standardizers) }
Invoke-Ctg $DistanceArgs

Invoke-Ctg @("aggregate", "--profile", $Profile, "--cells", (Join-Path $PairDirectory "cells.csv"),
    "--distances", $Distances, "--output", $Predictors)
Invoke-Ctg @("make-model-input", "--project", $Config, "--dataset", $Dataset,
    "--predictors", $Predictors, "--folds", $Folds, "--output", $ModelInput)
Invoke-Ctg @("fit-glmm-parallel", "--input", $ModelInput, "--output", $ModelOutput,
    "--jobs", "$Jobs")
Invoke-Ctg @("plot-profile", "--model-dir", $ModelOutput,
    "--output", (Join-Path $Figures "$RunId-profile"))
if ($Features.Count -ne 1) {
    Invoke-Ctg @("plot-distance-correlations", "--input", $Distances,
        "--output", (Join-Path $Figures "$RunId-distance-correlations"))
}

Write-Host "Completed $RunId"
