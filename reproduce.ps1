param(
    [Parameter(Mandatory = $true)][string]$DataDir,
    [string]$Python = "python",
    [string]$ResultsDir = "results_reproduced",
    [string]$ReportDir = "report_reproduced"
)
$ErrorActionPreference = "Stop"
function Invoke-PythonStage([string[]]$StageArguments) {
    & $Python @StageArguments
    if ($LASTEXITCODE -ne 0) { throw "Stage failed: $($StageArguments -join ' ')" }
}
Push-Location $PSScriptRoot
try {
    if ((Test-Path -LiteralPath $ResultsDir) -or (Test-Path -LiteralPath $ReportDir)) {
        throw "Choose new result and report directories to preserve existing experiment evidence."
    }
    foreach ($DataFile in @("train.csv", "test.csv", "sample_submission.csv", "data_description.txt")) {
        if (-not (Test-Path -LiteralPath (Join-Path $DataDir $DataFile))) { throw "Missing official input: $DataFile" }
    }
    Invoke-PythonStage @("-m", "pytest", "tests", "-q")
    Invoke-PythonStage @("src/run_experiments.py", "--data-dir", $DataDir, "--output-dir", $ResultsDir, "--stage", "develop", "--workers", "3", "--threads", "2")
    Invoke-PythonStage @("src/select_diverse_blend.py", "--data-dir", $DataDir, "--results", $ResultsDir)
    Invoke-PythonStage @("src/run_experiments.py", "--data-dir", $DataDir, "--output-dir", $ResultsDir, "--stage", "holdout")
    Invoke-PythonStage @("src/run_experiments.py", "--data-dir", $DataDir, "--output-dir", $ResultsDir, "--stage", "final")
    Invoke-PythonStage @("src/analyze_results.py", "--data-dir", $DataDir, "--results", $ResultsDir, "--report-dir", $ReportDir)
    Invoke-PythonStage @("src/diagnose_scaling.py", "--data-dir", $DataDir, "--results", $ResultsDir)
    Write-Output "Reproduction complete. Results are in $ResultsDir and figures in $ReportDir."
    Write-Output "This script does not upload predictions or replace the original dated Kaggle evidence."
}
finally { Pop-Location }
