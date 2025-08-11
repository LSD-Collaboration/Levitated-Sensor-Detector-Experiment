# scripts/bootstrap.ps1  — Windows PowerShell version
# One-click project setup: create venv, install locked deps, install package, sanity check.

# Go to repo root (this script lives in <repo>\scripts)
$repoRoot = Resolve-Path "$PSScriptRoot\.."
Set-Location $repoRoot

# Create venv if missing
if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
  python -m venv .venv
}

# Activate venv for this shell
& .\.venv\Scripts\Activate.ps1

# Install exact, locked dependencies
python -m pip install -U pip
pip install --require-hashes -r requirements-dev.txt

# Install our package (editable)
pip install -e software\analysis-pipeline

# Sanity check (PowerShell-friendly, no heredocs)
python -c "import analysis_pipeline, sys; print('analysis_pipeline OK from:', analysis_pipeline.__file__); print('Python:', sys.executable)"
