# Ensure we're at repo root
Set-Location $PSScriptRoot\..

# Create venv if missing
if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
  python -m venv .venv
}

# Activate venv (PATH is set to .venv\Scripts automatically)
.\.venv\Scripts\Activate.ps1

# Install exact, hashed deps
python -m pip install -U pip
pip install --require-hashes -r requirements-dev.txt

# Install your package (editable)
pip install -e software/analysis-pipeline

# Quick sanity
python -c "import analysis_pipeline, sys; print('OK from:', analysis_pipeline.__file__); print('Python:', sys.executable)"
