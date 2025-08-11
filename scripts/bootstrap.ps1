# scripts/bootstrap.ps1 — one-click setup for contributors
# - Creates/activates .venv
# - Installs locked deps + package (editable)
# - Installs & wires up pre-commit (Black/Ruff), runs it once on first setup

$ErrorActionPreference = "Stop"

# Go to repo root (this script lives in <repo>\scripts)
$repoRoot = Resolve-Path "$PSScriptRoot\.."
Set-Location $repoRoot

# 1) Ensure venv
if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
  python -m venv .venv
}
& .\.venv\Scripts\Activate.ps1

# 2) Dependencies (locked)
python -m pip install -U pip
if (Test-Path ".\requirements-dev.txt") {
  python -m pip install --require-hashes -r requirements-dev.txt
} elseif (Test-Path ".\requirements.txt") {
  python -m pip install --require-hashes -r requirements.txt
}

# 3) Install package (editable)
python -m pip install -e software\analysis-pipeline

# 4) Pre-commit: install hook and run once on first setup
#    (Skips cleanly if not a Git repo or git not on PATH)
$gitOk = $false
try { git --version >$null 2>&1; if ($LASTEXITCODE -eq 0) { $gitOk = $true } } catch {}
$inRepo = $false
if ($gitOk) {
  git rev-parse --is-inside-work-tree >$null 2>&1
  if ($LASTEXITCODE -eq 0) { $inRepo = $true }
}
if ($gitOk -and $inRepo) {
  # ensure pre-commit is available (fast no-op if already installed)
  python -m pip install -U pre-commit
  $hookPath = ".git/hooks/pre-commit"
  $hookExisted = Test-Path $hookPath
  pre-commit install
  if (-not $hookExisted) {
    Write-Host "Running pre-commit once to format/lint the repo..." -ForegroundColor Cyan
    pre-commit run --all-files
    Write-Host "`nIf files were modified, commit the changes:" -ForegroundColor Yellow
    Write-Host "  git add -A && git commit -m `"Apply pre-commit formatting`"" -ForegroundColor Yellow
  }
} else {
  Write-Host "Skipping pre-commit: git not found or not a Git repo." -ForegroundColor DarkYellow
}

# 5) Sanity ping
python -c "import analysis_pipeline, sys; print('analysis_pipeline:', analysis_pipeline.__file__); print('Python:', sys.executable)"
