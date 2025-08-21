# scripts/bootstrap.ps1 — one-click setup for contributors (hardened)

$ErrorActionPreference = "Stop"

# Go to repo root (this script lives in <repo>\scripts)
$repoRoot = Resolve-Path "$PSScriptRoot\.."
Set-Location $repoRoot

Write-Host "➡️  Ensuring virtual environment..." -ForegroundColor Cyan
if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
  # Prefer 'py' launcher if available
  $pyOk = $false
  try { py -V >$null 2>&1; if ($LASTEXITCODE -eq 0) { $pyOk = $true } } catch {}
  if ($pyOk) { py -3 -m venv .venv } else { python -m venv .venv }
}
& .\.venv\Scripts\Activate.ps1

Write-Host "➡️  Upgrading build tooling..." -ForegroundColor Cyan
python -m pip install -U pip setuptools wheel build hatchling pre-commit

# Requirements (dev preferred)
function Install-Reqs {
  param([string]$path)
  if (Test-Path $path) {
    Write-Host "➡️  Installing $path (with hashes)..." -ForegroundColor Cyan
    try {
      python -m pip install --require-hashes -r $path
    } catch {
      Write-Host "⚠️  Hash install failed; retrying without hashes..." -ForegroundColor Yellow
      python -m pip install -r $path
    }
    return $true
  }
  return $false
}

if (-not (Install-Reqs ".\requirements-dev.txt")) {
  if (-not (Install-Reqs ".\requirements.txt")) {
    Write-Host "ℹ️  No requirements file found; continuing." -ForegroundColor DarkYellow
  }
}

# Editable install: top-level or subpackage
$pkgInstalled = $false
if (Test-Path ".\pyproject.toml") {
  Write-Host "➡️  Installing project (editable, repo root)..." -ForegroundColor Cyan
  python -m pip install -e .
  $pkgInstalled = $true
} elseif (Test-Path ".\software\analysis-pipeline\pyproject.toml") {
  Write-Host "➡️  Installing project (editable, software/analysis-pipeline)..." -ForegroundColor Cyan
  python -m pip install -e software\analysis-pipeline
  $pkgInstalled = $true
} else {
  Write-Host "❌  No pyproject.toml found at repo root or software/analysis-pipeline." -ForegroundColor Red
  throw "Cannot perform editable install; verify your packaging layout."
}

# Pre-commit: install & first run
$gitOk = $false
try { git --version >$null 2>&1; if ($LASTEXITCODE -eq 0) { $gitOk = $true } } catch {}
$inRepo = $false
if ($gitOk) {
  git rev-parse --is-inside-work-tree >$null 2>&1
  if ($LASTEXITCODE -eq 0) { $inRepo = $true }
}

if ($gitOk -and $inRepo) {
  Write-Host "➡️  Installing pre-commit hook..." -ForegroundColor Cyan
  pre-commit install
  Write-Host "➡️  Validating pre-commit config..." -ForegroundColor Cyan
  pre-commit validate-config
  $hookPath = ".git/hooks/pre-commit"
  $hookExisted = Test-Path $hookPath
  if (-not $hookExisted) {
    Write-Host "➡️  Running pre-commit across repo (first setup)..." -ForegroundColor Cyan
    pre-commit run --all-files
    Write-Host "`nIf files were modified, commit the changes:" -ForegroundColor Yellow
    Write-Host "  git add -A && git commit -m `"Apply pre-commit formatting`"" -ForegroundColor Yellow
  }
} else {
  Write-Host "Skipping pre-commit: git not found or not a Git repo." -ForegroundColor DarkYellow
}

# Sanity import
Write-Host "➡️  Sanity import check..." -ForegroundColor Cyan
python - <<'PY'
import sys
print("Python:", sys.version.split()[0])
try:
    from analysis_pipeline.tdms_tools import TDMSDataset
    print("TDMSDataset import OK")
except Exception as e:
    print("TDMSDataset import FAILED:", e)
    raise
PY

# Detect-secrets baseline (optional)
if (-not (Test-Path ".\.secrets.baseline")) {
  Write-Host "➡️  Creating .secrets.baseline..." -ForegroundColor Cyan
  python -m pip install -U detect-secrets
  python - <<'PY'
import subprocess, re, pathlib
cmd = ["detect-secrets","scan","--exclude-files","data/manifests/.*"]
out = subprocess.check_output(cmd)
pathlib.Path(".secrets.baseline").write_bytes(out)
print("Wrote .secrets.baseline")
PY
  Write-Host "Created .secrets.baseline (review & commit it)." -ForegroundColor Yellow
}

Write-Host "`n✅ Bootstrap complete." -ForegroundColor Green
