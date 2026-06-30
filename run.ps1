# run.ps1 — launch Crate. Double-click (or: powershell -ExecutionPolicy Bypass -File run.ps1).
# Bootstraps a local .venv on first run, then opens the app window.
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$py   = Join-Path $here '.venv\Scripts\pythonw.exe'   # pythonw = no console window
$pyc  = Join-Path $here '.venv\Scripts\python.exe'
$ok   = Join-Path $here '.venv\.deps-ok'              # written ONLY after a clean install

# Gate on a success sentinel, not on python.exe existing — a half-finished first install still
# leaves python.exe behind, which would make the next run skip setup and launch windowless with
# an ImportError nobody can see. The sentinel makes setup retry until it actually succeeds.
if (-not (Test-Path $ok)) {
  Write-Host "First run: creating .venv and installing dependencies..."

  $python = Get-Command python -ErrorAction SilentlyContinue
  if (-not $python) {
    Write-Host "ERROR: Python 3.11+ was not found on PATH. Install it from" -ForegroundColor Red
    Write-Host "       https://www.python.org/downloads/ (tick 'Add python.exe to PATH'), then re-run." -ForegroundColor Red
    Read-Host "Press Enter to close"; exit 1
  }
  $ver = (& python -c "import sys; print('%d.%d' % sys.version_info[:2])").Trim()
  if ([version]$ver -lt [version]'3.11') {
    Write-Host "ERROR: Crate needs Python 3.11+, but found $ver. Install a newer Python and re-run." -ForegroundColor Red
    Read-Host "Press Enter to close"; exit 1
  }

  if (-not (Test-Path $pyc)) { python -m venv (Join-Path $here '.venv') }
  & $pyc -m pip install --upgrade pip
  & $pyc -m pip install -r (Join-Path $here 'requirements.txt')
  if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: dependency install failed (see the pip output above)." -ForegroundColor Red
    Write-Host "       Fix the cause (usually network), then re-run — setup will retry automatically." -ForegroundColor Red
    Read-Host "Press Enter to close"; exit 1
  }
  New-Item -ItemType File -Path $ok -Force | Out-Null
  Write-Host "Setup complete."
}

# Launch detached so the terminal/double-click returns immediately.
Start-Process -FilePath $py -ArgumentList (Join-Path $here 'app.py') -WorkingDirectory $here
