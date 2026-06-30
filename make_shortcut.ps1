# Creates a double-click "Crate" launcher on the Desktop (no console window, UMAP icon).
# Run once:  powershell -NoProfile -ExecutionPolicy Bypass -File make_shortcut.ps1
$ErrorActionPreference = "Stop"
$dir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$pyw   = Join-Path $dir ".venv\Scripts\pythonw.exe"
$icon  = Join-Path $dir "assets\crate_icon.ico"
$app   = Join-Path $dir "app.py"
if (-not (Test-Path $pyw)) { throw "venv not found - run run.ps1 once to bootstrap it first." }

$desktop = [Environment]::GetFolderPath("Desktop")
$lnkPath = Join-Path $desktop "Crate.lnk"
$ws = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut($lnkPath)
$lnk.TargetPath       = $pyw
$lnk.Arguments        = "`"$app`""
$lnk.WorkingDirectory = $dir
$lnk.IconLocation     = $icon
$lnk.Description       = "Crate - DJ set prep"
$lnk.Save()
Write-Host "Created shortcut: $lnkPath"
