# Run once from PowerShell; all paths are explicit and no environment variables are changed.
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path $PSScriptRoot -Parent
$Python = Join-Path $ProjectRoot '.venv/Scripts/python.exe'
& $Python (Join-Path $PSScriptRoot 'setup_occupancy.py')
if ($LASTEXITCODE -ne 0) { throw 'Occupancy setup failed; see the preceding error.' }
