$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path '.venv/Scripts/python.exe')) {
    throw 'Run scripts/setup.ps1 first. See docs/user-guide.md.'
}
& .venv/Scripts/python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
if ($LASTEXITCODE) { throw 'The app stopped with an error. See docs/user-guide.md.' }
