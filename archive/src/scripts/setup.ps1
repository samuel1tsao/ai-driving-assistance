$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv venv --python 3.11 .venv
if ($LASTEXITCODE) { throw 'Virtual environment creation failed' }
uv pip install --python .venv/Scripts/python.exe torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE) { throw 'PyTorch installation failed' }
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
if ($LASTEXITCODE) { throw 'Dependency installation failed' }
