$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Create the environment first: python -m venv .venv, then install requirements.txt.'
}
if (-not (Test-Path -LiteralPath 'models\model_metadata.json')) {
    & $projectPython main.py
    if ($LASTEXITCODE -ne 0) { throw 'Model training failed. See the console output.' }
}
& $projectPython -m streamlit run dashboard/app.py --server.address localhost
