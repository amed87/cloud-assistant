Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path $Python)) {
    throw 'Virtual environment is missing. Run .\setup.ps1 first.'
}

Push-Location $ProjectRoot
try {
    & $Python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
    if ($LASTEXITCODE -ne 0) {
        throw 'The demo server exited with an error.'
    }
} finally {
    Pop-Location
}
