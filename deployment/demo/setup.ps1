[CmdletBinding()]
param(
    [switch]$InstallOllama,
    [switch]$PullModels,
    [switch]$InitializeFaq
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
Push-Location $ProjectRoot

try {
    $PythonLauncher = Get-Command py -ErrorAction SilentlyContinue
    if (-not $PythonLauncher) {
        throw 'Python 3.14 with the Windows py launcher is required. Install Python 3.14, then rerun setup.ps1.'
    }

    & $PythonLauncher.Source -3.14 --version
    if ($LASTEXITCODE -ne 0) {
        throw 'Python 3.14 was not found. Install Python 3.14, then rerun setup.ps1.'
    }

    $Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path $Python)) {
        & $PythonLauncher.Source -3.14 -m venv .venv
        if ($LASTEXITCODE -ne 0) {
            throw 'Could not create the virtual environment.'
        }
    }

    & $Python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        throw 'Installing Python dependencies failed.'
    }

    if (-not (Test-Path '.env')) {
        Copy-Item '.env.example' '.env'
        Write-Host 'Created .env from .env.example.'
    } else {
        Write-Host 'Kept existing .env.'
    }

    New-Item -ItemType Directory -Path '.\data' -Force | Out-Null

    $OllamaCommand = Get-Command ollama -ErrorAction SilentlyContinue
    $OllamaPath = if ($OllamaCommand) { $OllamaCommand.Source } else { $null }
    if ($InstallOllama -and -not $OllamaPath) {
        $Winget = Get-Command winget -ErrorAction SilentlyContinue
        if (-not $Winget) {
            throw 'winget was not found. Install Ollama manually, then rerun setup.ps1.'
        }

        & $Winget.Source install --id Ollama.Ollama --exact --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) {
            throw 'Ollama installation failed or was cancelled.'
        }

        $OllamaCommand = Get-Command ollama -ErrorAction SilentlyContinue
        $OllamaPath = if ($OllamaCommand) { $OllamaCommand.Source } else { $null }
        if (-not $OllamaPath) {
            $Candidates = @(
                (Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'),
                (Join-Path $env:ProgramFiles 'Ollama\ollama.exe')
            )
            $OllamaPath = $Candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
        }

        if (-not $OllamaPath) {
            throw 'Ollama was installed. Open or restart Ollama, then rerun setup.ps1 -PullModels -InitializeFaq.'
        }
    }

    if ($PullModels -or $InitializeFaq) {
        if (-not $OllamaPath) {
            throw 'Ollama is not installed. Install it, then rerun setup.ps1 -PullModels -InitializeFaq.'
        }
    }

    if ($PullModels) {
        & $OllamaPath pull gemma:2b
        if ($LASTEXITCODE -ne 0) {
            throw 'Downloading the gemma:2b model failed. Make sure Ollama is running, then retry.'
        }

        & $OllamaPath pull nomic-embed-text
        if ($LASTEXITCODE -ne 0) {
            throw 'Downloading the nomic-embed-text model failed. Make sure Ollama is running, then retry.'
        }
    }

    if ($InitializeFaq) {
        $ImportCommand = 'import asyncio; from app.dependencies.db_update import update_service; asyncio.run(update_service.update_database())'
        & $Python -c $ImportCommand
        if ($LASTEXITCODE -ne 0) {
            throw 'FAQ indexing failed. Check that Ollama is running and both models are available.'
        }
    }

    Write-Host ''
    Write-Host 'Setup complete. Start the local demo with .\start.ps1.'
    if (-not $OllamaPath) {
        Write-Host 'Ollama is not installed. Install it, then run .\setup.ps1 -PullModels -InitializeFaq.'
    } elseif (-not $PullModels -or -not $InitializeFaq) {
        Write-Host 'After starting Ollama, run .\setup.ps1 -PullModels -InitializeFaq to download models and index the demo FAQ.'
    }
} finally {
    Pop-Location
}
