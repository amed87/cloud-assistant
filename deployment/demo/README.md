# FAQ-Bot Demo for Windows

This is a local demo package. Its FAQ data is copied from the `FAQ_INPUT_FILE` configured in the repository `.env`; the package itself contains only a sanitized `data/faq.csv`, never the `.env`. The FAQ that discloses the actual WLAN password is excluded. Keep the server bound to localhost: the admin endpoints are not authenticated.

## Requirements

- Windows 10 or 11
- Python 3.14 available through the `py` launcher
- Internet access for Python packages and, optionally, Ollama models
- Enough disk space for the Python environment and Ollama models

## Setup

Extract the ZIP, open PowerShell in the extracted folder, and run:

```powershell
.\setup.ps1
```

The setup creates `.venv`, installs the locked runtime requirements, creates `.env` from `.env.example` if needed, and preserves an existing `.env`. FAQ indexing uses the packaged `data/faq.csv` configured by the template.

Install Ollama with winget if it is not already installed:

```powershell
.\setup.ps1 -InstallOllama
```

Open Ollama and make sure its local service is running. Then download the models and build the demo FAQ index:

```powershell
.\setup.ps1 -PullModels -InitializeFaq
```

The models are `gemma:2b` and `nomic-embed-text`. Model downloads are optional setup steps because they can be large.

## Run

```powershell
.\start.ps1
```

Open <http://127.0.0.1:8000>. The demo uses a local Chroma directory under `data/chroma`. Do not expose the server to a network without first securing the admin endpoints.
