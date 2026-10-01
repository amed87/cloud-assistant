

# FAQ-Bot Demo für Windows

Dies ist ein lokales Demopaket. Halten Sie den Server auf localhost beschränkt: die Admin-Endpunkte sind nicht authentifiziert.

## Anforderungen

- Windows 10 oder 11
- Python 3.14 über den `py`-Launcher verfügbar
- Internetzugang für Python-Pakete und gegebenenfalls Ollama-Modelle
- Genügend Festplattenspeicher für die Python-Umgebung und Ollama-Modelle

## Einrichtung

Entpacken Sie das ZIP-Archiv, öffnen Sie PowerShell im entpackten Ordner und führen Sie Folgendes aus:

```powershell
.\setup.ps1
```

Die Einrichtung erstellt `.venv`, installiert die Laufzeit-Anforderungen, erstellt bei Bedarf `.env` aus `.env.example` und bewahrt eine bestehende `.env` auf. Die FAQ-Indizierung verwendet die vom Template konfigurierte mitgelieferte `data/faq.csv`.

Installieren Sie Ollama mit winget, falls es noch nicht installiert ist:

```powershell
.\setup.ps1 -InstallOllama
```

Öffnen Sie Ollama und stellen Sie sicher, dass der lokale Dienst läuft. Laden Sie dann die Modelle herunter und erstellen Sie den Demo-FAQ-Index:

```powershell
.\setup.ps1 -PullModels -InitializeFaq
```

Die Modelle sind `gemma:2b` und `nomic-embed-text`. Modell-Downloads sind optionale Einrichtungsschritte, da sie groß sein können.

## Starten

```powershell
.\start.ps1
```

Öffnen Sie <http://127.0.0.1:8000>. Die Demo verwendet ein lokales Chroma-Verzeichnis unter `data/chroma`. Stellen Sie den Server nicht ohne vorherige Absicherung der Admin-Endpunkte ins Netzwerk.