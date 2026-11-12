# Einrichtung und Betrieb

Diese Anleitung beschreibt die lokale Entwicklungs- und Betriebsumgebung des FAQ-Chatbots. Sie setzt Windows und PowerShell voraus; die Anwendung selbst ist eine Python/FastAPI-Anwendung.

## Voraussetzungen

- Python 3.14 und Git
- Ollama als lokal erreichbarer Dienst
- Die in `.env` konfigurierten Modelle:
  - Chatmodell aus `OLLAMA_MODEL`
  - Embedding-Modell `nomic-embed-text` (wird vom Embedding-Provider verwendet)
  - Decision-Modell aus `DECISION_MODEL`, standardmäßig `laya`
- Speicherplatz für die virtuelle Umgebung, Ollama-Modelle und den Chroma-Index

Modellnamen hängen von der installierten Ollama-Umgebung ab. Stelle sicher, dass alle verwendeten Modelle dort verfügbar sind, bevor du die RAG- oder Evaluationspfade startest.

## Installation

Im Repository-Verzeichnis `code`:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

Passe `.env` an die lokale Datenablage und Ollama-Installation an. Konfigurationsdetails und Pfadauflösung stehen in [Konfiguration](konfiguration.md).

Für ausschließlich Laufzeitbetrieb kann `requirements.txt` statt `requirements-dev.txt` installiert werden. Die Entwicklungsdatei enthält zusätzlich Test- und Dependency-Management-Werkzeuge.

## Start und Prüfung

Starte den Ollama-Dienst und anschließend aus `code`:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Die Anwendung ist dann lokal erreichbar:

- Chat-API: `POST /chat`
- Quick-Chat-API: `POST /chat/quick`
- API-Dokumentation: `/docs`
- Health-Endpoint: `GET /admin/health`

Beispiel für einen Quick-Chat-Aufruf in einer zweiten PowerShell:

```powershell
$body = @{
    conversation_id = "local-check"
    message = "Wie lautet meine Frage?"
} | ConvertTo-Json

Invoke-RestMethod -Uri http://127.0.0.1:8000/chat/quick -Method Post `
    -ContentType "application/json" -Body $body
```

## FAQ-Index aktualisieren und prüfen

Die Admin-Routen sind nicht authentifiziert und dürfen nicht ungeschützt im Netzwerk veröffentlicht werden.

- `POST /admin/update-db` startet einen FAQ-Import im Hintergrund. Die Antwort bestätigt nur den Start, nicht den erfolgreichen Abschluss.
- `POST /admin/faq-health-check` prüft die konfigurierte FAQ-Datei oder eine übergebene `file_path` in einem temporären Chroma-Store. Der Report enthält Self-Retrieval-, Margin-, Embedding- und Duplikatprüfungen.
- `GET /admin/db` gibt Datenbankinhalte zurück und sollte ebenfalls nur lokal beziehungsweise geschützt erreichbar sein.

Nutze vor dem Aktualisieren den Health Check und sichere bei kritischen Änderungen die persistente Chroma-Datenbank. Details zu CSV-Format und Versions-IDs stehen in [FAQ-Wissensbasis](wissensbasis.md).

## Datenablage

Im Standardlayout relativ zum Verzeichnis `code`:

| Daten | Standardpfad |
|---|---|
| Quell-FAQ | `..\data\DB-Input\Tabelle_Chatbotfragen.csv` |
| Persistenter Chroma-Index | `..\data\chroma` |
| Golden Dataset und Lasttest-Workload | `..\eval\data` |
| Evaluationsergebnisse und KPI-Berichte | `..\eval\output` |

Die Pfade können über `.env` geändert werden. Relative Pfade werden zum aktuellen Arbeitsverzeichnis aufgelöst; starte App und Evaluationsbefehle deshalb aus `code`, sofern die Pfade nicht angepasst wurden.

Für ein Backup die Anwendung anhalten und mindestens die Quell-FAQ sowie das konfigurierte Chroma-Verzeichnis sichern. Eine Wiederherstellung erfolgt durch Rückkopieren des Chroma-Verzeichnisses auf den konfigurierten Pfad. Prüfe danach mit dem Health Endpoint und einer bekannten FAQ-Anfrage, ob die Daten geladen werden.

### Abweichende Struktur des Demo-Deployments

Das Deployment-Skript erstellt ein eigenständiges Paket mit einer anderen Datenstruktur:

```text
FAQ-Bot-demo/
├── app/
├── data/
│   └── faq.csv
├── .env.example
├── content_config.yaml
├── setup.ps1
└── start.ps1
```

In diesem Paket verweist `.env.example` auf `./data/faq.csv` und `./data/chroma`. Diese Paketpfade dürfen nicht ungeprüft als Pfade für den Quellcode-Checkout übernommen werden. Der Paket-Builder verwendet die lokale `.env` im Repository, um die Quell-FAQ zu finden, und kopiert eine Demo-Kopie in das Paket. Prüfe vor Weitergabe die enthaltenen FAQs und Konfiguration.

Das Demo-Setup lädt `gemma:2b` und `nomic-embed-text`. Stelle zusätzlich sicher, dass das in `DECISION_MODEL` konfigurierte Modell in der lokalen Ollama-Installation verfügbar ist.

## Sicherheits- und Betriebsgrenzen

- Admin-Endpunkte besitzen aktuell keine Authentifizierung oder Autorisierung.
- Im Entwicklungsbetrieb auf `127.0.0.1` binden. Für Netzwerkbetrieb ist vorab eine geeignete Zugriffskontrolle und Netzwerkbegrenzung erforderlich.
- `.env` kann interne Pfade oder Zugangsdaten enthalten und ist nicht für Versionskontrolle bestimmt.
- Für Modellupdates, Datenupdates und Änderungen an Gewichten oder Prompts Ergebnisse vor und nach der Änderung mit derselben Evaluationsbasis vergleichen.
- Logging erfolgt über Python-Logging auf der konfigurierten Stufe `LOG_LEVEL`. Stacktraces und Providerfehler stehen in den Serverlogs.

## Windows-Demo-Paket

Ein separates, lokales Demo-ZIP wird aus den Dateien in `deployment` erstellt. Die Demo-Anleitung beschreibt Einrichtung und Start. Der Build benötigt eine lokale `.env` mit gültigem `FAQ_INPUT_FILE` und kopiert FAQ-Daten in das Paket. Vor Weitergabe unbedingt prüfen, dass keine vertraulichen FAQs oder Zugangsdaten enthalten sind.

```powershell
.\deployment\scripts\New-DemoPackage.ps1 -Version "demo"
```

Das Build-Skript erzeugt ein ZIP im übergeordneten Ordner `dist`. Die Demo ist für lokale Nutzung gedacht; ihre Admin-Routen sind ebenfalls ungeschützt.
