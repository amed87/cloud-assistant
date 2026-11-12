# FAQ-Chatbot

Lokaler FAQ-Chatbot auf Basis einer RAG-Pipeline (Retrieval-Augmented Generation). Die Anwendung stellt eine FastAPI bereit, erstellt Embeddings und Decision-Scores mit Ollama und speichert FAQ-Vektoren in ChromaDB.

Dieses README ist der Einstieg für Betrieb und Weiterentwicklung. Detaillierte Anleitungen:

- [Einrichtung und Betrieb](docs/betrieb.md)
- [Konfiguration](docs/konfiguration.md)
- [FAQ-Wissensbasis](docs/wissensbasis.md)
- [Tests und Entwicklung](docs/entwicklung-und-tests.md)
- [Evaluation und Tuning](docs/evaluation.md)
- [Windows-Demo-Paket](deployment/demo/README.md)
- [Dashboard-Anleitung](deployment/demo/README_DASHBOARD.md)

## Status und Umfang

- Die Kernanwendung enthält Chat-, Quick-Chat-, FAQ-Import- und FAQ-Health-Check-Funktionen.
- Der normale Chat kombiniert RAG-Kontext mit einer Ollama-Chat-Antwort. Quick Chat gibt eine passende FAQ-Antwort oder einen Fallback zurück.
- Teams-Endpunkte sind ein Dummy und keine implementierte Teams-Integration.
- Middleware und Streaming sind vorhanden, aber nicht Teil des aktuell priorisierten Testszenarios.
- Die Admin-Endpunkte sind nicht authentifiziert. Die Anwendung darf deshalb nur lokal oder hinter einer geeigneten Zugriffskontrolle betrieben werden.
- Die Anwendung, Quell-FAQ und Evaluationsergebnisse liegen in unterschiedlichen Ordnern. Siehe [Betrieb und Datenablage](docs/betrieb.md#datenablage).

## Architektur in Kürze

1. FastAPI nimmt eine Frage entgegen (`POST /chat` oder `POST /chat/quick`).
2. Der Embedding-Provider erstellt über Ollama einen Vektor für die Frage.
3. ChromaDB liefert ähnliche FAQ-Kandidaten.
4. Der Reranker kombiniert Retrieval-Scores mit Intent-, Keyword- und benannten Decision-Model-Features.
5. Beim Quick Chat entscheidet das Answerability-Gate anhand von Score, Score-Abstand und vorhandenem Antworttext, ob die FAQ-Antwort ausgegeben wird.
6. Beim normalen Chat wird der RAG-Kontext mit dem Gesprächsverlauf an das Chat-Modell übergeben.

Die zentralen Implementierungen liegen in `app/services`, `app/quality`, `app/documents`, `app/embeddings` und `app/vectorstores`.

## Schnellstart für Entwicklung (Windows PowerShell)

Voraussetzung: Python 3.14, Git und ein lokal erreichbarer Ollama-Dienst. Für die RAG-Funktionen müssen die konfigurierten Chat-, Embedding- und Decision-Modelle in Ollama verfügbar sein.

Im Verzeichnis `code`:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

`.env.example` ist primär für das Deployment-Paket vorgesehen: Dort liegen FAQ-CSV und Chroma-Index unter `.\data`. Für einen Quellcode-Checkout müssen mindestens `FAQ_INPUT_FILE` und `CHROMA_PATH` an die lokale Ablage im übergeordneten Ordner angepasst werden (standardmäßig `..\data\DB-Input\Tabelle_Chatbotfragen.csv` und `..\data\chroma`). Prüfe außerdem die Ollama-Modellnamen. Die Anzahl der Decision-Gewichte muss der Anzahl der Einträge in `decision_model_prompts` entsprechen. Einzelheiten stehen in [Konfiguration](docs/konfiguration.md).

Starte die Anwendung aus dem Verzeichnis `code`, damit relative Pfade wie in der Beispielkonfiguration aufgelöst werden:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Danach:

- API-Dokumentation: <http://127.0.0.1:8000/docs>
- Portal: <http://127.0.0.1:8000/>
- Dashboard: <http://127.0.0.1:8000/dashboard>

Die Anwendung lädt vorhandene Vektoren aus ChromaDB. Zum Erstellen oder Aktualisieren des FAQ-Index siehe [FAQ-Wissensbasis](docs/wissensbasis.md).

## Nützliche Befehle

```powershell
# Unit-Tests mit Coverage für app/
python -m pytest tests\Unit --cov=app --cov-report=term-missing

# Chroma-Integrationstest (kein Ollama-Modell erforderlich)
python -m pytest tests\Integration\test_vectorstore.py

# Qualitäts-Evaluation auf dem Golden Dataset (Ollama und befüllter Chroma-Index erforderlich)
python -m evaluation.evaluate --mode quality
```

Weitere Test- und Evaluationsbefehle sowie externe Voraussetzungen sind in [Tests und Entwicklung](docs/entwicklung-und-tests.md) und [Evaluation und Tuning](docs/evaluation.md) beschrieben.

## Repository- und Datenlayout

Das Git-Repository ist der Ordner `code`. Der übergeordnete Ordner `Chatbot-Code` enthält lokale Daten und Evaluationsergebnisse, die nicht Teil dieses Repositories sind:

```text
Chatbot-Code/
├── code/                  # Git-Repository und Arbeitsverzeichnis der App
│   ├── app/
│   ├── docs/
│   ├── evaluation/
│   └── tests/
├── data/
│   ├── DB-Input/          # Quell-FAQ
│   └── chroma/            # persistenter Vektorindex
└── eval/
    ├── data/              # Golden Dataset und Lasttest-Workload
    └── output/            # Evaluationsergebnisse und KPI-Berichte
```

Pfade in `.env` sind relativ zum aktuellen Arbeitsverzeichnis. Für die hier dokumentierten Befehle ist das der Ordner `code`.

## Weiterführende Hinweise

- `requirements.in` und `requirements-dev.in` sind die direkten Abhängigkeitslisten; gepinnte Dateien sind `requirements.txt` und `requirements-dev.txt`.
- Die Beispielkonfigurationen müssen gemeinsam gepflegt werden. Insbesondere muss `.env` für jedes konfigurierte Decision-Prompt genau ein Gewicht enthalten.
- Inhaltliche deutsche Chattexte und Fallbacks gehören in `content_config.yaml` beziehungsweise in den System-Prompt; technische Konfigurationsschlüssel bleiben Englisch.
- Bitte keine lokalen `.env`-Dateien, Zugangsdaten, echten Nutzerdaten oder lokalen Chroma-Daten einchecken.
