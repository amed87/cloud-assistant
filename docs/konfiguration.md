# Konfiguration

Die Laufzeitkonfiguration kommt aus Umgebungsvariablen und der `.env`-Datei. Das Modell ist in `app/config/settings.py` definiert. `content_config.yaml` enthält regel- und textbezogene Einstellungen. Starte die App aus `code`, damit relative Pfade konsistent aufgelöst werden.

## `.env`

`.env.example` wird beim Erstellen des Windows-Demo-Pakets mitgegeben und enthält deshalb Paketpfade. Für eine Entwicklungs-Checkout erstellst du `.env` daraus und passt mindestens `FAQ_INPUT_FILE` und `CHROMA_PATH` an die lokale Datenablage an. `.env` wird nicht eingecheckt.

| Variable | Bedeutung | Aktueller Standard |
|---|---|---|
| `APP_NAME` | Titel der FastAPI-Anwendung | `FAQ-Bot` |
| `OLLAMA_HOST` | Ollama-Serveradresse | `http://localhost:11434` |
| `OLLAMA_MODEL` | Modell für normale Chat-Antworten | `gemma:2b` |
| `OLLAMA_NUM_THREADS` | Threadanzahl für Modellaufrufe | `4` |
| `OLLAMA_NUM_CTX` | Kontextgröße der Modellaufrufe | `2048` |
| `LOG_LEVEL` | Python-Logstufe | `INFO` |
| `FAQ_INPUT_FILE` | Eingabe-CSV für FAQ-Import und Health Check | `../data/DB-Input/Tabelle_Chatbotfragen.csv` |
| `CONTENT_CONFIG_FILE` | YAML-Datei für Texte, Filter und Prompts | `./content_config.yaml` |
| `NUMBER_OF_TURNS` | Maximale Gesprächshistorie in Turns | `5` |
| `CHROMA_PATH` | Persistenter Chroma-Speicher | `../data/chroma` |
| `CHROMA_COLLECTION` | Name der persistenten Chroma-Collection | `faq-bot` |
| `CHROMA_TEMP_PATH` | Präfix/Pfad für temporäre QA-Collections | `./temp_qa_db` |
| `RAG_TOP_K` | Anzahl der Kandidaten für RAG-Kontext | `5` |
| `RAG_MIN_SCORE` | Mindestscore für regulären RAG-Kontext | `0.3` |
| `RAG_MIN_SCORE_GAP` | Mindestabstand der Quick-Chat-Top-Kandidaten | `0.1` |
| `INTENT_WEIGHT` | Gewicht des Intent-Features | `1.0` |
| `INTENT_MATCH_BONUS` | Bonus für übereinstimmenden Fragetyp | `0.15` |
| `INTENT_MISMATCH_PENALTY` | Malus für abweichenden Fragetyp | `0.1` |
| `KEYWORD_WEIGHT` | Gewicht des Keyword-Features | `0.0` |
| `KEYWORD_MATCH_NORMALIZATION` | Normalisierung für Keyword-Treffer | `5.0` |
| `DECISION_MODEL` | Ollama-Decision-Modell | `laya` |
| `DECISION_MODEL_WEIGHTS` | Ein Gewicht je `decision_model_prompts`-Eintrag | `[0.3]` im Code-Default |
| `ENABLE_PROFILER` | Profiler aktivieren | `true` im Code-Default |
| `PROFILER_SAMPLE_RATE` | Anteil der profilierten Requests | `0.1` |
| `PROFILER_OUTPUT_PATH` | Zieldatei für Profiler-Ausgabe | `./load-profiler.html` |
| `PROFILER_OUTPUT_FORMAT` | Ausgabeformat | `html` |
| `PROFILER_RUN_ID` | Optionale ID für einen Profiling-Lauf | nicht gesetzt |

Die tatsächlichen Laufzeitwerte können durch `.env` oder Prozess-Umgebungsvariablen überschrieben werden. Im Demo-Paket sind `FAQ_INPUT_FILE=./data/faq.csv` und `CHROMA_PATH=./data/chroma` passend zur Paketstruktur. Für den Quellcode-Checkout sind im Standardlayout typischerweise `FAQ_INPUT_FILE=../data/DB-Input/Tabelle_Chatbotfragen.csv` und `CHROMA_PATH=../data/chroma` erforderlich. Überprüfe die Werte nach Änderungen an den Prompt-Einträgen.

### Decision-Model-Gewichte

`DECISION_MODEL_WEIGHTS` wird als JSON-Zahlenliste geschrieben. Anzahl und Reihenfolge müssen mit den Einträgen in `decision_model_prompts` übereinstimmen. Das erste Gewicht gehört zum ersten YAML-Eintrag, das zweite zum nächsten usw.

```dotenv
DECISION_MODEL_WEIGHTS=[0.3,0.3,0.3,0.3]
```

Bei abweichender Anzahl bricht die Abhängigkeitskonfiguration mit einem `ValueError` ab. Beim Hinzufügen, Entfernen oder Umsortieren eines Decision-Prompts deshalb auch `.env` und die Deployment-Vorlage `.env.example` anpassen und die Reranker-Tests ausführen.

## `content_config.yaml`

Die derzeit von der App verwendeten Schlüssel sind:

| Schlüssel | Zweck |
|---|---|
| `standard_fallback` | Quick-Chat-Text bei keiner akzeptierten FAQ-Antwort |
| `fragetyp_filter` | Geordnete Regex-Muster für die Intent-/Fragetyp-Erkennung |
| `fragetyp_fallback` | Fragetyp, wenn kein Filter passt |
| `decision_model_prompts` | Benannte Decision-Aufgaben; Reihenfolge bestimmt die Gewichtszuteilung |

Fallbacks und Texte, die im Chat erscheinen, sollten auf Deutsch formuliert sein. Decision-Prompts sind Anweisungen an das Modell und sollten sprachlich zu den Nutzerfragen passen.

**Hinweis zum aktuellen YAML:** Die Datei enthält zusätzlich `keine_antwort`, `decision_model_prompt` und `alt`. Diese Schlüssel werden vom aktuellen Anwendungscode nicht ausgewertet; für die Decision-Features ist `decision_model_prompts` maßgeblich. Verlasse dich nicht auf diese übrigen Felder, bevor sie im Code angebunden sind.

Regex-Filter werden in ihrer YAML-Reihenfolge geprüft. Nach Änderungen Beispiele für Treffer und Nicht-Treffer testen, da ein früher passender Filter die spätere Klassifikation beeinflussen kann.

## Nach Änderungen

- Änderungen an `.env` erfordern einen Neustart des App-Prozesses.
- Änderungen an `content_config.yaml` erfordern ebenfalls einen Neustart; Konfigurationen werden gecacht.
- Nach Änderungen an `FAQ_INPUT_FILE` den FAQ Health Check auf genau diese Datei ausführen, bevor der persistente Index aktualisiert wird.
- Einstellungen und Prompts anhand der [Evaluation](evaluation.md) vergleichen, statt nur einzelne Antworten zu beurteilen.
