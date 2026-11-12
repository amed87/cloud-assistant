# Entwicklung und Tests

## Arbeitsumgebung

Das Git-Repository ist `code`. Verwende Python 3.14 und führe Befehle aus dem Repository-Stamm aus, damit `.env` und relative Datenpfade erwartungsgemäß geladen werden.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

Direkte Abhängigkeiten werden in `requirements.in` und `requirements-dev.in` gepflegt; die `requirements*.txt`-Dateien enthalten gepinnte Abhängigkeiten. Nach einer Änderung der Input-Dateien die gepinnten Dateien mit pip-tools reproduzierbar aktualisieren und den Diff prüfen.

## Tests

### Unit-Tests mit Coverage

```powershell
python -m pytest tests\Unit --cov=app --cov-report=term-missing
```

Die Coverage-Konfiguration liegt in `.coveragerc` und betrachtet den Anwendungscode unter `app`. Coverage-Prozentwerte sind Orientierung, kein Ersatz für aussagekräftige Assertions. Tests sollen eigenes Verhalten und relevante Fehlerpfade prüfen, nicht triviale Weiterleitungen.

### Integrationstests

Chroma-Integration ohne Ollama:

```powershell
python -m pytest tests\Integration\test_vectorstore.py
```

RAG-Integrationen benötigen einen erreichbaren Ollama-Dienst und installierte Modelle; sie nutzen temporäre Chroma-Verzeichnisse:

```powershell
python -m pytest tests\Integration\test_rag_pipeline.py tests\Integration\test_rag_relevance.py
```

`test_rag_relevance.py` ist bewusst nur ein Smoke-Test: Die Pipeline muss eine plausible Fixture-Antwort oder den konfigurierten Fallback zurückgeben. Fachliche Vergleiche, Schwellenoptimierung und Qualitätsmetriken gehören zur Evaluation, nicht in diesen Integrationstest.

Alle Tests sammeln:

```powershell
python -m pytest --collect-only -q
```

Die Sammlung umfasste bei Erstellung dieser Anleitung 66 Tests. Der Bestand kann sich ändern; den aktuellen Testumfang immer mit dem Collect-Befehl prüfen.

## Entwicklungsregeln

- Neue Logik mit Unit-Tests für erfolgreiches Verhalten und relevante Fehlerpfade absichern.
- Bei Ollama-Abhängigkeiten in Unit-Tests Provider-Clients mocken; echte externe Aufrufe in Integration/Evaluation belassen.
- Keine Integrationstests, die von persönlichen `.env`-Werten, einem bestehenden produktiven Chroma-Index oder echten Nutzerdaten abhängen.
- Für Reranker-Änderungen die Zuordnung zwischen `decision_model_prompts` und `DECISION_MODEL_WEIGHTS` testen.
- Für CSV-/Import-Änderungen Pflichtspalten, optionale Felder, Versions-ID und Entfernung veralteter Einträge prüfen.
- Vor dem Abschluss mindestens die betroffenen Tests und `git diff --check` ausführen.

## Abhängigkeiten und lokale Dateien

- `.env`, `.venv`, `.coverage`, Python-Caches und `data` sind ausgeschlossen beziehungsweise sollen lokal bleiben.
- Die CSV-Quellen und Evaluationsergebnisse liegen teilweise im übergeordneten Ordner `Chatbot-Code`, außerhalb des Git-Repositories.
- Keine Zugangsdaten, internen FAQ-Inhalte oder lokalen Datenbankdateien committen.
- Anwendung, Middleware, Teams-Dummy und Streaming sind nicht alle gleichwertig priorisiert; Tests am aktuellen Projektumfang ausrichten und Abweichungen hier beziehungsweise im README aktualisieren.

## Pull-Request-/Übergabe-Checkliste

- Zweck und Auswirkung der Änderung beschreiben.
- Konfiguration, Datenformat oder API-Änderungen samt Migration dokumentieren.
- Relevante Unit- und Integrationstests ausführen und Ergebnisse nennen.
- Prüfen, ob Beispielkonfiguration und Dokumentation mit der Implementierung übereinstimmen.
- Keine generierten Evaluationsergebnisse oder vertraulichen Daten unbeabsichtigt hinzufügen.
