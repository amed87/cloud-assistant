# Evaluation und Hyperparameter-Tuning

Die automatische Evaluation dient dazu, Änderungen an Retrieval, Reranking und Answerability-Gate anhand fester Datensätze zu vergleichen. Sie ist von den Integration-Smoke-Tests getrennt.

## Voraussetzungen und Eingaben

- Aus dem Repository-Stamm `code` starten.
- Ollama muss laufen und die in `.env` eingestellten Chat-/Decision-Modelle sowie das Embedding-Modell bereitstellen.
- Der persistente Chroma-Index muss mit der zu evaluierenden FAQ-Version befüllt sein.
- Golden Dataset und synthetischer Lasttest liegen im übergeordneten `eval/data`:
  - `golden_dataset.csv` für fachliche RAG-Evaluation
  - `synthetic_load_workload.csv` für Last-/Latenztests

Die Evaluationsdaten liegen außerhalb des Git-Repositories. Prüfe vor jedem Lauf, ob Datensatz und Chroma-Index zueinander passen. Bewertungsänderungen oder Datenänderungen können Resultate unvergleichbar machen.

## Qualitäts-Evaluation

```powershell
python -m evaluation.evaluate --mode quality
```

Der Runner startet `evaluation.evaluate_rag`, schreibt detaillierte Ergebnisse und berechnet anschließend KPIs. Ausgabe wird im übergeordneten `eval/output` abgelegt. Alternativ lässt sich die Einzelauswertung direkt starten:

```powershell
python -m evaluation.evaluate_rag --output ..\eval\output\quality-results.csv
```

Die Evaluation betrachtet Retrieval, Reranker, Answerability-Gate und Gesamtergebnis. Erwartete FAQ-IDs werden anhand der Golden-Dataset-Einträge verglichen.

## Lasttests und KPI-Berichte

Der Lasttest kann den lokalen Uvicorn-Prozess starten und wieder beenden. Er ist in `evaluation/request_looper.py` auf die Windows-`.venv`-Struktur ausgelegt:

```powershell
python -m evaluation.evaluate --mode latency
```

Nur vorhandene Berichte neu zusammenfassen:

```powershell
python -m evaluation.evaluate --mode kpis
```

Oder den KPI-Generator direkt aufrufen:

```powershell
python -m evaluation.calculate_kpis `
    --dir ..\eval\output `
    --output ..\eval\output\kpis-manual.json
```

Die KPI-Auswahl verwendet jeweils die neueste passende Ergebnisdatei im Eingabeordner. Vermische daher keine Läufe unterschiedlicher Konfiguration, wenn du Metriken vergleichen möchtest.

## Gesamtablauf

```powershell
python -m evaluation.evaluate --mode full
```

Der Modus führt Quality, Latency/Resources und KPI-Berechnung nacheinander aus. Er benötigt Ollama, einen passenden Index und funktionierende Windows-Prozessstarts. Bei Fehlersuche die Modi einzeln laufen lassen.

## Hyperparameter-Tuning

```powershell
python -m evaluation.hyperparameter_tuning
```

Der Sweep braucht einen erreichbaren Ollama-Dienst und einen passenden Chroma-Index. Er berechnet Decision-Feature-Scores vorab und variiert Feature-Gewichte sowie Gate-Schwellen. Die Zahl der Kombinationen wächst mit jeder Prompt-Aufgabe exponentiell in der Anzahl der Gewichtskandidaten; auch die derzeitige Konfiguration kann daher sehr rechen- und zeitintensiv sein. Bei Änderungen an Kandidatenbereichen erst Kombinationenzahl prüfen und ggf. auf einem kleineren Bereich beginnen.

## Interpretation und Reproduzierbarkeit

- **Retrieval** prüft, ob der erwartete FAQ-Eintrag in den Kandidaten vorkommt.
- **Reranker** prüft dessen Position nach Feature-Bewertung.
- **Gate** prüft, ob Score und Abstand eine Antwort zulassen.
- **Overall** bewertet die End-to-End-Entscheidung.
- `rag_relevance` prüft nur den grundlegenden Pipeline-Ausgang. Es ist kein Schwellen- oder Qualitätsbenchmark.
- Vergleiche Ergebnisse nur mit gleicher FAQ-Version, gleichem Golden Dataset, gleicher Ollama-Modellversion und dokumentierter Konfiguration.
- Berichte unter `..\eval\output` sind lokale Artefakte und kein automatischer Qualitätsnachweis für eine andere Umgebung.

Für das Dashboard siehe [Dashboard-Anleitung](../deployment/demo/README_DASHBOARD.md).
