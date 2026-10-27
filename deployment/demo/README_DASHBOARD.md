# FAQ-Chatbot · Operations Dashboard

Visuelles Dashboard zur Überwachung von Qualität, Betrieb und Diagnose des RAG FAQ-Chatbots.

## Überblick

Das Dashboard zeigt Evaluationsmetriken aus der letzten Evaluation:
- **Service Quality** — Precision, Recall, F0.5 Score
- **Operational Health** — Latenz, Durchsatz, CPU/Memory
- **Problem Localization** — Wo entstehen Fehler?
- **Retrieval Potential** — Top-K Coverage
- **Pipeline Comparison** — Chroma vs. Reranker
- **Knowledge Base Health** — Datenqualitäts-Checks

## Voraussetzungen

Das Dashboard benötigt eine einzige Datei:
- **`metrics.json`** — Generiert durch die Evaluationspipeline (siehe `README_EVALUATION.md`)

## Installation

### 1. `metrics.json` bereitstellen

Lege die generierte `metrics.json` in den Ordner:
```
web/dashboard/metrics.json
```

**Woher bekomme ich `metrics.json`?**  
Die Datei wird durch die Evaluationspipeline erzeugt. Siehe [`README_EVALUATION.md`](../README_EVALUATION.md) für den genauen Ablauf.

### 2. Dashboard starten

**Über FastAPI (empfohlen):**
```bash
python -m uvicorn app.main:app
```
Öffne dann: <http://127.0.0.1:8000/dashboard>

**Alternative — Direkter File-Server:**
```bash
cd web/dashboard
python -m http.server 8080
```
Öffne dann: <http://127.0.0.1:8080/index.html>

## Nutzung

### Dashboard-Seitenstruktur

```
┌─────────────────────────────────────────┐
│ Gesamtbewertung (Verdict)               │  ← ACCEPTABLE / POOR / GOOD
├─────────────────────────────────────────┤
│ Knowledge Base Health                   │  ← Datenschicht (Datenqualität)
├─────────────────────────────────────────┤
│ Service Quality     │  Operational Health│  ← Chatbot
├─────────────────────────────────────────┤
│ Problem Localization│  Retrieval Potential│ ← Diagnose (ausklappbar)
│ Pipeline Comparison │  Problem Areas    │
├─────────────────────────────────────────┤
│ Technical Details                       │  ← Detailanalysen (ausklappbar)
└─────────────────────────────────────────┘
```

### Farb-Indikatoren

Jede Metrik hat einen Status:

| Farbe | Label | Bedeutung |
|---|---|---|
| 🟢 Grün | **OK** | Metrik erfüllt Anforderung |
| 🟡 Gelb | **ATTENTION** | Verbesserung empfohlen |
| 🔴 Rot | **CRITICAL** | Dringender Handlungsbedarf |
| ⚪ Grau | **-** | Keine Daten verfügbar |

### Gesamtbewertung (Verdict)

Oben im Dashboard erscheint eine Gesamtbewertung mit farbigem Balken links:

| Farbe | Rating | Wann? |
|---|---|---|
| 🔵 Blau | **EXCELLENT** | Precision ≥95%, F0.5 ≥90%, KB OK |
| 🟢 Grün | **GOOD** | Solide Qualität ohne kritische Probleme |
| 🟡 Gelb | **ACCEPTABLE** | Precision 85-95% oder KB-Probleme |
| 🔴 Rot | **POOR** | Precision <85% oder F0.5 <80% |
| ⚫ Schwarz | **BAD** | Dringender Handlungsbedarf |

**Beispiel:**  
Mit aktuellen Daten (Precision 95.51%, F0.5 89.66%, KB Status FAIL) erscheint **ACCEPTABLE** (gelber Balken), weil der KB-Status `FAIL` das Rating senkt.

### Service Quality 

**Was zeigt diese Sektion?**  
Endgültige Antwortqualität nach dem Answerability Gate.

| Metrik | Was bedeutet das? | Schwellen |
|---|---|---|
| **Precision** | Anteil korrekter Antworten | ≥95% OK, ≥85% ATTENTION |
| **Correctness Recall** | Fähigkeit, richtige Antworten zu finden | ≥85% OK, ≥70% ATTENTION |
| **F0.5 Score** | Gewichtung: Precision doppelt so wichtig wie Recall | ≥90% OK, ≥80% ATTENTION |

**Interpretation:**  
Wenn Precision hoch, aber Correctness Recall niedrig ist → Das System filtert zu strikt (zu viele False Negatives). Wenn Recall hoch, aber Precision niedrig → Zu viele falsche Antworten werden akzeptiert.

### Operational Health 

**Was zeigt diese Sektion?**  
Latenz, Durchsatz und Ressourcenverbrauch.

#### Latenz

| Metrik | Was bedeutet das? | Schwellen |
|---|---|---|
| **Avg** | Durchschnittliche Antwortzeit | ≤120ms OK, ≤240ms ATTENTION |
| **P95** | 95. Perzentil (langsamste 5%) | ≤265ms OK, ≤460ms ATTENTION |
| **Request Throughput** | Anfragen pro Sekunde | Keine Bewertung |

#### Ressourcen

| Metrik | Was bedeutet das? | Schwellen |
|---|---|---|
| **Chatbot CPU (%)** | Chatbot-Prozess-Auslastung | ≤60% OK, ≤85% ATTENTION |
| **Chatbot Memory (MB)** | Speicher des Chatbot-Prozesses | Keine Bewertung |
| **Ollama CPU (%)** | Ollama/Llama-Server-Auslastung | ≤60% OK, ≤85% ATTENTION |
| **Ollama Memory (MB)** | Speicher des Ollama-Prozesses | Keine Bewertung |

**Interpretation:**  
Hohe CPU bei niedriger Latenz → System ist effizient, aber ausgelastet. Hohe Latenz bei niedriger CPU → Möglicherweise I/O-Engpass oder Modell-Inferenz zu langsam.

### Diagnose (ausklappbar)

#### Error Attribution

**Balkendiagramm:** Wo gehen Antworten verloren?

| Komponente | Wann wird hier ein Fehler gezählt? |
|---|---|
| **Retrieval** | Korrekte FAQ nicht im Top-K Kandidaten |
| **Reranker** | Korrekte FAQ vorhanden, aber falsch gerankt |
| **Gate** | Falsche Aussortierung durch Threshold |

**Interpretation:**  
Hauptanteil bei Retrieval → Verbessere Embedding/Vectorstore. Hauptanteil bei Gate → Threshold zu streng oder Confidence schlecht kalibriert - hierbei können auch Verbeserungen des Rerankers potentiell unterstützen.

#### Retrieval Potential

**Top-K Coverage:** Wie oft ist die richtige FAQ überhaupt in den Kandidaten?

| Metrik | Was bedeutet das? |
|---|---|
| **Top-3** | Richtige FAQ unter Top-3 |
| **Top-5** | Richtige FAQ unter Top-5 |
| **Top-10** | Richtige FAQ unter Top-10 |

**Wichtig:** Wenn Top-5 hoch (z.B. 97%), aber Precision niedrig (z.B. 70%) → **Ranking ist das Problem**, nicht die Retrieval-Basis.

#### Pipeline Comparison

| Sektion | Was zeigt das? |
|---|---|
| **Stage Performance** | Genauigkeit von Retrieval und Reranker einzeln |
| **Gate Comparison** | Nutzen des Rerankers (Chroma vs. Reranker) |

**Gate Comparison Tabelle:**  
Drei Zeilen (Precision, Recall, F0.5) mit zwei Werten pro Zeile:
- Links: Nur Chroma (ohne Reranker)
- Rechts: Mit Reranker
- Pfeil ↑: Reranker verbessert die Metrik

#### Problem Areas

**Bottom 5 Kategorien:** Welche Fragetypen funktionieren am schlechtesten?

| Spalte | Bedeutung |
|---|---|
| **Name** | Kategorie (z.B. "paraphrase", "grenzfall_mehrdeutig") |
| **Erfolgsrate** | Anteil korrekter Antworten in dieser Kategorie |
| **Bar** | Farbcodiert (grün ≥85%, gelb ≥60%, rot <60%) |
| **Tests** | Anzahl der Tests in dieser Kategorie |

**Beispiel:**  
`grenzfall_mehrdeutig` mit 0% Erfolgsrate bei 2 Tests → Diese Kategorie muss dringend verbessert werden.

### Knowledge Base Health (Datenschicht)

**Was zeigt diese Sektion?**  
Datenqualität des Knowledge Base, unabhängig vom Chatbot.

| Metrik | Was bedeutet das? |
|---|---|
| **Duplicate Pairs** | Semantisch identische Einträge |
| **Affected Entries** | Anzahl betroffener Einträge |
| **Low Margin Entries** | Fälle mit geringer Trennschärfe (<0.1) |
| **Avg Margin** | Durchschnittliche Margin zwischen Rank-1 und Rank-2 |

**Status-Badge:**
- 🟢 OK: Keine Probleme erkannt
- 🟡 WARNING: Einige Probleme gefunden
- 🔴 FAIL: Kritische Probleme (z.B. Duplicate > Threshold)

**Beispiel:**  
Mit 3 Duplicate Pairs, 6 betroffenen Einträgen, Status `FAIL` → KB-Qualität ist problematisch, muss bereinigt werden.

### Technical Details (ausklappbar)

**Margin Analysis:**
- **Chroma Margin Separation** — Trennschärfe zwischen richtigen/falschen Antworten (nur Chroma)
- **Reranker Margin Separation** — Trennschärfe (nach Reranker)
- **Margin Separation Gain** — Verbesserung durch Reranker

**Gate Details:**
- **Gate Precision/Recall/F0.5** — Gate-Metriken ohne Reranker


## Metrik-Schwellenwerte

Alle Status-Indikatoren basieren auf diesen Schwellen:

| Metrik | GOOD | ATTENTION | CRITICAL | Richtung |
|---|---|---|---|---|
| **Precision** | ≥95% | ≥85% | <85% | ↑ höher besser |
| **Correctness Recall** | ≥85% | ≥70% | <70% | ↑ |
| **F0.5 Score** | ≥90% | ≥80% | <80% | ↑ |
| **Coverage (Top-K)** | ≥95% | ≥85% | <85% | ↑ |
| **Avg Latency** | ≤120ms | ≤240ms | >240ms | ↓ niedriger besser |
| **P95 Latency** | ≤265ms | ≤460ms | >460ms | ↓ |
| **Chatbot/Ollama CPU** | ≤60% | ≤85% | >85% | ↓ |

**Richtung erklären:**  
`↑` = Je höher, desto besser. `↓` = Je niedriger, desto besser.

## Häufige Fragen

### Warum ist meine Gesamtbewertung "ACCEPTABLE" statt "GOOD"?

Checke die Einzelmetriken:
- Precision <95%? → Senkt Rating auf ACCEPTABLE
- F0.5 <90%? → Senkt Rating auf ACCEPTABLE
- Knowledge Base Status ≠ OK? → Senkt Rating auf mindestens ACCEPTABLE

### Was bedeutet der Unterschied zwischen Precision und Correctness Recall?

| Metrik | Frage | Beispiel |
|---|---|---|
| **Precision** | Von allen beantworteten Fragen: Wie viele waren korrekt? | 95 von 100 Antworten richtig → 95% Precision |
| **Correctness Recall** | Von allen richtigen Antworten: Wie viele wurden gefunden? | 72 von 100 möglichen Antworten gefunden → 72% Recall |

**Trade-off:**  
Je strenger das Gate (hohe Precision), desto mehr Fragen werden abgelehnt (niedriger Recall). Die F0.5-Metrik gewichtet Precision doppelt so stark wie Recall, weil im Chatbot-Kontext falsche Antworten schlimmer sind als verpasste.
