# parameter_tuning

Optimiert Reranker-Gewichte und Answerability-Gate-Thresholds des FAQ-Chatbots
gegen das Golden-Set. Läuft nach jedem größeren Update der Wissensdatenbank –
Overfitting aufs aktuelle KB ist hier gewollt, da die Thresholds die aktuelle
Datenlage adaptieren sollen.

Ersetzt den früheren Grid-Sweep (`hyperparameter_tuning.py`), der bei
~15 Mio. Konfigurationen × 118 Cases × Gate-Aufrufen nicht mehr terminierte.

## Ansatz

Der Sweep ist zu groß, nicht das Problem selbst. Drei Ursachen werden direkt
adressiert:

1. **Gate-Thresholds werden nie gesampelt.** Das Gate prüft nur
    `top_score >= min_score` und `gap >= min_gap`. Für eine feste
    Reranker-Konfiguration kann man daher aus ~118 (top, gap)-Paaren die
    komplette Konfusionsmatrix über *alle* Threshold-Kombinationen analytisch
    ableiten (Landscape). Das Grid über min_score/min_gap entfällt komplett –
    und die Thresholds kommen aus den beobachteten Werten statt aus geratenen
    Grenzen.
2. **Feature-Scores sind konfigurationsunabhängig.** Intent liefert einen
    Faktor (Bonus / Penalty / neutral), Keyword einen Faktor, DM-Scores sind
    fix. Alle Matrizen werden einmal vorberechnet; eine Konfiguration ist nur
    noch ein gewichteter Summenzugriff (numpy), kein Reranker-Lauf.
3. **Optuna (TPE) statt Grid** für die Gewichtungsdimensionen. Pro Trial:
    Score-Matrix → Ranking → Landscape → bestes F0.5. Ein Trial kostet
    Millisekunden.
4. **Threshold-Kandidaten sind Mittelpunkte zwischen beobachteten Werten**
    statt der Werte selbst. Der Decision-Provider (Ollama) jittert Scores um
    ~±0.001–0.004 zwischen Prozessläufen (Thread-Scheduling ändert die
    Summationsreihenfolge der Floats); ein Threshold, der exakt auf einem
    beobachteten Score liegt, ist eine Messerkante – ein Grenzfall kippt
    accept/reject und verschiebt das gesamte F0.5 (beobachtet am
    12.11.2026: 0.904 vs. 0.897 bei identischen Settings). Mittelpunkte
    halten auf beiden Seiten halben lokalen Abstand Puffer.

## Phasen

| Phase | Modul | Was passiert |
|-------|-------|--------------|
| 0 – Cache | `cache.py` + `service.py` | Einmal Retrieval pro Golden-Case (`quick_retrieve(eval=True)`); sammelt Base-Scores, Intent-Faktoren, Keyword-Faktoren (je Normalisierung), DM-Scores pro Prompt **sowie Centroid-Match-Zustände je Facette**. Der einzige teure Teil (Ollama/Chroma). |
| 1 – Matrizen | `matrices.py` | Cases zu dichten numpy-Arrays stapeln (`(C, K, …)`). |
| 2 – Suche | `search.py` | Optuna-Study über Reranker-Gewichte; Gate-Parameter sind **nicht** im Suchraum. |
| 3 – Landscape | `landscape.py` | Analytische F0.5-Fläche über alle (min_score, min_gap)-Paare (Mittelpunkt-Grid); Cross-Check gegen das echte `AnswerabilityGate`. |
| 4 – Finalisierung | `finalize.py` | Sieger wählen, Plateau-Check, **Jitter-robuste Threshold-Auswahl**, Boundary-Warnings, Export. |

## Tuning-Gruppen konfigurieren

`tuning_config.yaml` (im code-Root, Default liegt neben der `.env`) legt fest,
welche Gruppen gesampelt werden:

    ```yaml
    tune:
      intent: true          # INTENT_WEIGHT/-BONUS/-PENALTY
      keyword: true         # KEYWORD_WEIGHT + KEYWORD_MATCH_NORMALIZATION
      decision_model: true  # ein Gewicht pro decision_model_prompts-Eintrag
      centroid: true        # CENTROID_FACET_WEIGHTS + MATCH_BONUS/-PENALTY
    ```

- `false` = Gruppe wird **nicht gesampelt**, sondern auf den Snapshot-Werten
  des Caches (frischer Cache: den aktuellen `.env`-Werten) festgehalten.
- Festgehaltene Gruppen erscheinen trotzdem vollständig in der
  `recommendation` des Reports — beim Übertragen in die `.env` wird also
  nichts aus Versehen auf einen alten Wert zurückgesetzt. `tuned_groups` im
  Report dokumentiert, was tatsächlich gesampelt wurde.
- Die Gate-Thresholds sind keine Gruppe: sie werden pro Trial aus der
  Threshold-Landschaft abgeleitet und stehen immer in der Empfehlung.
- **Centroid-Tuning:** Die Centroid-Parameter hängen an gecachten
  Match-Zuständen (Query-Facetten-Label ↔ Kandidaten-Label). Diese werden beim
  Cache-Build aus den aktiven `CentroidMatchFeature`s abgeleitet — Voraussetzung
  ist daher, dass Zentroide existieren (DB-Update gelaufen) und
  `CENTROID_MATCH_BONUS`/`_PENALTY > 0` sind. Der Cache-Build primes den
  Facetten-Klassifikator selbst, die Zustände sind also unabhängig von den
  aktuellen `CENTROID_FACET_WEIGHTS`.
- `CENTROID_CONFIDENCE_FLOOR`/`_MARGIN` werden **nicht** getunt (sie steuern
  die Diskretisierung der gecachten Zustände, nicht die Score-Gewichtung).
  Eine Änderung macht den Cache unbrauchbar → Neubau mit `--refresh-cache`
  nach dem Umstellen in der `.env`.

## Benutzung

    ```bash
    python -m parameter_tuning --trials 2000 \
        --study-db sqlite:///tuning.db \
        --output recommended_settings.json
    ```

Alle Flags optional mit den Defaults: `--trials` (2000), `--output`
(`recommended_settings.json`), `--study-db` (ohne: in-memory), `--seed`.

Ohne `--study-db` ist ein Lauf ein Wegwerf-Lauf. Mit SQLite-Storage lässt sich
die Studie fortsetzen (`load_if_exists=True`) und über Zeit vergleichen.
Das Beispiel benutzt eine eingerückte Codezeile statt eines Fenced Blocks –


## Output

**Terminal:** bestes F0.5, die empfohlenen Settings, Boundary-Warnings und
eine Warnung, falls das echte Gate von der analytischen Zählung abweicht.

**JSON-Report:** zusätzlich Confusion-Counts am Optimum, Plateau-Analyse
(wie flach ist die Threshold-Umgebung des Optimums?), Gate-Validierung und
Robustheit: Unter `robustness` simuliert der Report die beobachtete
Score-Jitter des Decision-Providers (Monte-Carlo: `TOP_JITTER`/`MARGIN_JITTER`
auf jedem Case, `ROBUST_REPEATS` Wiederholungen, fester Seed) und meldet
`f05_at_recommendation`, `simulated_mean_f05` und `simulated_worst_f05` am
empfohlenen Paar. Differieren mean/worst deutlich vom Nominalwert, sitzt die
empfohlene Schwelle trotzdem nahe an einem Grenzfall.

Die `recommendation`-Keys entsprechen 1:1 den `settings.*`-Namen
(INTENT_WEIGHT, INTENT_MATCH_BONUS, INTENT_MISMATCH_PENALTY, KEYWORD_WEIGHT,
KEYWORD_MATCH_NORMALIZATION, DECISION_MODEL_WEIGHTS, CENTROID_FACET_WEIGHTS,
CENTROID_MATCH_BONUS, CENTROID_MATCH_PENALTY, RAG_MIN_SCORE,
RAG_MIN_SCORE_GAP) und können direkt ins Config-Repository übernommen werden.

## Interpretation

- **Boundary-Warnings:** Liegt das Optimum an einer Range-Grenze, war die
    Range zu eng → in `search.py` (`FLOAT_RANGES`/`DM_RANGE`) erweitern und mit
    gleichem Storage weiteroptimieren. Damit ist die Frage „traue ich den
    Erfahrungswerten?" datenbasiert beantwortet.
- **Plateau:** Viele Threshold-Paare nahe am Bestwert ⇒ robuste Wahl.
    Nadelpunkt ⇒ Einstellung ist Zufall der Stichprobe, Vorsicht beim
    Übernehmen. Die empfohlenen `RAG_MIN_SCORE`/`RAG_MIN_SCORE_GAP` sind
    davon unabhängig immer das jitter-robusteste Plateau-Paar (höchster
    simulierter Mittelwert-F0.5), nicht der Nominal-Argmax – deshalb kann
    `f05_at_recommendation` geringfügig unter `best_f05` liegen; das ist
    gewollt (Trading der letzten Nachkommastelle gegen Stabilität).
- **Robustheit:** `simulated_worst_f05` ≈ `simulated_mean_f05` am empfohlenen
    Paar ⇒ die Thresholds vertragen die reale Score-Jitter des Providers.
    Große Lücke ⇒ nachbessern.
- **Gate-Validierung:** Muss immer übereinstimmen. Falls nicht: eine Annahme
    über das Gate stimmt nicht (siehe unten).

## Annahmen / Prüfpunkte

Diese Annahmen gehen in den Code ein und sollten bei Änderungen am Rest des
Systems einmal verifiziert werden:

1. Jeder Golden-Case liefert **genau 5 Chroma-Kandidaten**; der Reranker
    sortiert nur um (der Gap ist daher immer definiert).
2. Das Gate entscheidet ausschließlich über Score- und Gap-Schwellen (kein
    weiteres Kriterium in `evaluate_quick`).
3. `IntentFeature` gibt bei Mismatch einen **negativen** Faktor zurück, dessen
    Betrag die (positive) Penalty-Magnitude ist; die Detection in `cache.py`
    (`> 0` → Bonus, `< 0` → Penalty`) hängt daran — analog leitet der
    Centroid-Zustand aus dem Vorzeichen des Feature-Scores ab.
4. Fehlende Scores sind `0.0`, niemals `None`.
5. `rr.feature_scores` ist pro Kandidat-Id (mit Versionssuffix) und pro
    Prompt-Key getastet.
6. Die Reranker-Verdrahtung kommt aus `app.quality.ranking.feature_factory`
    — dieselbe Factory wie für Server und Evaluation. Direkte
    IntentFeature/KeywordFeature/DMSingleFeature-Instanziierung in anderen
    Modulen wäre ein Divergenz-Risiko für die Kalibrierungs-Prämisse.

## Abhängigkeiten

- `numpy` (ohnehin transitiv via ChromaDB vorhanden)
- `optuna` (pip, MIT-Lizenz)
- Rest wie das Bewertungssystem (App-Module; Ollama und ChromaDB müssen für
    Phase 0 laufen)

## Nicht im Umfang

Latency-/Resource-KPIs, Kategorie-Gewichtung der Objective, Holdout-Split
(bewusst weggelassen, da Overfitting aufs KB gewollt ist; den Golden-Set-Stand
daher im Report-Header mitloggen, sobald Datensätze versioniert werden).

## Retrigger-Regeln

Der persistente Cache (`--cache`, Default `eval/output/Tuning/tuning_case_cache.json`)
wird automatisch verworfen (Neubau mit frischen Ollama-Abrufen), sobald:

- die Golden-Rows sich ändern (Reihenfolge oder Fragetext),
- das Decision-Model-Prompt-Set sich ändert,
- Normalisierungswerte fehlen,
- der Cache vor der Centroid-Erweiterung gebaut wurde,
- `CENTROID_CONFIDENCE_FLOOR`/`_MARGIN` seit dem Build geändert wurden
  (die Match-Zustände sind gegen genau diese Werte diskretisiert).

Gewicht-Differenzen zwischen Snapshot und jetziger `.env` sind **kein**
Abbruch, sondern ein Hinweis: Kalibrierung und Festhaltung deaktivierter
Gruppen vergleichen gegen den Snapshot.

Nach jedem Update der Wissensdatenbank: `--refresh-cache` (neue Fragen, neue
Zentroiden → neue Match-Zustände).